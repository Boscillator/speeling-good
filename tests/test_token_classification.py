from typing import cast

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import DataCollatorForTokenClassification, PreTrainedTokenizerFast

from speeling_good.dataloaders import prepare_dataset
from speeling_good.dataloaders.token_classification import (
    ValidationCollator,
    chunk_boundaries,
)
from speeling_good.evaluation import evaluated_word_count


@pytest.fixture
def tokenizer():
    vocabulary = [
        "[UNK]",
        "[PAD]",
        "[CLS]",
        "[SEP]",
        "🐈",
        "spe",
        "##ling",
        "okay",
        "tail",
    ]
    backend = Tokenizer(
        WordPiece({token: index for index, token in enumerate(vocabulary)})
    )
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=512,
    )


def test_unicode_subwords_chunking_and_batch_padding(tokenizer):
    rows = Dataset.from_list(
        [
            {
                "text": "🐈 speling okay tail",
                "annotations": [
                    {
                        "original": "speling",
                        "corrected": "spelling",
                        "span_start": 2,
                        "span_end": 9,
                    },
                    {
                        "original": "tail",
                        "corrected": "tale",
                        "span_start": 15,
                        "span_end": 19,
                    },
                ],
            },
            {"text": "okay", "annotations": []},
        ]
    )

    dataset = prepare_dataset(rows, tokenizer, max_length=6)
    batch = next(
        iter(
            DataLoader(
                cast(TorchDataset, dataset),
                batch_size=3,
                collate_fn=DataCollatorForTokenClassification(tokenizer),
            )
        )
    )

    assert batch["input_ids"].tolist() == [
        [2, 4, 5, 6, 7, 3],
        [2, 8, 3, 1, 1, 1],
        [2, 7, 3, 1, 1, 1],
    ]
    assert batch["labels"].tolist() == [
        [-100, 0, 1, 2, 0, -100],
        [-100, 1, -100, -100, -100, -100],
        [-100, 0, -100, -100, -100, -100],
    ]
    assert batch["attention_mask"].tolist() == [
        [1] * 6,
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
    ]
    assert batch["is_clean"].tolist() == [False, False, True]
    assert batch["is_clean"].dtype == torch.bool
    assert set(batch) <= {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "labels",
        "is_clean",
    }
    assert all(
        tensor.dtype == torch.long for key, tensor in batch.items() if key != "is_clean"
    )


@pytest.mark.parametrize("validation", [False, True])
@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_cleanliness_and_metadata_follow_chunks(tokenizer, validation, padding_side):
    tokenizer.padding_side = padding_side
    rows = Dataset.from_list(
        [
            {
                "text": "okay okay speling",
                "annotations": [
                    {
                        "original": "speling",
                        "corrected": "spelling",
                        "span_start": 10,
                        "span_end": 17,
                    }
                ],
            },
            {"text": "okay", "annotations": []},
        ]
    )
    dataset = prepare_dataset(
        rows, tokenizer, max_length=4, include_metadata=validation
    )
    if validation:
        dataset = dataset.add_column("passage_index", list(range(len(dataset))))
    collator = (
        ValidationCollator(tokenizer)
        if validation
        else DataCollatorForTokenClassification(tokenizer)
    )
    batch = collator(list(dataset))

    assert batch["is_clean"].tolist() == [True, False, True]
    assert batch["is_clean"].dtype == torch.bool
    assert batch["labels"][batch["labels"] != -100].tolist() == [0, 0, 1, 2, 0]
    if validation:
        assert [item["text"] for item in batch["metadata"]] == [
            "okay okay",
            "speling",
            "okay",
        ]
        assert [
            evaluated_word_count(item["text"], item["offset_mapping"], labels)
            for item, labels in zip(
                batch["metadata"], batch["labels"].tolist(), strict=True
            )
        ] == [2, 1, 1]
        for item in batch["metadata"]:
            assert (
                min(start for start, end in item["offset_mapping"] if start != end) == 0
            )


def test_failed_annotations_are_excluded_but_clean_and_partial_rows_remain(tokenizer):
    rows = Dataset.from_list(
        [
            {"text": "tail", "annotations": None, "failures": ["failed"]},
            {"text": "okay", "annotations": [], "failures": []},
            {
                "text": "speling",
                "annotations": [
                    {
                        "original": "speling",
                        "corrected": "spelling",
                        "span_start": 0,
                        "span_end": 7,
                    }
                ],
                "failures": ["earlier attempt failed"],
            },
        ]
    )

    dataset = prepare_dataset(rows, tokenizer)

    assert list(dataset["labels"]) == [[-100, 0, -100], [-100, 1, 2, -100]]
    assert list(dataset["input_ids"]) == [[2, 7, 3], [2, 5, 6, 3]]


def test_no_usable_annotations(tokenizer):
    rows = Dataset.from_list([{"text": "speling", "annotations": None}])

    with pytest.raises(ValueError, match="No passages with usable annotations"):
        prepare_dataset(rows, tokenizer)


def test_blank_and_duplicate_passages_are_excluded_before_tokenization(tokenizer):
    rows = Dataset.from_list(
        [
            {"text": "okay", "annotations": None},
            {"text": " \n\t", "annotations": []},
            {"text": "okay", "annotations": []},
            {"text": "okay", "annotations": []},
            {"text": "tail", "annotations": []},
        ]
    )

    dataset = prepare_dataset(rows, tokenizer)

    assert list(dataset["input_ids"]) == [[2, 7, 3], [2, 8, 3]]
    assert list(dataset["labels"]) == [[-100, 0, -100], [-100, 0, -100]]


def test_only_blank_passages_are_rejected(tokenizer):
    rows = Dataset.from_list([{"text": " \n\t", "annotations": []}])

    with pytest.raises(ValueError, match="No nonblank passages"):
        prepare_dataset(rows, tokenizer)


def test_chunk_boundary_keeps_multi_token_word_together(tokenizer):
    rows = Dataset.from_list(
        [
            {
                "text": "🐈 speling tail",
                "annotations": [
                    {
                        "original": "speling",
                        "corrected": "spelling",
                        "span_start": 2,
                        "span_end": 9,
                    }
                ],
            }
        ]
    )
    dataset = prepare_dataset(rows, tokenizer, max_length=4, include_metadata=True)

    assert list(dataset["input_ids"]) == [[2, 4, 3], [2, 5, 6, 3], [2, 8, 3]]
    assert list(dataset["labels"]) == [
        [-100, 0, -100],
        [-100, 1, 2, -100],
        [-100, 0, -100],
    ]
    assert list(dataset["text"]) == ["🐈", "speling", "tail"]
    assert list(dataset["offset_mapping"])[1] == [[0, 0], [0, 3], [3, 7], [0, 0]]
    assert (
        sum(
            evaluated_word_count(row["text"], row["offset_mapping"], row["labels"])
            for row in dataset
        )
        == 3
    )


@pytest.mark.parametrize(
    "text,offsets,budget,expected",
    [
        (
            "🐈 speling tail",
            [(0, 1), (2, 5), (5, 9), (10, 14)],
            2,
            [(0, 1), (1, 3), (3, 4)],
        ),
        ("speling", [(0, 3), (3, 7)], 1, [(0, 1), (1, 2)]),
        ("", [], 2, [(0, 0)]),
        ("okay\n\ttail", [(0, 4), (6, 10)], 1, [(0, 1), (1, 2)]),
        ("okay tail", [(0, 4), (4, 9)], 1, [(0, 1), (1, 2)]),
    ],
)
def test_chunk_boundaries_prefer_whole_words(text, offsets, budget, expected):
    assert chunk_boundaries(text, offsets, budget) == expected


def test_oversized_word_retains_all_tokens_and_error_labels(tokenizer):
    rows = Dataset.from_list(
        [
            {
                "text": "speling",
                "annotations": [
                    {
                        "original": "speling",
                        "corrected": "spelling",
                        "span_start": 0,
                        "span_end": 7,
                    }
                ],
            },
            {"text": "speling tail", "annotations": []},
        ]
    )
    dataset = prepare_dataset(rows, tokenizer, max_length=3, include_metadata=True)
    assert list(dataset["input_ids"]) == [
        [2, 5, 3],
        [2, 6, 3],
        [2, 5, 3],
        [2, 6, 3],
        [2, 8, 3],
    ]
    assert list(dataset["labels"]) == [
        [-100, 1, -100],
        [-100, 1, -100],
        [-100, 0, -100],
        [-100, 0, -100],
        [-100, 0, -100],
    ]

    assert list(dataset["word_count"]) == [1, 0, 1, 0, 1]
    assert list(dataset["is_clean"]) == [False, False, True, True, True]
