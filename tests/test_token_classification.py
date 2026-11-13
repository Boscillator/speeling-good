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
from speeling_good.dataloaders.token_classification import ValidationCollator


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


def test_unicode_subwords_truncation_and_batch_padding(tokenizer):
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
                batch_size=2,
                collate_fn=DataCollatorForTokenClassification(tokenizer),
            )
        )
    )

    assert batch["input_ids"].tolist() == [[2, 4, 5, 6, 7, 3], [2, 7, 3, 1, 1, 1]]
    assert batch["labels"].tolist() == [
        [-100, 0, 1, 2, 0, -100],
        [-100, 0, -100, -100, -100, -100],
    ]
    assert batch["attention_mask"].tolist() == [[1] * 6, [1, 1, 1, 0, 0, 0]]
    assert batch["is_clean"].tolist() == [False, True]
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
def test_cleanliness_uses_original_annotations_before_truncation(
    tokenizer, validation, padding_side
):
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
        dataset = dataset.add_column("passage_index", [0, 1])
    collator = (
        ValidationCollator(tokenizer)
        if validation
        else DataCollatorForTokenClassification(tokenizer)
    )
    batch = collator(list(dataset))

    assert batch["is_clean"].tolist() == [False, True]
    assert batch["is_clean"].dtype == torch.bool
    assert batch["labels"][batch["labels"] != -100].tolist() == [0, 0, 0]
    if validation:
        assert [item["text"] for item in batch["metadata"]] == rows["text"]


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
