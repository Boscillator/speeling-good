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
    assert set(batch) <= {"input_ids", "attention_mask", "token_type_ids", "labels"}
    assert all(tensor.dtype == torch.long for tensor in batch.values())


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
