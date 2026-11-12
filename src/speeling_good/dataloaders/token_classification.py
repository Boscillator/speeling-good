"""Build indexed training batches from prepared annotation shards.

Example::

    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from speeling_good.dataloaders import create_dataloader
    from speeling_good.dataloaders.annotation_to_bio import ID2LABEL, LABEL2ID

    checkpoint = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    loader = create_dataloader("data/fineweb", tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )
    loss = model(**next(iter(loader))).loss
"""

from pathlib import Path
from typing import Any, cast

import torch
from datasets import Dataset, load_dataset
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    DataCollatorForTokenClassification,
    PreTrainedTokenizerBase,
)

from speeling_good.dataloaders.annotation_to_bio import annotation_to_bio
from speeling_good.preprocessing.types import SpellingAnnotation


def prepare_dataset(
    rows: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int = 512,
) -> Dataset:
    """Filter failed annotations and tokenize an indexed annotation dataset.

    Only text and annotations are consumed. Empty annotation lists represent
    clean passages; null annotations are excluded. Long passages are truncated.
    The returned dataset contains unpadded model inputs and BIO label IDs.
    """
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for character offsets")
    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer must define a padding token")
    if max_length <= tokenizer.num_special_tokens_to_add(pair=False):
        raise ValueError("max_length must leave room for passage tokens")
    if max_length > tokenizer.model_max_length:
        raise ValueError("max_length exceeds the tokenizer's model_max_length")

    rows = rows.select_columns(["text", "annotations"])
    usable = rows.filter(
        lambda annotations: annotations is not None,
        input_columns=["annotations"],
        desc="Filtering failed annotations",
    )
    logger.info(
        "Retained {} passages; excluded {} failed annotations",
        len(usable),
        len(rows) - len(usable),
    )
    if not len(usable):
        raise ValueError("No passages with usable annotations remain")

    def encode(batch: dict[str, Any]) -> dict[str, Any]:
        encoding = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        offsets = encoding.pop("offset_mapping")
        masks = encoding.pop("special_tokens_mask")
        encoding["labels"] = [
            annotation_to_bio(
                passage_offsets,
                [SpellingAnnotation(**annotation) for annotation in annotations],
                special_tokens_mask=mask,
            )
            for passage_offsets, mask, annotations in zip(
                offsets, masks, batch["annotations"], strict=True
            )
        ]
        return dict(encoding)

    return usable.map(
        encode,
        batched=True,
        remove_columns=usable.column_names,
        desc="Tokenizing annotated passages",
    )


def create_dataloader(
    data_dir: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int = 8,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Read sorted part-*.jsonl shards and return dynamically padded CPU batches.

    Hugging Face caches the indexed dataset and tokenization on disk. The seed
    controls shuffling; successive epochs advance the generator normally.
    Labels for special tokens and batch padding are -100, ignored by the loss.
    """
    shards = sorted(Path(data_dir).glob("part-*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No part-*.jsonl shards found in {data_dir}")
    rows = load_dataset(
        "json", data_files=[str(path) for path in shards], split="train"
    )
    dataset = prepare_dataset(rows, tokenizer, max_length=max_length)
    return DataLoader(
        # HF datasets implement PyTorch's indexed dataset protocol without
        # inheriting its Dataset class.
        cast(TorchDataset, dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=DataCollatorForTokenClassification(
            tokenizer, label_pad_token_id=-100, return_tensors="pt"
        ),
    )
