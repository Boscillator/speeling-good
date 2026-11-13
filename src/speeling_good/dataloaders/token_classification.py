"""Build indexed training batches from prepared annotation shards.

TODO: Rename to dataloader.py

Example::

    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from speeling_good.dataloaders import create_dataloaders
    from speeling_good.dataloaders.annotation_to_bio import ID2LABEL, LABEL2ID

    checkpoint = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    loader, validation_loader = create_dataloaders("data/fineweb", tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )
    batch = next(iter(loader))
    batch.pop("is_clean")
    loss = model(**batch).loss
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
    include_metadata: bool = False,
) -> Dataset:
    """Clean and tokenize an indexed annotation dataset.

    Only text and annotations are consumed. Empty annotation lists represent
    clean passages; null annotations, blank text, and duplicate passages are
    excluded. Long passages are truncated; unsupervised passages are excluded.
    The returned dataset contains unpadded model inputs, BIO label IDs, and an
    is_clean flag derived from the original annotations before truncation.
    include_metadata retains text and character offsets for validation reporting.
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

    seen: set[str] = set()
    indices = []
    for index, text in enumerate(usable["text"]):
        if text.strip() and text not in seen:
            seen.add(text)
            indices.append(index)
    usable = usable.select(indices)
    if not len(usable):
        raise ValueError("No nonblank passages with usable annotations remain")

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
        encoding["is_clean"] = [not annotations for annotations in batch["annotations"]]
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
        if include_metadata:
            encoding["text"] = batch["text"]
            encoding["offset_mapping"] = offsets
        return dict(encoding)

    dataset = usable.map(
        encode,
        batched=True,
        remove_columns=usable.column_names,
        desc="Tokenizing annotated passages",
    )
    dataset = dataset.filter(
        lambda labels: any(label != -100 for label in labels),
        input_columns=["labels"],
        desc="Filtering unsupervised passages",
    )
    if not len(dataset):
        raise ValueError("No supervised passages remain")
    return dataset


class ValidationCollator:
    """Keep passage metadata outside the padded model inputs."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.collator = DataCollatorForTokenClassification(tokenizer)
        self.padding_side = tokenizer.padding_side

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        metadata = [
            {key: feature[key] for key in ("text", "offset_mapping", "passage_index")}
            for feature in features
        ]
        batch = self.collator(
            [
                {key: value for key, value in feature.items() if key not in metadata[0]}
                for feature in features
            ]
        )
        length = batch["labels"].shape[1]
        for item in metadata:
            offsets = item["offset_mapping"]
            padding = [(0, 0)] * (length - len(offsets))
            item["offset_mapping"] = (
                offsets + padding if self.padding_side == "right" else padding + offsets
            )
        return {**batch, "metadata": metadata}


def create_dataloaders(
    directory: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int = 8,
    max_length: int = 512,
    validation_fraction: float = 0.2,
    num_workers: int = 0,
    seed: int = 0,
) -> tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, Any]]]:
    """Load prepared shards and return training and validation loaders.

    Hugging Face caches the indexed dataset and tokenization on disk. The seed
    controls shuffling; successive epochs advance the generator normally.
    Duplicate passages are removed before the seeded split to prevent leakage.
    Labels for special tokens and batch padding are -100, ignored by the loss.
    """
    shards = sorted(Path(directory).glob("part-*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No part-*.jsonl shards found in {directory}")
    rows = load_dataset(
        "json", data_files=[str(path) for path in shards], split="train"
    )
    dataset = prepare_dataset(
        rows, tokenizer, max_length=max_length, include_metadata=True
    )
    if len(dataset) < 2:
        raise ValueError("Need at least two distinct supervised passages")
    split = dataset.train_test_split(test_size=validation_fraction, seed=seed)
    logger.info(
        "Training on {} passages; validating on {}",
        len(split["train"]),
        len(split["test"]),
    )
    split["train"] = split["train"].remove_columns(["text", "offset_mapping"])
    split["test"] = split["test"].add_column(
        "passage_index", list(range(len(split["test"])))
    )
    loaders = []
    for name in ("train", "test"):
        loaders.append(
            DataLoader(
                # HF datasets implement PyTorch's indexed dataset protocol.
                cast(TorchDataset, split[name]),
                batch_size=batch_size,
                shuffle=name == "train",
                num_workers=num_workers,
                generator=torch.Generator().manual_seed(seed),
                collate_fn=(
                    DataCollatorForTokenClassification(tokenizer)
                    if name == "train"
                    else ValidationCollator(tokenizer)
                ),
            )
        )
    return loaders[0], loaders[1]
