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

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import torch
from datasets import Dataset, load_dataset
from loguru import logger
from tokenizers import Encoding
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    DataCollatorForTokenClassification,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
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
    excluded. Long passages become chunks that prefer whole words; unsupervised chunks
    are excluded. The returned dataset contains unpadded model inputs, BIO label
    IDs, and an is_clean flag derived from each chunk's labels. include_metadata
    retains chunk text and local character offsets for validation reporting.
    """
    return _tokenize_passages(
        _clean_passages(rows),
        tokenizer,
        max_length=max_length,
        include_metadata=include_metadata,
    )


def _tokenize_passages(
    rows: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    include_metadata: bool,
) -> Dataset:
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for character offsets")
    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer must define a padding token")
    if max_length <= tokenizer.num_special_tokens_to_add(pair=False):
        raise ValueError("max_length must leave room for passage tokens")
    if max_length > tokenizer.model_max_length:
        raise ValueError("max_length exceeds the tokenizer's model_max_length")

    fast_tokenizer = cast(PreTrainedTokenizerFast, tokenizer)
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)

    def encode(batch: dict[str, Any]) -> dict[str, Any]:
        passages = fast_tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        ).encodings
        chunks = []
        for text, passage, raw_annotations in zip(
            batch["text"], passages, batch["annotations"], strict=True
        ):
            annotations = [SpellingAnnotation(**item) for item in raw_annotations]
            for start, end in chunk_boundaries(text, passage.offsets, budget):
                chunks.append(
                    _encode_chunk(
                        text,
                        passage,
                        start,
                        end,
                        annotations,
                        fast_tokenizer,
                        include_metadata,
                    )
                )
        return {key: [chunk[key] for chunk in chunks] for key in chunks[0]}

    dataset = rows.map(
        encode,
        batched=True,
        remove_columns=rows.column_names,
        desc="Chunking annotated passages",
    )
    dataset = dataset.filter(
        lambda labels: any(label != -100 for label in labels),
        input_columns=["labels"],
        desc="Filtering unsupervised chunks",
    )
    if not len(dataset):
        raise ValueError("No supervised chunks remain")
    return dataset


def chunk_boundaries(
    text: str, offsets: Sequence[Sequence[int]], max_tokens: int
) -> list[tuple[int, int]]:
    """Return token ranges, preferring whitespace boundaries within the budget.

    A whitespace-delimited word longer than the budget is split to make progress.
    Offsets come from tokenizing the complete passage without special tokens.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not offsets:
        return [(0, 0)]
    boundaries = []
    start = 0
    while start < len(offsets):
        limit = min(start + max_tokens, len(offsets))
        end = limit
        while end < len(offsets) and end > start:
            position = offsets[end][0]
            if offsets[end - 1][1] <= position and (
                text[position - 1 : position].isspace()
                or text[position : position + 1].isspace()
            ):
                break
            end -= 1
        if end == start:
            end = limit
        boundaries.append((start, end))
        start = end
    return boundaries


def _chunk_metadata(
    text: str, offsets: Sequence[Sequence[int]], mask: Sequence[int]
) -> dict[str, Any]:
    """Slice visible chunk text and rebase its character offsets."""
    visible = [
        (start, end)
        for (start, end), special in zip(offsets, mask, strict=True)
        if not special and start != end
    ]
    start = min((start for start, _ in visible), default=0)
    end = max((end for _, end in visible), default=0)
    word_count = len(text[start:end].split())
    if (
        start > 0
        and text[start:end]
        and not text[start - 1].isspace()
        and not text[start].isspace()
    ):
        # A continuation of an oversized word was counted in an earlier chunk.
        word_count -= 1
    return {
        "text": text[start:end],
        "word_count": word_count,
        "offset_mapping": [
            (left - start, right - start) if not special and left != right else (0, 0)
            for (left, right), special in zip(offsets, mask, strict=True)
        ],
    }


def _encode_chunk(
    text: str,
    passage: Encoding,
    start: int,
    end: int,
    annotations: list[SpellingAnnotation],
    tokenizer: PreTrainedTokenizerFast,
    include_metadata: bool,
) -> dict[str, Any]:
    # Encoding has no slice operation. Trim a copy from each side, preserving
    # original token IDs and offsets, then add the tokenizer's special tokens.
    chunk = deepcopy(passage)
    chunk.truncate(end)
    chunk.truncate(end - start, direction="left")
    chunk = tokenizer.backend_tokenizer.post_process(chunk)
    labels = annotation_to_bio(
        chunk.offsets, annotations, special_tokens_mask=chunk.special_tokens_mask
    )
    result = {
        "input_ids": chunk.ids,
        "attention_mask": chunk.attention_mask,
        "labels": labels,
        "is_clean": not any(label > 0 for label in labels),
    }
    if "token_type_ids" in tokenizer.model_input_names:
        result["token_type_ids"] = chunk.type_ids
    if include_metadata:
        result.update(_chunk_metadata(text, chunk.offsets, chunk.special_tokens_mask))
    return result


def _clean_passages(rows: Dataset) -> Dataset:
    """Remove unusable and duplicate passages before splitting or chunking."""
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
    return usable


class ValidationCollator:
    """Keep passage metadata outside the padded model inputs."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.collator = DataCollatorForTokenClassification(tokenizer)
        self.padding_side = tokenizer.padding_side

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        metadata = [
            {
                key: feature[key]
                for key in ("text", "offset_mapping", "passage_index", "word_count")
            }
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
    rows = _clean_passages(rows)
    if len(rows) < 2:
        raise ValueError("Need at least two distinct usable passages")
    split = rows.train_test_split(test_size=validation_fraction, seed=seed)
    for name in ("train", "test"):
        split[name] = _tokenize_passages(
            split[name],
            tokenizer,
            max_length=max_length,
            include_metadata=name == "test",
        )
    logger.info(
        "Training on {} chunks; validating on {}",
        len(split["train"]),
        len(split["test"]),
    )
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
