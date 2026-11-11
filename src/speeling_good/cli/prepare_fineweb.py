"""Stream FineWeb into cached, annotated JSONL shards."""

import json
import os
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO, Annotated, cast

import hydra
import yaml
from datasets import Value, load_dataset
from joblib import Memory, Parallel, delayed
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from speeling_good.preprocessing.llm_annotation_provider import (
    AnnotationConfig,
    LlmAnnotationProvider,
)
from speeling_good.preprocessing.types import AnnotatedPassage, RawPassage

# Bump when resolution rules or output types change; prompts are hashed directly.
ANNOTATION_CACHE_VERSION = 2


class PrepareConfig(BaseModel):
    """Validate run settings before creating output or opening external services."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    subset: str
    split: str
    take: Annotated[int, Field(strict=True, gt=0)] | None
    output: Path
    workers: Annotated[int, Field(strict=True, gt=0)]
    shard_size: Annotated[int, Field(strict=True, gt=0)]
    annotation: AnnotationConfig


def cached_annotation(
    text: str,
    _annotation: dict[str, object],
    _endpoint: str,
    _cache_version: int,
    provider: LlmAnnotationProvider,
) -> AnnotatedPassage:
    """Annotate text, retaining underscore-prefixed arguments only as cache keys.

    Joblib hashes _annotation, _endpoint, and _cache_version even though the body does
    not use them. The provider is excluded explicitly when configuring the cache.
    """
    return provider.annotate(RawPassage(text))


def annotate_batches(
    rows: Iterable[Mapping[str, object]],
    annotate: Callable[[RawPassage], AnnotatedPassage],
    *,
    take: int | None,
    shard_size: int,
    workers: int,
) -> Iterator[list[AnnotatedPassage]]:
    """Annotate schema-validated rows, retaining partial and failed results."""

    def annotate_row(row: Mapping[str, object]) -> AnnotatedPassage:
        return annotate(RawPassage(cast(str, row["text"])))

    source = iter(rows)
    try:
        limited = islice(source, take)
        with Parallel(n_jobs=workers, backend="threading", batch_size=1) as parallel:
            while batch := list(islice(limited, shard_size)):
                yield parallel(delayed(annotate_row)(row) for row in batch)
    finally:
        # A take limit leaves streaming generators suspended inside file readers.
        if isinstance(source, Generator):
            source.close()


@contextmanager
def atomic_text(path: Path) -> Generator[IO[str]]:
    """Replace a completed file atomically, removing temporary files on failure."""
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            yield stream.file
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _prepare_output(output: Path, config: dict[str, object]) -> None:
    """Create a fresh output directory and persist its configuration."""
    output.mkdir(parents=True)
    config_path = output / "config.yaml"
    with atomic_text(config_path) as stream:
        yaml.safe_dump(config, stream, sort_keys=True)


def _write_shard(path: Path, passages: Iterable[AnnotatedPassage]) -> None:
    """Serialize a complete JSONL shard before replacing its destination."""
    with atomic_text(path) as stream:
        for passage in passages:
            stream.write(json.dumps(asdict(passage), ensure_ascii=False) + "\n")


def prepare(args: PrepareConfig, endpoint: str, token: str) -> None:
    output: Path = args.output
    config = args.model_dump(mode="json") | {"endpoint": endpoint}
    _prepare_output(output, config)

    logger.info(
        "Preparing {}/{} ({}, take={}, workers={})",
        args.dataset,
        args.subset,
        args.split,
        args.take,
        args.workers,
    )
    rows = load_dataset(
        args.dataset,
        name=args.subset,
        split=args.split,
        streaming=True,
    )
    if rows.features is None or rows.features["text"] not in (
        Value("string"),
        Value("large_string"),
    ):
        raise TypeError("Dataset 'text' feature must be a string")
    memory = Memory(output / ".cache", verbose=0)
    cached = memory.cache(cached_annotation, ignore=["provider"])
    annotation = args.annotation.model_dump()
    written = failed = with_failures = 0
    with OpenAI(base_url=endpoint, api_key=token) as client:
        provider = LlmAnnotationProvider(
            client=client,
            config=args.annotation,
            max_inflight=args.workers,
        )

        def annotate(passage: RawPassage) -> AnnotatedPassage:
            return cached(
                passage.text,
                annotation,
                endpoint,
                ANNOTATION_CACHE_VERSION,
                provider,
            )

        for index, batch in enumerate(
            annotate_batches(
                rows,
                annotate,
                take=args.take,
                shard_size=args.shard_size,
                workers=args.workers,
            )
        ):
            shard = output / f"part-{index:05d}.jsonl"
            _write_shard(shard, batch)
            batch_failed = sum(passage.annotations is None for passage in batch)
            batch_with_failures = sum(bool(passage.failures) for passage in batch)
            written += len(batch)
            failed += batch_failed
            with_failures += batch_with_failures
            logger.info(
                "Wrote {} ({} usable, {} failed, {} with failure history)",
                shard.name,
                len(batch) - batch_failed,
                batch_failed,
                batch_with_failures,
            )

    logger.info(
        "Finished: {} written, {} usable, {} failed, {} with failure history",
        written,
        written - failed,
        failed,
        with_failures,
    )


@hydra.main(
    version_base="1.3",
    config_path=str(Path(__file__).resolve().parents[3] / "config"),
    config_name="prepare_fineweb",
)
def main(config: DictConfig) -> None:
    args = PrepareConfig.model_validate(OmegaConf.to_container(config, resolve=True))
    endpoint = os.environ.get("MODEL_ENDPOINT", "").strip().rstrip("/")
    token = os.environ.get("MODEL_TOKEN", "").strip()
    if not endpoint or not token:
        raise ValueError("MODEL_ENDPOINT and MODEL_TOKEN must be set")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    prepare(args, endpoint + "/", token)


if __name__ == "__main__":
    main()
