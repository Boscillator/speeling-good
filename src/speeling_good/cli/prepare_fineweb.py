"""Stream FineWeb into cached, annotated JSONL shards."""

import argparse
import json
import os
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO, cast

import yaml
from datasets import Value, load_dataset
from joblib import Memory, Parallel, delayed
from loguru import logger
from openai import OpenAI

from speeling_good.preprocessing.llm_annotation_provider import (
    LlmAnnotationProvider,
)
from speeling_good.preprocessing.types import AnnotatedPassage, RawPassage

# Bump when annotation prompts, resolution rules, or output types change.
ANNOTATION_CACHE_VERSION = 2


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def take_count(value: str) -> int | None:
    return None if value == "all" else positive_int(value)


def cached_annotation(
    text: str,
    _model: str,
    _endpoint: str,
    _cache_version: int,
    provider: LlmAnnotationProvider,
) -> AnnotatedPassage:
    """Annotate text, retaining underscore-prefixed arguments only as cache keys.

    Joblib hashes _model, _endpoint, and _cache_version even though the body does
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


def prepare(args: argparse.Namespace, endpoint: str, token: str) -> None:
    output: Path = args.output
    config = vars(args) | {"output": str(output), "endpoint": endpoint}
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
    if rows.features["text"] not in (Value("string"), Value("large_string")):
        raise TypeError("Dataset 'text' feature must be a string")
    memory = Memory(output / ".cache", verbose=0)
    cached = memory.cache(cached_annotation, ignore=["provider"])
    written = failed = with_failures = 0
    with OpenAI(base_url=endpoint, api_key=token) as client:
        provider = LlmAnnotationProvider(
            client=client,
            model=args.model,
            max_inflight=args.workers,
        )

        def annotate(passage: RawPassage) -> AnnotatedPassage:
            return cached(
                passage.text,
                args.model,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--subset", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--take", type=take_count, default=100, metavar="N|all")
    parser.add_argument("--model", default="qwen/qwen3.8-27b")
    parser.add_argument("--output", type=Path, default=Path("data/fineweb"))
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--shard-size", type=positive_int, default=100)
    args = parser.parse_args()
    endpoint = os.environ.get("MODEL_ENDPOINT", "").strip().rstrip("/")
    token = os.environ.get("MODEL_TOKEN", "").strip()
    if not endpoint or not token:
        parser.error("MODEL_ENDPOINT and MODEL_TOKEN must be set")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    prepare(args, endpoint + "/", token)


if __name__ == "__main__":
    main()
