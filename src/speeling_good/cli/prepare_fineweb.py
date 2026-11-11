"""Stream FineWeb into cached, annotated JSONL shards."""

import json
import os
import random
import sys
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
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
from pydantic import BaseModel, ConfigDict, Field, field_validator

from speeling_good.preprocessing.early_decision_provider import (
    EarlyDecisionConfig,
    EarlyDecisionProvider,
)
from speeling_good.preprocessing.llm_annotation_provider import (
    AnnotationConfig,
    LlmAnnotationProvider,
)
from speeling_good.preprocessing.types import (
    AnnotatedPassage,
    PassageWithEarlyDecision,
    RawPassage,
)
from speeling_good.preprocessing.usage import UsageTracker

# Bump when resolution rules or output types change; prompts are hashed directly.
ANNOTATION_CACHE_VERSION = 3
EARLY_DECISION_CACHE_VERSION = 1


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
    p_annotated_anyway: Annotated[
        float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)
    ] = 1.0
    early_decision_endpoint: Annotated[str, Field(min_length=1)] = (
        "https://openrouter.ai/api/alpha"
    )
    early_decision: EarlyDecisionConfig
    annotation: AnnotationConfig

    @field_validator("early_decision_endpoint")
    @classmethod
    def normalize_decision_endpoint(cls, value: str) -> str:
        endpoint = value.strip().rstrip("/")
        if not endpoint:
            raise ValueError("must not be empty after normalization")
        return endpoint


@dataclass(frozen=True, slots=True)
class PreparedPassage:
    """A pipeline result with skip information used only for progress accounting."""

    passage: AnnotatedPassage
    skipped: bool = False


def cached_early_decision(
    text: str,
    _early_decision: dict[str, object],
    _endpoint: str,
    _cache_version: int,
    provider: EarlyDecisionProvider,
) -> PassageWithEarlyDecision:
    """Cache decision results independently of gate sampling and annotation."""
    return provider.decide(RawPassage(text))


def cached_annotation(
    passage: PassageWithEarlyDecision,
    _annotation: dict[str, object],
    _endpoint: str,
    _cache_version: int,
    provider: LlmAnnotationProvider,
) -> AnnotatedPassage:
    """Annotate text, retaining underscore-prefixed arguments only as cache keys.

    Joblib hashes _annotation, _endpoint, and _cache_version even though the body does
    not use them. The provider is excluded explicitly when configuring the cache.
    """
    return provider.annotate(passage)


def should_annotate(
    passage: PassageWithEarlyDecision, p_annotated_anyway: float
) -> bool:
    """Always annotate errors and unknowns; sample only known-clean passages."""
    return (
        passage.decision is None
        or passage.decision.contains_spelling_error
        or p_annotated_anyway == 1.0
        or (p_annotated_anyway > 0.0 and random.random() < p_annotated_anyway)
    )


def annotate_batches(
    rows: Iterable[Mapping[str, object]],
    annotate: Callable[[RawPassage], PreparedPassage],
    *,
    take: int | None,
    shard_size: int,
    workers: int,
) -> Iterator[list[PreparedPassage]]:
    """Annotate schema-validated rows, retaining partial and failed results."""

    def annotate_row(row: Mapping[str, object]) -> PreparedPassage:
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


def _write_usage(output: Path, usage: UsageTracker) -> None:
    with atomic_text(output / "usage.json") as stream:
        json.dump(usage.snapshot(), stream, indent=2)
        stream.write("\n")


def prepare(args: PrepareConfig, endpoint: str, token: str) -> None:
    config = args.model_dump(mode="json") | {"endpoint": endpoint}
    _prepare_output(args.output, config)
    usage = UsageTracker()
    _write_usage(args.output, usage)
    try:
        _run_pipeline(args, endpoint, token, usage)
    finally:
        # A failed final write must not hide the error that stopped the pipeline.
        failed = sys.exception() is not None
        try:
            _write_usage(args.output, usage)
        except OSError as error:
            if not failed:
                raise
            logger.error("Could not save token usage after pipeline failure: {}", error)


def _run_pipeline(
    args: PrepareConfig, endpoint: str, token: str, usage: UsageTracker
) -> None:
    output = args.output
    decision_endpoint = args.early_decision_endpoint

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
    cached_decide = memory.cache(cached_early_decision, ignore=["provider"])
    annotation = args.annotation.model_dump()
    early_decision = args.early_decision.model_dump()
    written = failed = with_failures = decision_failures = skipped = 0
    with (
        OpenAI(base_url=endpoint, api_key=token) as client,
        OpenAI(base_url=decision_endpoint, api_key=token) as decision_client,
    ):
        provider = LlmAnnotationProvider(
            client=client,
            config=args.annotation,
            max_inflight=args.workers,
            usage=usage,
        )
        decision_provider = EarlyDecisionProvider(
            client=decision_client,
            config=args.early_decision,
            max_inflight=args.workers,
            usage=usage,
        )

        def annotate(passage: RawPassage) -> PreparedPassage:
            decided = cached_decide(
                passage.text,
                early_decision,
                decision_endpoint,
                EARLY_DECISION_CACHE_VERSION,
                decision_provider,
            )
            if not should_annotate(decided, args.p_annotated_anyway):
                return PreparedPassage(
                    AnnotatedPassage(
                        decided.text,
                        [],
                        decision=decided.decision,
                        early_decision_failures=decided.failures,
                    ),
                    skipped=True,
                )
            return PreparedPassage(
                cached(
                    decided,
                    annotation,
                    endpoint,
                    ANNOTATION_CACHE_VERSION,
                    provider,
                )
            )

        for index, results in enumerate(
            annotate_batches(
                rows,
                annotate,
                take=args.take,
                shard_size=args.shard_size,
                workers=args.workers,
            )
        ):
            batch = [result.passage for result in results]
            skipped += sum(result.skipped for result in results)
            shard = output / f"part-{index:05d}.jsonl"
            _write_shard(shard, batch)
            _write_usage(output, usage)
            batch_failed = sum(passage.annotations is None for passage in batch)
            batch_with_failures = sum(bool(passage.failures) for passage in batch)
            batch_decision_failures = sum(
                bool(passage.early_decision_failures) for passage in batch
            )
            written += len(batch)
            failed += batch_failed
            with_failures += batch_with_failures
            decision_failures += batch_decision_failures
            logger.info(
                "Wrote {} ({} usable, {} failed, {} with annotation failure history, "
                "{} with decision failure history; {} skipped annotation so far)",
                shard.name,
                len(batch) - batch_failed,
                batch_failed,
                batch_with_failures,
                batch_decision_failures,
                skipped,
            )

    logger.info(
        "Finished: {} written, {} usable, {} failed, {} with annotation failure "
        "history, {} with decision failure history, {} skipped annotation",
        written,
        written - failed,
        failed,
        with_failures,
        decision_failures,
        skipped,
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
