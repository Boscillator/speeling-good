"""Stream FineWeb into resumable annotated JSONL shards."""

import json
import os
import random
import sys
from collections import deque
from collections.abc import Callable, Generator, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing, contextmanager
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO, Annotated, cast

import hydra
import yaml
from datasets import Value, load_dataset
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
    skip_early_decision: Annotated[bool, Field(strict=True)] = False
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


def _annotate_in_order(
    passages: Iterable[RawPassage],
    annotate: Callable[[RawPassage], PreparedPassage],
    *,
    workers: int,
    window: int,
) -> Generator[PreparedPassage]:
    """Run bounded concurrent requests, retrieving results and errors in order."""
    source = iter(passages)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        try:
            for passage in islice(source, window):
                pending.append(executor.submit(annotate, passage))
            while pending:
                yield pending.popleft().result()
                passage = next(source, None)
                if passage is not None:
                    pending.append(executor.submit(annotate, passage))
        finally:
            for future in pending:
                future.cancel()


def annotate_batches(
    rows: Iterable[Mapping[str, object]],
    annotate: Callable[[RawPassage], PreparedPassage],
    *,
    take: int | None,
    shard_size: int,
    workers: int,
    completed_rows: int = 0,
) -> Generator[list[PreparedPassage]]:
    """Keep a bounded request window running across ordered shard boundaries."""

    source = iter(rows)
    try:
        if take is not None and completed_rows >= take:
            return
        remaining = islice(source, completed_rows, take)
        passages = (RawPassage(cast(str, row["text"])) for row in remaining)
        with closing(
            _annotate_in_order(
                passages,
                annotate,
                workers=workers,
                window=shard_size + 2 * workers,
            )
        ) as results:
            while batch := list(islice(results, shard_size)):
                yield batch
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


def _completed_shards(output: Path) -> tuple[int, int]:
    """Count the committed input prefix, rejecting ambiguous checkpoints."""
    shards = sorted(output.glob("part-*.jsonl"), key=lambda path: int(path.stem[5:]))
    completed_rows = 0
    for index, shard in enumerate(shards):
        if shard.name != f"part-{index:05d}.jsonl":
            raise ValueError(f"Expected part-{index:05d}.jsonl, found {shard.name}")
        count = 0
        with shard.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or not isinstance(
                        row.get("text"), str
                    ):
                        raise TypeError("Expected a passage object with string text")
                except (ValueError, TypeError) as error:
                    raise ValueError(
                        f"Invalid checkpoint {shard}:{line_number}"
                    ) from error
                count += 1
        if not count:
            raise ValueError(f"Empty checkpoint: {shard}")
        completed_rows += count
    return len(shards), completed_rows


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
    args.output.mkdir(parents=True, exist_ok=True)
    completed_shards, completed_rows = _completed_shards(args.output)
    usage_path = args.output / "usage.json"
    usage = (
        UsageTracker.from_snapshot(json.loads(usage_path.read_text(encoding="utf-8")))
        if usage_path.exists()
        else UsageTracker()
    )
    config = args.model_dump(mode="json") | {"endpoint": endpoint}
    with atomic_text(args.output / "config.yaml") as stream:
        yaml.safe_dump(config, stream, sort_keys=True)
    logger.info(
        "Resuming from {} completed shards ({} rows)", completed_shards, completed_rows
    )
    if args.take is not None and completed_rows >= args.take:
        logger.info("Requested take={} is already complete", args.take)
        return
    _write_usage(args.output, usage)
    try:
        _run_pipeline(args, endpoint, token, usage, completed_shards, completed_rows)
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
    args: PrepareConfig,
    endpoint: str,
    token: str,
    usage: UsageTracker,
    completed_shards: int,
    completed_rows: int,
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
    written = failed = with_failures = decision_failures = skipped = 0
    with ExitStack() as stack:
        client = stack.enter_context(OpenAI(base_url=endpoint, api_key=token))
        provider = LlmAnnotationProvider(
            client=client,
            config=args.annotation,
            max_inflight=args.workers,
            usage=usage,
        )
        decision_provider = None
        if args.skip_early_decision:
            logger.info("Early decision disabled; annotating every passage")
        else:
            decision_client = stack.enter_context(
                OpenAI(base_url=decision_endpoint, api_key=token)
            )
            decision_provider = EarlyDecisionProvider(
                client=decision_client,
                config=args.early_decision,
                max_inflight=args.workers,
                usage=usage,
            )

        def annotate(passage: RawPassage) -> PreparedPassage:
            decided = PassageWithEarlyDecision(passage.text, None)
            if decision_provider is not None:
                decided = decision_provider.decide(passage)
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
            return PreparedPassage(provider.annotate(decided))

        for index, results in enumerate(
            annotate_batches(
                rows,
                annotate,
                take=args.take,
                shard_size=args.shard_size,
                workers=args.workers,
                completed_rows=completed_rows,
            ),
            start=completed_shards,
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
        "Finished this invocation: {} written, {} usable, {} failed, {} with annotation failure "
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
