"""Measure recursive correction and independent annotation location stability."""

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from difflib import SequenceMatcher
from itertools import chain
from pathlib import Path
from threading import Lock, local

import yaml
from loguru import logger
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from speeling_good.preprocessing.llm_annotation_provider import (
    AnnotationConfig,
    LlmAnnotationProvider,
)
from speeling_good.preprocessing.types import PassageWithEarlyDecision
from speeling_good.preprocessing.usage import UsageTracker


def sample_passages(source: Path, size: int, seed: int) -> tuple[list[dict], int]:
    """Reservoir-sample original texts, without consulting stored annotations."""
    rng = random.Random(seed)
    sample = []
    count = 0
    for shard in sorted(source.glob("part-*.jsonl")):
        with shard.open() as stream:
            for line_number, line in enumerate(stream, 1):
                text = json.loads(line)["text"]
                item = {
                    "source": str(shard),
                    "line": line_number,
                    "source_index": count,
                    "text": text,
                }
                count += 1
                if len(sample) < size:
                    sample.append(item)
                else:
                    index = rng.randrange(count)
                    if index < size:
                        sample[index] = item
    if len(sample) != size:
        raise ValueError(f"Only {len(sample)} passages available")
    sample.sort(key=lambda row: row["source_index"])
    for index, item in enumerate(sample):
        item["id"] = index
    return sample, count


def changed_tokens(text: str, edit: dict) -> set[int]:
    """Locate actual changes, excluding unchanged context within broad spans.

    Whitespace-delimited source tokens are location anchors. Insertions on a
    boundary touch both neighbors; insertions inside a word touch that word.
    """
    tokens = list(re.finditer(r"\S+", text))
    touched = set()
    matcher = SequenceMatcher(None, edit["original"], edit["corrected"], autojunk=False)
    for tag, start, end, _, _ in matcher.get_opcodes():
        if tag == "equal":
            continue
        start += edit["span_start"]
        end += edit["span_start"]
        changed = {
            index
            for index, token in enumerate(tokens)
            if (
                token.start() <= start <= token.end()
                if start == end
                else token.start() < end and start < token.end()
            )
        }
        touched.update(changed)
        # A whitespace-only change belongs to the neighboring source words.
        if not changed:
            left = [i for i, token in enumerate(tokens) if token.end() <= start]
            right = [i for i, token in enumerate(tokens) if token.start() >= end]
            if left:
                touched.add(left[-1])
            if right:
                touched.add(right[0])
    return touched


def compare_locations(text: str, first: list[dict], second: list[dict]) -> dict:
    """An edit is different only if no edit in the other run touches its location.

    No one-to-one constraint: splitting one edit into several does not itself
    create a location disagreement.
    """
    first_tokens = [changed_tokens(text, edit) for edit in first]
    second_tokens = [changed_tokens(text, edit) for edit in second]
    only_first = [
        edit
        for edit, tokens in zip(first, first_tokens, strict=True)
        if not any(tokens & other for other in second_tokens)
    ]
    only_second = [
        edit
        for edit, tokens in zip(second, second_tokens, strict=True)
        if not any(tokens & other for other in first_tokens)
    ]
    return {
        "different_locations": bool(only_first or only_second),
        "only_first": only_first,
        "only_second": only_second,
        "first_edit_count": len(first),
        "second_edit_count": len(second),
        "first_clean": not first,
        "second_clean": not second,
    }


class RecordingProvider(LlmAnnotationProvider):
    """Keep the production provider behavior, recording every request and retry."""

    def __init__(self, *, output: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self.output = output
        self.context = local()
        self.write_lock = Lock()

    def write(self, name: str, row: dict) -> None:
        with self.write_lock, (self.output / name).open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _request_completion(
        self, messages: list[ChatCompletionMessageParam]
    ) -> ChatCompletion:
        started = time.monotonic()
        record = dict(self.context.run)
        try:
            completion = super()._request_completion(messages)
        except Exception as error:
            record["error"] = str(error)
            raise
        else:
            record["completion"] = completion.model_dump(mode="json")
            return completion
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            self.write("requests.jsonl", record)

    def run_pass(self, passage_id: int, phase: str, number: int, text: str) -> dict:
        self.context.run = {"id": passage_id, "phase": phase, "pass": number}
        result = super().annotate(PassageWithEarlyDecision(text, None))
        row = {**self.context.run, **asdict(result)}
        self.write("passes.jsonl", row)
        return row


def experiment(provider: RecordingProvider, passage: dict, max_passes: int) -> dict:
    passage_id, text = passage["id"], passage["text"]
    first = provider.run_pass(passage_id, "recursive", 1, text)
    second = provider.run_pass(passage_id, "independent", 1, text)
    comparison = (
        compare_locations(text, first["annotations"], second["annotations"])
        if first["annotations"] is not None and second["annotations"] is not None
        else None
    )
    current = first
    seen = {text}
    repeated_states = 0
    correction_counts = []
    status = "pass_limit"
    for number in range(1, max_passes + 1):
        edits = current["annotations"]
        if edits is None:
            status = "unusable"
            break
        correction_counts.append(len(edits))
        if not edits:
            status = "clean"
            break
        if number == max_passes:
            break
        for edit in sorted(edits, key=lambda item: item["span_start"], reverse=True):
            start, end = edit["span_start"], edit["span_end"]
            assert text[start:end] == edit["original"]
            text = text[:start] + edit["corrected"] + text[end:]
        repeated_states += text in seen
        seen.add(text)
        current = provider.run_pass(passage_id, "recursive", number + 1, text)
    result = {
        "id": passage_id,
        "status": status,
        "passes": number,
        "correction_counts": correction_counts,
        "repeated_states": repeated_states,
        "comparison": comparison,
        "final_text": text,
    }
    provider.write("results.jsonl", result)
    return result


def summarize(results: list[dict]) -> dict:
    valid = [row["comparison"] for row in results if row["comparison"] is not None]
    return {
        "completed_passages": len(results),
        "status": dict(Counter(row["status"] for row in results)),
        "passes_until_clean": dict(
            sorted(
                Counter(
                    row["passes"] for row in results if row["status"] == "clean"
                ).items()
            )
        ),
        "comparison_passages": len(valid),
        "different_location_passages": sum(row["different_locations"] for row in valid),
        "clean_both": sum(row["first_clean"] and row["second_clean"] for row in valid),
        "clean_exactly_one": sum(
            row["first_clean"] != row["second_clean"] for row in valid
        ),
        "edits_both": sum(
            not row["first_clean"] and not row["second_clean"] for row in valid
        ),
        "first_edits": sum(row["first_edit_count"] for row in valid),
        "second_edits": sum(row["second_edit_count"] for row in valid),
        "unmatched_first_edits": sum(len(row["only_first"]) for row in valid),
        "unmatched_second_edits": sum(len(row["only_second"]) for row in valid),
        "repeated_states": sum(row["repeated_states"] for row in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/fineweb-2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-passes", type=int, default=30)
    args = parser.parse_args()
    if min(args.size, args.workers, args.max_passes) < 1:
        parser.error("size, workers and max-passes must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    passages, population = sample_passages(args.source, args.size, args.seed)
    with (args.output / "sample.jsonl").open("w") as stream:
        for passage in passages:
            stream.write(json.dumps(passage, ensure_ascii=False) + "\n")
    config = AnnotationConfig.model_validate(
        yaml.safe_load(Path("config/prepare_fineweb.yaml").read_text())["annotation"]
    )
    metadata = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "population": population,
        "annotation": config.model_dump(),
        "endpoint": os.environ["MODEL_ENDPOINT"],
    }
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    usage = UsageTracker()
    started = time.monotonic()
    results = []
    with (
        OpenAI(
            base_url=os.environ["MODEL_ENDPOINT"], api_key=os.environ["MODEL_TOKEN"]
        ) as client,
        ThreadPoolExecutor(max_workers=args.workers) as executor,
    ):
        provider = RecordingProvider(
            output=args.output,
            client=client,
            config=config,
            max_inflight=args.workers,
            usage=usage,
        )
        # Validate endpoint, schema and credentials before scheduling the sample.
        first = experiment(provider, passages[0], args.max_passes)
        pending = [
            executor.submit(experiment, provider, passage, args.max_passes)
            for passage in passages[1:]
        ]
        completed = (future.result() for future in as_completed(pending))
        for result in chain((first,), completed):
            results.append(result)
            summary = {
                **summarize(results),
                "usage": usage.snapshot(),
                "elapsed_seconds": time.monotonic() - started,
            }
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            if len(results) % 10 == 0:
                logger.info(
                    "Completed {}/{} passages; {}",
                    len(results),
                    args.size,
                    summary["passes_until_clean"],
                )
    logger.info("Results: {}", args.output / "summary.json")


if __name__ == "__main__":
    main()
