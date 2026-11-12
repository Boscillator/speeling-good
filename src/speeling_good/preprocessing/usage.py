"""Thread-safe accounting of token usage reported during a pipeline run."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite
from threading import Lock
from typing import Literal

type UsageStage = Literal["early_decision", "annotation"]


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0
    responses: int = 0
    responses_without_usage: int = 0
    responses_without_cost: int = 0


def _token_count(usage: Mapping[str, object], *names: str) -> int | None:
    for name in names:
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


class UsageTracker:
    """Sum reported tokens, including retries across resumed runs.

    Missing counts contribute zero; totals fall back to input plus output when
    omitted. A response is counted as missing usage if it has no valid token
    counts. Detail counts are subsets, not additional tokens. Cost sums reported
    charges without estimating missing values; missing cost is counted separately.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._stages: dict[UsageStage, _Usage] = {
            "early_decision": _Usage(),
            "annotation": _Usage(),
        }

    @classmethod
    def from_snapshot(cls, snapshot: object) -> "UsageTracker":
        """Restore reported stage totals; overall totals are derived from stages."""
        if not isinstance(snapshot, Mapping):
            raise TypeError("Usage snapshot must be an object")
        tracker = cls()
        for stage, usage in tracker._stages.items():
            counts = snapshot.get(stage)
            if not isinstance(counts, Mapping):
                raise TypeError(f"Missing usage stage: {stage}")
            for name in asdict(usage):
                value = counts.get(name)
                valid = (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and isfinite(value)
                    and value >= 0
                    and (name == "cost" or isinstance(value, int))
                )
                if not valid:
                    raise ValueError(f"Invalid usage count: {stage}.{name}")
                setattr(usage, name, value)
        return tracker

    def record(self, stage: UsageStage, usage: object) -> None:
        counts = usage if isinstance(usage, Mapping) else {}
        input_tokens = _token_count(counts, "input_tokens", "prompt_tokens")
        output_tokens = _token_count(counts, "output_tokens", "completion_tokens")
        total_tokens = _token_count(counts, "total_tokens")
        prompt_details = counts.get("prompt_tokens_details")
        completion_details = counts.get("completion_tokens_details")
        cached_tokens = (
            _token_count(prompt_details, "cached_tokens")
            if isinstance(prompt_details, Mapping)
            else None
        )
        reasoning_tokens = (
            _token_count(completion_details, "reasoning_tokens")
            if isinstance(completion_details, Mapping)
            else None
        )
        reported_cost = counts.get("cost")
        cost = (
            float(reported_cost)
            if isinstance(reported_cost, (int, float))
            and not isinstance(reported_cost, bool)
            and isfinite(reported_cost)
            and reported_cost >= 0
            else None
        )
        missing = (
            input_tokens is None and output_tokens is None and total_tokens is None
        )
        if total_tokens is None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
        with self._lock:
            result = self._stages[stage]
            result.input_tokens += input_tokens or 0
            result.output_tokens += output_tokens or 0
            result.total_tokens += total_tokens
            result.cached_tokens += cached_tokens or 0
            result.reasoning_tokens += reasoning_tokens or 0
            result.cost += cost or 0.0
            result.responses += 1
            result.responses_without_usage += missing
            result.responses_without_cost += cost is None

    def snapshot(self) -> dict[str, dict[str, int | float]]:
        """Return an independent, consistent JSON-ready summary."""
        with self._lock:
            stages = {stage: asdict(usage) for stage, usage in self._stages.items()}
        total = {
            name: sum(stage[name] for stage in stages.values())
            for name in asdict(_Usage())
        }
        return {"total": total, **stages}
