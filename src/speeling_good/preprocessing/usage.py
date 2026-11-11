"""Thread-safe accounting of token usage reported during a pipeline run."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Literal

type UsageStage = Literal["early_decision", "annotation"]


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    responses: int = 0
    responses_without_usage: int = 0


def _token_count(usage: Mapping[str, object], *names: str) -> int | None:
    for name in names:
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


class UsageTracker:
    """Sum reported tokens, including retries but excluding cached results.

    Missing counts contribute zero; totals fall back to input plus output when
    omitted. A response is counted as missing usage if it has no valid token
    counts. Provider-specific detail counts are subsets, not additional tokens.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._stages: dict[UsageStage, _Usage] = {
            "early_decision": _Usage(),
            "annotation": _Usage(),
        }

    def record(self, stage: UsageStage, usage: object) -> None:
        counts = usage if isinstance(usage, Mapping) else {}
        input_tokens = _token_count(counts, "input_tokens", "prompt_tokens")
        output_tokens = _token_count(counts, "output_tokens", "completion_tokens")
        total_tokens = _token_count(counts, "total_tokens")
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
            result.responses += 1
            result.responses_without_usage += missing

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Return an independent, consistent JSON-ready summary."""
        with self._lock:
            stages = {stage: asdict(usage) for stage, usage in self._stages.items()}
        total = {
            name: sum(stage[name] for stage in stages.values())
            for name in asdict(_Usage())
        }
        return {"total": total, **stages}
