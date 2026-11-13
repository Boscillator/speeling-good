"""BIO span evaluation and passage-level diagnostics.

The positive class is a misspelling. Span matches require identical token
boundaries; character offsets are used only to display those token spans.
"""

from collections.abc import Sequence
from typing import Any

import torch
from torchmetrics import Metric

Span = tuple[int, int]
TABLE_COLUMNS: list[str | int] = [
    "passage",
    "evaluated_end",
    "predicted_spans",
    "reference_spans",
    "true_positives",
    "false_positives",
    "false_negatives",
    "marked_passage",
]


def bio_spans(labels: Sequence[int]) -> set[Span]:
    """Decode half-open token spans, interpreting an orphan I as a new span."""
    spans: set[Span] = set()
    start: int | None = None
    for index, label in enumerate([*labels, 0]):
        if start is not None and label != 2:
            spans.add((start, index))
            start = None
        if label == 1 or (label == 2 and start is None):
            start = index
    return spans


class SpanDetectionMetrics(Metric):
    """Pool exact span counts and evaluated words across batches and workers."""

    full_state_update = False
    counts: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "counts", default=torch.zeros(4, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, predicted: set[Span], reference: set[Span], words: int) -> None:
        self.counts += self.counts.new_tensor(
            [
                len(predicted & reference),
                len(predicted - reference),
                len(reference - predicted),
                words,
            ]
        )

    def compute(self) -> dict[str, torch.Tensor]:
        tp, fp, fn, words = self.counts.float().unbind()
        return {
            "misspelling_span_precision": tp / (tp + fp).clamp_min(1),
            "misspelling_span_recall": tp / (tp + fn).clamp_min(1),
            "misspelling_span_f1": 2 * tp / (2 * tp + fp + fn).clamp_min(1),
            "false_positive_spans_per_1000_words": torch.where(
                words > 0, 1000 * fp / words.clamp_min(1), 0.0
            ),
        }


def evaluated_word_count(
    text: str, offsets: Sequence[Sequence[int]], labels: Sequence[int]
) -> int:
    end = max(
        end for (_, end), label in zip(offsets, labels, strict=True) if label != -100
    )
    return len(text[:end].split())


def passage_evaluation(
    text: str,
    offsets: Sequence[Sequence[int]],
    labels: Sequence[int],
    predicted: set[Span],
    reference: set[Span],
) -> list[Any]:
    """Build a table row whose TP/FP/FN marks use the scalar metric matches."""

    def describe(spans: set[Span], predicted_side: bool) -> list[dict[str, Any]]:
        result = []
        for start, end in sorted(spans):
            char_start, char_end = offsets[start][0], offsets[end - 1][1]
            status = (
                "TP"
                if (start, end) in predicted & reference
                else ("FP" if predicted_side else "FN")
            )
            result.append(
                {
                    "start": char_start,
                    "end": char_end,
                    "text": text[char_start:char_end],
                    "status": status,
                }
            )
        return result

    predictions, references = describe(predicted, True), describe(reference, False)
    # Boundary mismatches can overlap: event markers preserve both FP and FN.
    events: dict[int, list[str]] = {}
    for span in predictions + [span for span in references if span["status"] == "FN"]:
        events.setdefault(span["start"], []).append(f"[{span['status']}]")
        events.setdefault(span["end"], []).append(f"[/{span['status']}]")
    end = max(
        end for (_, end), label in zip(offsets, labels, strict=True) if label != -100
    )
    marked, previous = [], 0
    for position in sorted(events):
        marked.extend([text[previous:position], *events[position]])
        previous = position
    marked.append(text[previous:end])
    return [
        text,
        end,
        predictions,
        references,
        len(predicted & reference),
        len(predicted - reference),
        len(reference - predicted),
        "".join(marked),
    ]
