import pytest

from speeling_good.evaluation import (
    TABLE_COLUMNS,
    SpanDetectionMetrics,
    bio_spans,
    evaluated_word_count,
    passage_evaluation,
)


@pytest.mark.parametrize(
    "predicted,reference,expected",
    [
        ([0, 1, 2, 0], [0, 1, 2, 0], (1, 1, 1)),
        ([0, 1, 0, 0], [0, 1, 2, 0], (0, 0, 0)),
        ([1, 1, 2, 0], [1, 1, 2, 0], (1, 1, 1)),
        ([2, 2, -100, 2], [1, 2, -100, 1], (1, 1, 1)),
        ([0, 0], [0, 0], (0, 0, 0)),
        ([1, 0], [0, 0], (0, 0, 0)),
    ],
)
def test_exact_span_matching_and_bio_boundaries(predicted, reference, expected):
    metrics = SpanDetectionMetrics()
    metrics.update(bio_spans(predicted), bio_spans(reference), words=2)
    result = metrics.compute()
    assert (
        tuple(
            float(result[f"misspelling_span_{name}"])
            for name in ("precision", "recall", "f1")
        )
        == expected
    )


def test_pool_counts_across_passages_and_reset():
    metrics = SpanDetectionMetrics()
    metrics.update({(0, 1)}, {(0, 1)}, words=2)
    metrics.update({(0, 1), (2, 3)}, {(1, 2)}, words=8)
    result = metrics.compute()
    assert float(result["misspelling_span_precision"]) == pytest.approx(1 / 3)
    assert float(result["misspelling_span_recall"]) == pytest.approx(1 / 2)
    assert float(result["misspelling_span_f1"]) == pytest.approx(2 / 5)
    assert float(result["false_positive_spans_per_1000_words"]) == 200
    metrics.reset()
    metrics.update(set(), {(0, 1)}, words=3)
    assert all(float(value) == 0 for value in metrics.compute().values())


def test_unicode_subwords_and_truncated_passage_diagnostics():
    text = "🐈 speling okay tail"
    offsets = [(0, 0), (0, 1), (2, 5), (5, 9), (10, 12), (0, 0), (0, 0)]
    labels = [-100, 0, 1, 2, 0, -100, -100]
    predictions = [-100, 1, 1, 0, 0, -100, -100]
    row = dict(
        zip(
            TABLE_COLUMNS,
            passage_evaluation(
                text, offsets, labels, bio_spans(predictions), bio_spans(labels)
            ),
            strict=True,
        )
    )
    assert evaluated_word_count(text, offsets, labels) == 3
    assert row["evaluated_end"] == 12
    assert (row["true_positives"], row["false_positives"], row["false_negatives"]) == (
        0,
        2,
        1,
    )
    assert row["reference_spans"] == [
        {"start": 2, "end": 9, "text": "speling", "status": "FN"}
    ]
    assert row["predicted_spans"] == [
        {"start": 0, "end": 1, "text": "🐈", "status": "FP"},
        {"start": 2, "end": 5, "text": "spe", "status": "FP"},
    ]
    assert "[FP]" in row["marked_passage"] and "[FN]" in row["marked_passage"]
    assert "tail" not in row["marked_passage"]


def test_passage_marks_distinguish_matches_from_missed_spans():
    row = dict(
        zip(
            TABLE_COLUMNS,
            passage_evaluation(
                "teh wrng", [(0, 3), (4, 8)], [1, 1], {(0, 1)}, {(0, 1), (1, 2)}
            ),
            strict=True,
        )
    )
    assert row["marked_passage"] == "[TP]teh[/TP] [FN]wrng[/FN]"
    assert (row["true_positives"], row["false_positives"], row["false_negatives"]) == (
        1,
        0,
        1,
    )
