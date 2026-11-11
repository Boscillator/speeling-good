from threading import Event

import pytest

from speeling_good.cli.prepare_fineweb import annotate_batches
from speeling_good.preprocessing.types import (
    AnnotatedPassage,
    AnnotationFailure,
    SpellingAnnotation,
)


def test_batches_are_bounded_and_count_failed_inputs_toward_take():
    consumed = []
    failure = AnnotationFailure(1, "decoding", "Invalid JSON", "{")

    def rows():
        for text in ("first", "invalid", "third", "fourth", "unrequested"):
            consumed.append(text)
            yield {"text": text}

    def annotate(passage):
        if passage.text == "invalid":
            return AnnotatedPassage(passage.text, None, [failure])
        if passage.text == "third":
            return AnnotatedPassage(
                passage.text, [SpellingAnnotation("third", "thirds", 0, 5)], [failure]
            )
        return AnnotatedPassage(passage.text, [])

    batches = annotate_batches(rows(), annotate, take=4, shard_size=2, workers=2)
    first = next(batches)
    assert consumed == ["first", "invalid"]
    assert first == [
        AnnotatedPassage("first", []),
        AnnotatedPassage("invalid", None, [failure]),
    ]
    assert list(batches) == [
        [
            AnnotatedPassage(
                "third", [SpellingAnnotation("third", "thirds", 0, 5)], [failure]
            ),
            AnnotatedPassage("fourth", []),
        ]
    ]
    assert consumed == ["first", "invalid", "third", "fourth"]


def test_concurrent_results_keep_input_order_with_a_partial_final_batch():
    second_finished = Event()

    def annotate(passage):
        if passage.text == "first":
            assert second_finished.wait(5), "Annotations did not run concurrently"
        elif passage.text == "second":
            second_finished.set()
        return AnnotatedPassage(passage.text, [])

    batches = list(
        annotate_batches(
            [{"text": text} for text in ("first", "second", "third")],
            annotate,
            take=None,
            shard_size=2,
            workers=2,
        )
    )
    assert batches == [
        [AnnotatedPassage("first", []), AnnotatedPassage("second", [])],
        [AnnotatedPassage("third", [])],
    ]


def test_nonvalidation_errors_stop_processing():
    def annotate(passage):
        raise ConnectionError("server unavailable")

    with pytest.raises(ConnectionError, match="server unavailable"):
        list(
            annotate_batches(
                [{"text": "hello"}],
                annotate,
                take=None,
                shard_size=2,
                workers=2,
            )
        )
