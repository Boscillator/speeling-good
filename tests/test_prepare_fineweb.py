from threading import Event

import pytest
from pydantic import ValidationError

from speeling_good.cli.prepare_fineweb import PrepareConfig, annotate_batches
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


@pytest.mark.parametrize(
    "override,location",
    [
        ({"take": 0}, ("take",)),
        ({"take": "all"}, ("take",)),
        ({"workers": True}, ("workers",)),
        ({"shard_size": -1}, ("shard_size",)),
        ({"annotation": {"model": " "}}, ("annotation", "model")),
        ({"annotation": {"prompt": "\n"}}, ("annotation", "prompt")),
        ({"annotation": {"temperature": 2.1}}, ("annotation", "temperature")),
        ({"annotation": {"temperature": float("nan")}}, ("annotation", "temperature")),
        ({"annotation": {"top_p": -0.1}}, ("annotation", "top_p")),
        ({"annotation": {"top_p": 1.1}}, ("annotation", "top_p")),
        (
            {"annotation": {"max_completion_tokens": 0}},
            ("annotation", "max_completion_tokens"),
        ),
        ({"annotation": {"seed": True}}, ("annotation", "seed")),
        ({"annotation": {"max_attempts": 0}}, ("annotation", "max_attempts")),
        ({"annotation": {"timeout": float("inf")}}, ("annotation", "timeout")),
        ({"annotation": {"timeout": 0}}, ("annotation", "timeout")),
        ({"annotation": {"temprature": 0.5}}, ("annotation", "temprature")),
    ],
)
def test_invalid_run_settings_report_the_configuration_field(override, location):
    config = {
        "dataset": "test",
        "subset": "sample",
        "split": "train",
        "take": 10,
        "output": "unused-output",
        "workers": 2,
        "shard_size": 5,
        "annotation": {"model": "test-model", "prompt": "Find spelling mistakes."},
    }
    annotation = config["annotation"] | override.get("annotation", {})
    config.update(override)
    config["annotation"] = annotation
    with pytest.raises(ValidationError) as raised:
        PrepareConfig.model_validate(config)
    assert location in [error["loc"] for error in raised.value.errors()]
