from threading import Event

import pytest
from pydantic import ValidationError

from speeling_good.cli.prepare_fineweb import (
    PrepareConfig,
    PreparedPassage,
    annotate_batches,
    should_annotate,
)
from speeling_good.preprocessing.types import (
    AnnotatedPassage,
    AnnotationFailure,
    CorrectionType,
    EarlyDecision,
    PassageWithEarlyDecision,
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
            return PreparedPassage(AnnotatedPassage(passage.text, None, [failure]))
        if passage.text == "third":
            return PreparedPassage(
                AnnotatedPassage(
                    passage.text,
                    [
                        SpellingAnnotation(
                            "third",
                            "thirds",
                            0,
                            5,
                            correction_type=CorrectionType.SPELLING,
                        )
                    ],
                    [failure],
                )
            )
        return PreparedPassage(AnnotatedPassage(passage.text, []))

    batches = annotate_batches(rows(), annotate, take=4, shard_size=2, workers=2)
    first = next(batches)
    assert consumed[:2] == ["first", "invalid"]
    assert len(consumed) <= 4
    assert first == [
        PreparedPassage(AnnotatedPassage("first", [])),
        PreparedPassage(AnnotatedPassage("invalid", None, [failure])),
    ]
    assert list(batches) == [
        [
            PreparedPassage(
                AnnotatedPassage(
                    "third",
                    [
                        SpellingAnnotation(
                            "third",
                            "thirds",
                            0,
                            5,
                            correction_type=CorrectionType.SPELLING,
                        )
                    ],
                    [failure],
                )
            ),
            PreparedPassage(AnnotatedPassage("fourth", [])),
        ]
    ]
    assert consumed == ["first", "invalid", "third", "fourth"]


def test_workers_continue_past_a_slow_shard_while_preserving_order():
    third_finished = Event()

    def annotate(passage):
        if passage.text == "first":
            assert third_finished.wait(5), "Next shard was blocked by the first shard"
        elif passage.text == "third":
            third_finished.set()
        return PreparedPassage(AnnotatedPassage(passage.text, []))

    batches = list(
        annotate_batches(
            [
                {"text": text}
                for text in ("first", "second", "third", "fourth", "fifth")
            ],
            annotate,
            take=None,
            shard_size=2,
            workers=2,
        )
    )
    assert [[item.passage.text for item in batch] for batch in batches] == [
        ["first", "second"],
        ["third", "fourth"],
        ["fifth"],
    ]


def test_request_lookahead_is_bounded_and_closing_releases_the_source():
    consumed = []
    closed = Event()

    def rows():
        try:
            for index in range(1000):
                consumed.append(index)
                yield {"text": str(index)}
        finally:
            closed.set()

    batches = annotate_batches(
        rows(),
        lambda passage: PreparedPassage(AnnotatedPassage(passage.text, [])),
        take=None,
        shard_size=2,
        workers=2,
    )
    try:
        assert [item.passage.text for item in next(batches)] == ["0", "1"]
        # One shard under construction plus a six-entry request/result window.
        assert len(consumed) <= 8
    finally:
        batches.close()
    assert closed.is_set()


@pytest.mark.parametrize("failed_index", [2, 4])
def test_later_failure_preserves_earlier_complete_shards(failed_index):
    release_first = Event()

    def annotate(passage):
        index = int(passage.text)
        if index == 0:
            assert release_first.wait(5), "Later passages did not run concurrently"
        elif index == failed_index:
            raise RuntimeError("fatal annotation failure")
        elif index == failed_index + 1:
            # With the first worker blocked, this worker has already finished
            # the failed future before it can release the first passage.
            release_first.set()
        return PreparedPassage(AnnotatedPassage(passage.text, []))

    batches = annotate_batches(
        [{"text": str(index)} for index in range(failed_index + 2)],
        annotate,
        take=None,
        shard_size=2,
        workers=2,
    )
    written = []
    with pytest.raises(RuntimeError, match="fatal annotation failure"):
        for batch in batches:
            written.append([item.passage.text for item in batch])
    assert written == ([["0", "1"]] if failed_index == 2 else [["0", "1"], ["2", "3"]])


def test_concurrent_results_keep_input_order_with_a_partial_final_batch():
    second_finished = Event()

    def annotate(passage):
        if passage.text == "first":
            assert second_finished.wait(5), "Annotations did not run concurrently"
        elif passage.text == "second":
            second_finished.set()
        return PreparedPassage(
            AnnotatedPassage(passage.text, []), skipped=passage.text == "second"
        )

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
        [
            PreparedPassage(AnnotatedPassage("first", [])),
            PreparedPassage(AnnotatedPassage("second", []), skipped=True),
        ],
        [PreparedPassage(AnnotatedPassage("third", []))],
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
        ({"early_decision_endpoint": ""}, ("early_decision_endpoint",)),
        ({"early_decision_endpoint": "   "}, ("early_decision_endpoint",)),
        ({"early_decision_endpoint": "/"}, ("early_decision_endpoint",)),
        ({"p_annotated_anyway": -0.1}, ("p_annotated_anyway",)),
        ({"p_annotated_anyway": 1.1}, ("p_annotated_anyway",)),
        ({"p_annotated_anyway": float("nan")}, ("p_annotated_anyway",)),
        ({"p_annotated_anyway": float("inf")}, ("p_annotated_anyway",)),
        ({"p_annotated_anyway": True}, ("p_annotated_anyway",)),
        (
            {"early_decision": {"model": "", "prompt": "Find errors."}},
            ("early_decision", "model"),
        ),
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
        (
            {"annotation": {"reasoning_effort": "maximum"}},
            ("annotation", "reasoning_effort"),
        ),
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
        "early_decision": {
            "model": "test-decision",
            "prompt": "Find spelling mistakes.",
        },
        "annotation": {"model": "test-model", "prompt": "Find spelling mistakes."},
    }
    annotation = config["annotation"] | override.get("annotation", {})
    config.update(override)
    config["annotation"] = annotation
    with pytest.raises(ValidationError) as raised:
        PrepareConfig.model_validate(config)
    assert location in [error["loc"] for error in raised.value.errors()]


@pytest.mark.parametrize("probability", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("decision", [None, EarlyDecision(True, 0.01)])
def test_errors_and_unknowns_are_always_annotated(monkeypatch, probability, decision):
    def unexpected_sample():
        pytest.fail("Only clean passages should be sampled")

    monkeypatch.setattr(
        "speeling_good.cli.prepare_fineweb.random.random", unexpected_sample
    )
    assert should_annotate(PassageWithEarlyDecision("text", decision), probability)


@pytest.mark.parametrize(
    "probability,draw,expected",
    [(0.0, 0.0, False), (1.0, 0.999, True), (0.4, 0.399, True), (0.4, 0.4, False)],
)
def test_clean_passages_are_sampled_by_probability(
    monkeypatch, probability, draw, expected
):
    monkeypatch.setattr("speeling_good.cli.prepare_fineweb.random.random", lambda: draw)
    passage = PassageWithEarlyDecision("clean", EarlyDecision(False, 1.0))
    assert should_annotate(passage, probability) is expected


@pytest.mark.parametrize(
    "take,expected", [(5, ["3", "4"]), (None, ["3", "4", "5"]), (2, [])]
)
def test_resume_skips_completed_rows_and_keeps_take_as_total(take, expected):
    requested = []

    def annotate(passage):
        requested.append(passage.text)
        return PreparedPassage(AnnotatedPassage(passage.text, []))

    batches = list(
        annotate_batches(
            [{"text": str(index)} for index in range(6)],
            annotate,
            take=take,
            shard_size=2,
            workers=1,
            completed_rows=3,
        )
    )
    assert [item.passage.text for batch in batches for item in batch] == expected
    assert requested == expected
