import pytest

from speeling_good.preprocessing.llm_annotation_provider import (
    Correction,
    resolve_partial,
)
from speeling_good.preprocessing.types import SpellingAnnotation


@pytest.mark.parametrize(
    "text,corrections,expected",
    [
        pytest.param(
            "🐈 teh cat saw teh dog and wierd birds.",
            [
                Correction(original="wierd", correction="weird", before="", after=""),
                Correction(
                    original="teh", correction="the", before="saw ", after=" dog"
                ),
                Correction(
                    original="teh", correction="the", before="🐈 ", after=" cat"
                ),
            ],
            [
                SpellingAnnotation("teh", "the", 2, 5),
                SpellingAnnotation("teh", "the", 14, 17),
                SpellingAnnotation("wierd", "weird", 26, 31),
            ],
            id="repeated-words-unicode-and-sorting",
        ),
        pytest.param(
            "teh\n\twierd",
            [
                Correction(
                    original="teh", correction="the", before="", after="\n\twierd"
                ),
                Correction(
                    original="wierd", correction="weird", before="teh\n\t", after=""
                ),
            ],
            [
                SpellingAnnotation("teh", "the", 0, 3),
                SpellingAnnotation("wierd", "weird", 5, 10),
            ],
            id="boundaries-and-overlapping-verbatim-context",
        ),
        pytest.param(
            "He had alot of icecreem.",
            [
                Correction(original="alot", correction="a lot", before="", after=""),
                Correction(
                    original="icecreem", correction="ice cream", before="", after="."
                ),
            ],
            [
                SpellingAnnotation("alot", "a lot", 7, 11),
                SpellingAnnotation("icecreem", "ice cream", 15, 23),
            ],
            id="length-changing-and-multiword-corrections",
        ),
        pytest.param(
            "teh,wrng",
            [
                Correction(original="teh,", correction="the,", before="", after=""),
                Correction(original="wrng", correction="wrong", before="", after=""),
            ],
            [
                SpellingAnnotation("teh,", "the,", 0, 4),
                SpellingAnnotation("wrng", "wrong", 4, 8),
            ],
            id="adjacent-spans-are-not-overlapping",
        ),
    ],
)
def test_resolve_spans(text, corrections, expected):
    originals = [item.model_copy(deep=True) for item in corrections]
    result, errors = resolve_partial(text, list(enumerate(corrections)))

    assert result == expected
    assert errors == []
    assert all(
        text[item.span_start : item.span_end] == item.original for item in result
    )
    assert corrections == originals


@pytest.mark.parametrize(
    "text,corrections,expected",
    [
        pytest.param(
            "aaa",
            [Correction(original="aa", correction="a", before="", after="")],
            [],
            id="overlapping-context-occurrences-are-ambiguous",
        ),
        pytest.param(
            "teh\tcat",
            [Correction(original="teh", correction="the", before="", after=" cat")],
            [],
            id="context-whitespace-must-match-exactly",
        ),
        pytest.param(
            "teh cat",
            [
                Correction(original="teh", correction="the", before="", after=""),
                Correction(original="teh", correction="the", before="", after=" cat"),
            ],
            [],
            id="duplicate-span-with-different-context",
        ),
        pytest.param(
            "teh cat",
            [
                Correction(original="teh", correction="the", before="", after=""),
                Correction(original="missing", correction="word", before="", after=""),
            ],
            [SpellingAnnotation("teh", "the", 0, 3)],
            id="invalid-correction-preserves-unrelated-edit",
        ),
    ],
)
def test_partial_resolution_handles_invalid_context_and_spans(
    text, corrections, expected
):
    annotations, errors = resolve_partial(text, list(enumerate(corrections)))

    assert annotations == expected
    assert errors


def correction(original, corrected, before="", after=""):
    return Correction(
        original=original, correction=corrected, before=before, after=after
    )


@pytest.mark.parametrize(
    "invalid,reason",
    [
        pytest.param([correction("teh", "the")], "ambiguous", id="ambiguous"),
        pytest.param([correction("missing", "word")], "missing", id="missing"),
        pytest.param([correction("teh", "teh")], "unchanged", id="unchanged"),
        pytest.param([correction("", "word")], "nonempty", id="empty-original"),
        pytest.param([correction("teh", "")], "nonempty", id="empty-correction"),
        pytest.param(
            [correction("teh", "the", after=" cat")] * 2,
            "duplicate",
            id="duplicate-span",
        ),
        pytest.param(
            [correction("teh", "the", after=" cat"), correction("teh cat", "the cat")],
            "overlapping",
            id="overlapping-spans",
        ),
    ],
)
def test_partial_resolution_preserves_unrelated_corrections(invalid, reason):
    corrections = invalid + [correction("wierd", "weird")]
    originals = [item.model_copy(deep=True) for item in corrections]

    annotations, errors = resolve_partial(
        "teh cat, teh dog, wierd", list(enumerate(corrections))
    )

    assert annotations == [SpellingAnnotation("wierd", "weird", 18, 23)]
    assert errors
    assert reason in errors[0]
    assert "corrections.0" in errors[0]
    if reason in ("duplicate", "overlapping"):
        assert "corrections.1" in errors[0]
    assert corrections == originals


def test_partial_resolution_excludes_every_correction_in_a_containing_span():
    corrections = [
        correction("teh cat", "the cat"),
        correction("teh", "the"),
        correction("cat", "cats"),
        correction("wierd", "weird"),
    ]

    annotations, errors = resolve_partial("teh cat wierd", list(enumerate(corrections)))

    assert annotations == [SpellingAnnotation("wierd", "weird", 8, 13)]
    assert any(
        "corrections.0" in error and "corrections.1" in error for error in errors
    )
    assert any(
        "corrections.0" in error and "corrections.2" in error for error in errors
    )


def test_partial_resolution_preserves_original_response_indices():
    annotations, errors = resolve_partial(
        "teh cat",
        [(2, correction("teh", "the")), (5, correction("absent", "word"))],
    )

    assert annotations == [SpellingAnnotation("teh", "the", 0, 3)]
    assert len(errors) == 1
    assert "corrections.5" in errors[0]
