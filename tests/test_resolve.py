import pytest

from speeling_good.preprocessing.llm_annotation_provider import (
    AnnotationError,
    Correction,
    Corrections,
    resolve,
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
    proposals = Corrections(corrections=corrections)
    original_proposals = proposals.model_copy(deep=True)

    result = resolve(text, proposals)

    assert result == expected
    assert all(
        text[item.span_start : item.span_end] == item.original for item in result
    )
    assert proposals == original_proposals


@pytest.mark.parametrize(
    "text,corrections",
    [
        pytest.param(
            "teh cat, teh dog",
            [Correction(original="teh", correction="the", before="", after="")],
            id="ambiguous-original",
        ),
        pytest.param(
            "aaa",
            [Correction(original="aa", correction="a", before="", after="")],
            id="overlapping-context-occurrences-are-ambiguous",
        ),
        pytest.param(
            "teh cat",
            [Correction(original="teh", correction="the", before="", after=" dog")],
            id="missing-context",
        ),
        pytest.param(
            "teh\tcat",
            [Correction(original="teh", correction="the", before="", after=" cat")],
            id="context-whitespace-must-match-exactly",
        ),
        pytest.param(
            "teh cat",
            [Correction(original="", correction="the", before="", after="teh cat")],
            id="empty-original",
        ),
        pytest.param(
            "teh cat",
            [Correction(original="teh", correction="", before="", after="")],
            id="empty-correction",
        ),
        pytest.param(
            "teh cat",
            [Correction(original="teh", correction="teh", before="", after="")],
            id="unchanged-correction",
        ),
        pytest.param(
            "teh cat",
            [
                Correction(original="teh", correction="the", before="", after=""),
                Correction(original="teh", correction="the", before="", after=" cat"),
            ],
            id="duplicate-span-with-different-context",
        ),
        pytest.param(
            "teh cat",
            [
                Correction(original="teh", correction="the", before="", after=""),
                Correction(
                    original="teh cat", correction="the cat", before="", after=""
                ),
            ],
            id="overlapping-spans",
        ),
        pytest.param(
            "teh cat",
            [
                Correction(original="teh", correction="the", before="", after=""),
                Correction(original="missing", correction="word", before="", after=""),
            ],
            id="invalid-correction-rejects-entire-batch",
        ),
    ],
)
def test_resolve_rejects_invalid_corrections(text, corrections):
    with pytest.raises(AnnotationError):
        resolve(text, Corrections(corrections=corrections))
