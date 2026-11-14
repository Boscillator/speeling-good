import pytest

from speeling_good.preprocessing.llm_annotation_provider import (
    Correction,
    resolve_partial,
)
from speeling_good.preprocessing.types import CorrectionType, SpellingAnnotation


@pytest.mark.parametrize(
    "text,corrections,expected",
    [
        pytest.param(
            "🐈 teh cat saw teh dog and wierd birds.",
            [
                Correction(
                    original="wierd",
                    correction="weird",
                    before="",
                    after="",
                    correction_type=CorrectionType.SPELLING,
                ),
                Correction(
                    original="teh",
                    correction="the",
                    before="saw ",
                    after=" dog",
                    correction_type=CorrectionType.SPELLING,
                ),
                Correction(
                    original="teh",
                    correction="the",
                    before="🐈 ",
                    after=" cat",
                    correction_type=CorrectionType.SPELLING,
                ),
            ],
            [
                SpellingAnnotation(
                    "teh", "the", 2, 5, correction_type=CorrectionType.SPELLING
                ),
                SpellingAnnotation(
                    "teh", "the", 14, 17, correction_type=CorrectionType.SPELLING
                ),
                SpellingAnnotation(
                    "wierd", "weird", 26, 31, correction_type=CorrectionType.SPELLING
                ),
            ],
            id="repeated-words-unicode-and-sorting",
        ),
        pytest.param(
            "teh\n\twierd",
            [
                Correction(
                    original="teh",
                    correction="the",
                    before="",
                    after="\n\twierd",
                    correction_type=CorrectionType.SPELLING,
                ),
                Correction(
                    original="wierd",
                    correction="weird",
                    before="teh\n\t",
                    after="",
                    correction_type=CorrectionType.SPELLING,
                ),
            ],
            [
                SpellingAnnotation(
                    "teh", "the", 0, 3, correction_type=CorrectionType.SPELLING
                ),
                SpellingAnnotation(
                    "wierd", "weird", 5, 10, correction_type=CorrectionType.SPELLING
                ),
            ],
            id="boundaries-and-overlapping-verbatim-context",
        ),
        pytest.param(
            "He had alot of icecreem.",
            [
                Correction(
                    original="alot",
                    correction="a lot",
                    before="",
                    after="",
                    correction_type=CorrectionType.SPELLING,
                ),
                Correction(
                    original="icecreem",
                    correction="ice cream",
                    before="",
                    after=".",
                    correction_type=CorrectionType.SPELLING,
                ),
            ],
            [
                SpellingAnnotation(
                    "alot", "a lot", 7, 11, correction_type=CorrectionType.SPELLING
                ),
                SpellingAnnotation(
                    "icecreem",
                    "ice cream",
                    15,
                    23,
                    correction_type=CorrectionType.SPELLING,
                ),
            ],
            id="length-changing-and-multiword-corrections",
        ),
        pytest.param(
            "teh,wrng",
            [
                Correction(
                    original="teh,",
                    correction="the,",
                    before="",
                    after="",
                    correction_type=CorrectionType.SPELLING,
                ),
                Correction(
                    original="wrng",
                    correction="wrong",
                    before="",
                    after="",
                    correction_type=CorrectionType.SPELLING,
                ),
            ],
            [
                SpellingAnnotation(
                    "teh", "the", 0, 3, correction_type=CorrectionType.SPELLING
                ),
                SpellingAnnotation(
                    "wrng", "wrong", 4, 8, correction_type=CorrectionType.SPELLING
                ),
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
            [
                Correction(
                    original="aa",
                    correction="a",
                    before="",
                    after="",
                    correction_type=CorrectionType.SPELLING,
                )
            ],
            [],
            id="overlapping-context-occurrences-are-ambiguous",
        ),
        pytest.param(
            "teh\tcat",
            [
                Correction(
                    original="teh",
                    correction="the",
                    before="",
                    after=" cat",
                    correction_type=CorrectionType.SPELLING,
                )
            ],
            [],
            id="context-whitespace-must-match-exactly",
        ),
        pytest.param(
            "teh cat",
            [
                Correction(
                    original="teh",
                    correction="the",
                    before="",
                    after="",
                    correction_type=CorrectionType.SPELLING,
                ),
                Correction(
                    original="missing",
                    correction="word",
                    before="",
                    after="",
                    correction_type=CorrectionType.SPELLING,
                ),
            ],
            [
                SpellingAnnotation(
                    "teh", "the", 0, 3, correction_type=CorrectionType.SPELLING
                )
            ],
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
        original=original,
        correction=corrected,
        before=before,
        after=after,
        correction_type=CorrectionType.SPELLING,
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
            [
                correction("teh", "the", after=" cat"),
                correction("teh", "ten", after=" cat"),
            ],
            "duplicate",
            id="duplicate-span",
        ),
        pytest.param(
            [correction("teh", "the", after=" cat"), correction("teh cat", "teh-cat")],
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

    assert annotations == [
        SpellingAnnotation(
            "wierd", "weird", 18, 23, correction_type=CorrectionType.SPELLING
        )
    ]
    assert errors
    assert reason in errors[0]
    assert "corrections.0" in errors[0]
    if reason in ("duplicate", "overlapping"):
        assert "corrections.1" in errors[0]
    assert corrections == originals


def test_normalization_removes_unrelated_context_before_conflict_checks():
    corrections = [
        correction("teh cat", "the cat"),
        correction("teh", "the"),
        correction("cat", "cats"),
        correction("wierd", "weird"),
    ]

    annotations, errors = resolve_partial("teh cat wierd", list(enumerate(corrections)))

    assert annotations == [
        SpellingAnnotation("teh", "the", 0, 3, CorrectionType.SPELLING),
        SpellingAnnotation("cat", "cats", 4, 7, CorrectionType.SPELLING),
        SpellingAnnotation("wierd", "weird", 8, 13, CorrectionType.SPELLING),
    ]
    assert errors == []


def test_partial_resolution_preserves_original_response_indices():
    annotations, errors = resolve_partial(
        "teh cat",
        [(2, correction("teh", "the")), (5, correction("absent", "word"))],
    )

    assert annotations == [
        SpellingAnnotation("teh", "the", 0, 3, correction_type=CorrectionType.SPELLING)
    ]
    assert len(errors) == 1
    assert "corrections.5" in errors[0]


def test_equivalent_normalized_edits_are_deduplicated():
    annotations, errors = resolve_partial(
        "a tees worth",
        [
            (0, correction("a tees worth", "a tee's worth")),
            (1, correction("tees", "tee's")),
        ],
    )
    assert annotations == [
        SpellingAnnotation("tees", "tee's", 2, 6, CorrectionType.SPELLING)
    ]
    assert errors == []


def test_conflict_in_one_split_edit_preserves_its_independent_sibling():
    annotations, errors = resolve_partial(
        "its Harvards",
        [
            (2, correction("its Harvards", "it's Harvard's")),
            (5, correction("its", "his")),
        ],
    )
    assert annotations == [
        SpellingAnnotation("Harvards", "Harvard's", 4, 12, CorrectionType.SPELLING)
    ]
    assert len(errors) == 1
    assert "corrections.2" in errors[0] and "corrections.5" in errors[0]


def test_expanding_clipped_edits_exposes_conflicting_replacements():
    annotations, errors = resolve_partial(
        "wrong",
        [(0, correction("wr", "wro")), (1, correction("wrong", "wrung"))],
    )
    assert annotations == []
    assert len(errors) == 1
