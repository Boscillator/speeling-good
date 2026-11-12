import pytest

from speeling_good.dataloaders.annotation_to_bio import annotation_to_bio
from speeling_good.preprocessing.types import SpellingAnnotation


@pytest.mark.parametrize(
    "offsets,spans,expected",
    [
        pytest.param(
            [(0, 0), (0, 2), (2, 4), (5, 7), (7, 9), (10, 13), (0, 0)],
            [(5, 9), (0, 4)],
            [-100, 1, 2, 1, 2, 0, -100],
            id="split-words-and-unsorted-errors",
        ),
        pytest.param(
            [(0, 3), (3, 6), (6, 9), (9, 12)],
            [(2, 7)],
            [1, 2, 2, 0],
            id="partial-token-overlaps",
        ),
        pytest.param(
            [(0, 2), (2, 4), (4, 6)],
            [(2, 4), (4, 6)],
            [0, 1, 1],
            id="adjacent-errors-and-exclusive-ends",
        ),
        pytest.param(
            [(0, 5), (5, 8)],
            [(3, 8), (0, 2)],
            [1, 1],
            id="shared-token-prefers-earliest-error",
        ),
        pytest.param(
            [(0, 4), (4, 7), (7, 9)],
            [(3, 9), (1, 6)],
            [1, 2, 1],
            id="overlapping-annotations",
        ),
        pytest.param(
            [(0, 0), (6, 8), (8, 10), (0, 0)],
            [(0, 2), (4, 12), (20, 22)],
            [-100, 1, 2, -100],
            id="truncation-keeps-only-visible-error-tokens",
        ),
        pytest.param(
            [(0, 1), (0, 1), (2, 4)],
            [(0, 1)],
            [1, 2, 0],
            id="multiple-byte-tokens-for-one-code-point",
        ),
    ],
)
def test_align_error_spans(offsets, spans, expected):
    annotations = [SpellingAnnotation("teh", "the", start, end) for start, end in spans]
    original_annotations = annotations.copy()
    original_offsets = offsets.copy()

    assert annotation_to_bio(offsets, annotations) == expected
    assert annotations == original_annotations
    assert offsets == original_offsets


def test_unicode_code_point_offsets_and_combining_marks():
    text = "🐈 cafe\u0301 speling"
    accent = text.index("\u0301")
    error = text.index("speling")
    annotations = [
        SpellingAnnotation("\u0301", "", accent, accent + 1),
        SpellingAnnotation("speling", "spelling", error, len(text)),
    ]
    offsets = [
        (0, 1),
        (2, accent),
        (accent, accent + 1),
        (error, error + 3),
        (error + 3, len(text)),
    ]

    assert annotation_to_bio(offsets, annotations) == [0, 0, 1, 1, 2]


@pytest.mark.parametrize(
    "spans,expected", [([], [-100, 0, 0, -100]), ([(0, 5)], [-100, 1, 2, -100])]
)
def test_ignore_special_tokens_with_nonempty_offsets_and_padding(spans, expected):
    offsets = [(0, 2), (0, 2), (2, 5), (0, 0)]
    annotations = [
        SpellingAnnotation("wrng", "wrong", start, end) for start, end in spans
    ]

    assert (
        annotation_to_bio(offsets, annotations, special_tokens_mask=[1, 0, 0, 1])
        == expected
    )


@pytest.mark.parametrize("span", [(-1, 3), (3, 2), (2, 2)])
def test_invalid_annotation_spans(span):
    with pytest.raises(ValueError, match="annotation span"):
        annotation_to_bio([(0, 4)], [SpellingAnnotation("teh", "the", *span)])


def test_mask_length_must_match_tokens():
    with pytest.raises(ValueError, match="one entry per token"):
        annotation_to_bio([(0, 2), (2, 5)], [], special_tokens_mask=[0])
