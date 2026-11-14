"""Source-based span contracts, independent of model or research artifacts."""

import pytest

from speeling_good.preprocessing.annotation_postprocessing import (
    canonicalize_edit,
    is_in_scope,
)
from speeling_good.preprocessing.types import CorrectionType, SpellingAnnotation


@pytest.mark.parametrize(
    "text,original,corrected,expected",
    [
        ("a tees worth", "a tees worth", "a tee's worth", [("tees", "tee's")]),
        ("didnt.", "didnt.", "didn't.", [("didnt", "didn't")]),
        ("'teh'", "'teh'", "'the'", [("teh", "the")]),
        ("‘teh’", "‘teh’", "‘the’", [("teh", "the")]),
        ("'the teh dog'", "'the teh dog'", "'the the dog'", [("teh", "the")]),
        ("'teh'", "teh'", "the'", [("teh", "the")]),
        ("‘teh’", "teh’", "the’", [("teh", "the")]),
        ("dogs", "dogs", "dogs'", [("dogs", "dogs'")]),
        ("dogs'", "dogs'", "dogs", [("dogs'", "dogs")]),
        ("dogs'", "dogs'", "dog's", [("dogs'", "dog's")]),
        ("dogs’", "dogs’", "dog’s", [("dogs’", "dog’s")]),
        ("'dogs'", "'dogs'", "'dog's'", [("dogs", "dog's")]),
        ("hair legs", "hair legs", "hairy legs", [("hair", "hairy")]),
        ("super-yaught", "super-yaught", "super-yacht", [("yaught", "yacht")]),
        ("possess-ing", "possess-ing", "possessing", [("possess-ing", "possessing")]),
        ("the the", "the the", "the", [("the the", "the")]),
        ("the the.", "the the.", "the.", [("the the", "the")]),
        (
            "I saw the the cat.",
            "I saw the the cat.",
            "I saw the cat.",
            [("the the", "the")],
        ),
        ("The the", "The the", "The", [("The the", "The")]),
        ("A RE", "A RE", "ARE", [("A RE", "ARE")]),
        (
            "asbatteries",
            "asbatteries",
            "as batteries",
            [("asbatteries", "as batteries")],
        ),
        ("Video.I", "Video.I", "Video. I", [("Video.I", "Video. I")]),
        ("bags,take", "bags,take", "bags, take", [("bags,take", "bags, take")]),
        ("I saw Video.I", "I saw Video.I", "I saw Video. I", [("Video.I", "Video. I")]),
        ("bags,take", "take", " take", [("bags,take", "bags, take")]),
        (
            "its Harvards ",
            "its Harvards ",
            " it's Harvard's",
            [("its Harvards ", " it's Harvard's")],
        ),
        ("Thirty one", "Thirty ", "Thirty-", [("Thirty one", "Thirty-one")]),
        ("wrong", "wr", "wro", [("wrong", "wroong")]),
        ("don't", "don", "doesn", [("don't", "doesn't")]),
        ("8,000", "000", "001", [("8,000", "8,001")]),
        ("8,000", "8", "9", [("8,000", "9,000")]),
        ("a lot", " ", "", [("a lot", "alot")]),
        (
            "its Harvards",
            "its Harvards",
            "it's Harvard's",
            [("its", "it's"), ("Harvards", "Harvard's")],
        ),
        (
            "its\tHarvards",
            "its\tHarvards",
            "it's\tHarvard's",
            [("its", "it's"), ("Harvards", "Harvard's")],
        ),
        (
            "its Harvards",
            "its Harvards",
            "it's  Harvard's",
            [("its Harvards", "it's  Harvard's")],
        ),
    ],
)
def test_canonical_spans_preserve_complete_correction(
    text, original, corrected, expected
):
    start = text.index(original)
    proposal = SpellingAnnotation(
        original, corrected, start, start + len(original), CorrectionType.SPELLING
    )
    edits = canonicalize_edit(proposal, text)
    assert [(edit.original, edit.corrected) for edit in edits] == expected
    result = text
    for edit in reversed(edits):
        assert text[edit.span_start : edit.span_end] == edit.original
        result = result[: edit.span_start] + edit.corrected + result[edit.span_end :]
    assert result == text[:start] + corrected + text[start + len(original) :]


@pytest.mark.parametrize(
    "original,corrected,accepted",
    [
        ("that", "on that", False),
        ("terrific", "terrific results", False),
        ("house", "the house", False),
        ("cat", "a cats", False),
        ("a cat", "cats", False),
        ("a house", "house", False),
        ("ice cream", "ice", False),
        ("icecreem", "ice cream", True),
        ("ice cream", "icecreem", True),
        ("icecrem", "ice cream", True),
        ("icecreamz", "ice cream", True),
        ("icecraem", "ice cream", True),
        ("something", "some thing", True),
        ("situtation room", "situationroom", True),
        ("terrific", "very nice", False),
        ("not a thing", "nothing", False),
        ("one two three four", "one two three five", False),
        ("one two one", "one two", False),
        ("hello,", "hello;", False),
        ('"hello"', "'hello'", False),
        ("“hello”", '"hello"', False),
        ("the the", "the", True),
        ("The the", "The", True),
        ("year round", "year-round", True),
        ("A RE", "ARE", True),
        ("Video.I", "Video. I", True),
        ("appear", "appearing", True),
        ("Harvards", "Harvard's", True),
    ],
)
def test_local_scope_policy(original, corrected, accepted):
    annotation = SpellingAnnotation(
        original, corrected, 0, len(original), CorrectionType.GRAMMAR
    )
    assert is_in_scope(annotation) is accepted
