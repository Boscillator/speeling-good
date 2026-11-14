"""Deterministic annotation boundaries and local spell-checker scope policy.

Inputs are edits already resolved exactly against the unchanged source passage.
Canonicalization preserves their complete correction transformation; the scope
policy then decides which local edits to retain, independently of predicted type.
"""

import re
from dataclasses import replace

from speeling_good.preprocessing.types import SpellingAnnotation


def _complete_source_span(
    annotation: SpellingAnnotation, text: str
) -> SpellingAnnotation:
    """Complete clipped source tokens without changing the proposed repair."""
    original, corrected = annotation.original, annotation.corrected
    start, end = annotation.span_start, annotation.span_end
    # A model can return changed letters or stop before the second half of
    # a hyphenation. Expand against the source before trimming equal context.
    left, right = start, end
    if (
        corrected[:1].isspace()
        and start > 1
        and text[start - 1] in ".,;:!?"
        and text[start - 2].isalnum()
    ):
        left = start - 1
        while left > 0 and text[left - 1].isalnum():
            left -= 1
    for word in re.finditer(r"\d+(?:,\d+)+|\w+(?:['’]\w+)*", text):
        overlaps = word.start() < end and word.end() > start
        joins_left = original.isspace() and word.end() == start
        joins_right = (
            original.isspace() or corrected.endswith("-")
        ) and word.start() == end
        if overlaps or joins_left or joins_right:
            left = min(left, word.start())
            right = max(right, word.end())
    if (left, right) != (start, end):
        annotation = replace(
            annotation,
            original=text[left:right],
            corrected=text[left:start] + corrected + text[end:right],
            span_start=left,
            span_end=right,
        )
    return annotation


def _trim_context(annotation: SpellingAnnotation) -> SpellingAnnotation:
    """Remove equal context while retaining changed words and repetitions."""
    original, corrected = annotation.original, annotation.corrected
    # Recognize repetitions before trimming: a deletion must retain both words,
    # even when the proposal also includes unchanged punctuation or context.
    for repeated in re.finditer(r"\b(\w+)\s+\1\b", original, re.IGNORECASE):
        prefix, suffix = original[: repeated.start()], original[repeated.end() :]
        corrected_end = len(corrected) - len(suffix)
        replacement = corrected[len(prefix) : corrected_end]
        if (
            corrected.startswith(prefix)
            and corrected.endswith(suffix)
            and replacement.casefold() == repeated.group(1).casefold()
        ):
            return replace(
                annotation,
                original=repeated.group(),
                corrected=replacement,
                span_start=annotation.span_start + repeated.start(),
                span_end=annotation.span_start + repeated.end(),
            )
    left = 0
    while (
        left < min(len(original), len(corrected)) and original[left] == corrected[left]
    ):
        left += 1
    right, corrected_end = len(original), len(corrected)
    while (
        right > left
        and corrected_end > left
        and original[right - 1] == corrected[corrected_end - 1]
    ):
        right -= 1
        corrected_end -= 1
    # Keep the complete changed token, including apostrophes and numeric groups.
    for word in re.finditer(r"\d+(?:,\d+)+|\w+(?:['’]\w+)*", original):
        if word.start() <= right and word.end() >= left:
            left = min(left, word.start())
            right = max(right, word.end())
    # Unchanged edge quotes stay in context. Internal apostrophes are already
    # part of the word pattern; apostrophes changed by the repair stay in the span.
    # A missing space at an internal punctuation boundary includes both words.
    if (
        corrected[left : len(corrected) - (len(original) - right)][:1].isspace()
        and left > 0
        and original[left - 1] in ".,;:!?"
        and left > 1
        and original[left - 2].isalnum()
    ):
        left -= 1
        while left > 0 and original[left - 1].isalnum():
            left -= 1
    end_new = len(corrected) - (len(original) - right)
    if not original[left:right] or original[left:right] == corrected[left:end_new]:
        return annotation
    return replace(
        annotation,
        original=original[left:right],
        corrected=corrected[left:end_new],
        span_start=annotation.span_start + left,
        span_end=annotation.span_start + right,
    )


def _is_spelling_resegmentation(original: list[str], corrected: list[str]) -> bool:
    """Allow a short typo repair combined with a word join or split.

    One side must be a single word, and that word must change: preserving it
    while adding or deleting other words is content editing. At most two letter
    edits are allowed, with only one for words shorter than eight characters.
    This is a local scope heuristic, not a spelling or meaning classifier.
    """
    if not original or not corrected or min(len(original), len(corrected)) != 1:
        return False
    if set(original) & set(corrected):
        return False
    source, replacement = "".join(original), "".join(corrected)
    limit = min(2, max(1, min(len(source), len(replacement)) // 4))
    if abs(len(source) - len(replacement)) > limit:
        return False
    # Levenshtein distance counts substitutions, insertions and deletions alike.
    previous = list(range(len(replacement) + 1))
    for row, source_char in enumerate(source, start=1):
        current = [row]
        for column, replacement_char in enumerate(replacement, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (source_char != replacement_char),
                )
            )
        if min(current) > limit:
            return False
        previous = current
    return previous[-1] <= limit


def is_in_scope(annotation: SpellingAnnotation) -> bool:
    """Reject added/deleted content and punctuation-only edits, without a model.

    This deliberately does not decide whether a name, real word, or hyphenation
    is correct. Apply after whole-word normalization, never as a type whitelist.
    """
    original = re.findall(r"\w+(?:['’]\w+)*", annotation.original)
    corrected = re.findall(r"\w+(?:['’]\w+)*", annotation.corrected)
    compact_old = re.sub(r"[\s-]", "", annotation.original)
    compact_new = re.sub(r"[\s-]", "", annotation.corrected)
    adjacent_repeat = (
        len(original) == 2
        and original[0].casefold() == original[1].casefold()
        and [word.casefold() for word in corrected] == [original[0].casefold()]
    )
    if (
        len(original) != len(corrected)
        and compact_old != compact_new
        and not adjacent_repeat
        and not _is_spelling_resegmentation(original, corrected)
    ):
        return False
    if len(original) > 3:
        return False
    if '"' in annotation.original + annotation.corrected and re.sub(
        r"['’\"]", "", annotation.original
    ) == re.sub(r"['’\"]", "", annotation.corrected):
        return False
    # Apostrophes/hyphens/spacing remain in scope; other punctuation alone does not.
    strip_punctuation = r"[.,;:!?()\[\]{}\"“”]"
    return re.sub(strip_punctuation, "", annotation.original) != re.sub(
        strip_punctuation, "", annotation.corrected
    )


def canonicalize_edit(
    annotation: SpellingAnnotation, text: str
) -> list[SpellingAnnotation]:
    """Separate independent word edits when token alignment is unambiguous."""
    annotation = _complete_source_span(annotation, text)
    annotation = _trim_context(annotation)
    original, corrected = (
        list(re.finditer(r"\S+", annotation.original)),
        list(re.finditer(r"\S+", annotation.corrected)),
    )
    if len(original) != len(corrected) or len(original) < 2:
        return [annotation]
    if re.findall(r"\s+", annotation.original) != re.findall(
        r"\s+", annotation.corrected
    ):
        return [annotation]
    if (
        annotation.original[: original[0].start()]
        != annotation.corrected[: corrected[0].start()]
        or annotation.original[original[-1].end() :]
        != annotation.corrected[corrected[-1].end() :]
    ):
        return [annotation]
    changed = [
        (source_token, corrected_token)
        for source_token, corrected_token in zip(original, corrected, strict=True)
        if source_token.group() != corrected_token.group()
    ]
    if len(changed) < 2:
        return [annotation]
    edits: list[SpellingAnnotation] = []
    for source_token, corrected_token in changed:
        edit = replace(
            annotation,
            original=source_token.group(),
            corrected=corrected_token.group(),
            span_start=annotation.span_start + source_token.start(),
            span_end=annotation.span_start + source_token.end(),
        )
        edit = _complete_source_span(edit, text)
        edits.append(_trim_context(edit))
    return edits
