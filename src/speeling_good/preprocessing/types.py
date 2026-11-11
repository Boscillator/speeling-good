from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class RawPassage:
    """Minimally processed data from a dataset"""

    text: str


type AnnotationSource = Literal["original_data"] | str
"""Name of the model that performed the annotation, or original_data if this dataset contains spelling annotations"""


@dataclass(frozen=True, slots=True)
class SpellingAnnotation:
    """Correction for a single word (or words) of a passage"""

    original: str
    corrected: str
    span_start: int
    span_end: int


@dataclass(frozen=True, slots=True)
class AnnotationFailure:
    """An invalid model response; attempt numbers start at one."""

    attempt: int
    stage: Literal["decoding", "resolution"]
    error: str
    response: str | None


@dataclass(frozen=True, slots=True)
class AnnotatedPassage:
    """Annotations may be partial; None means no usable result was obtained."""

    text: str
    annotations: list[SpellingAnnotation] | None
    failures: list[AnnotationFailure] = field(default_factory=list)
