from dataclasses import dataclass
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
class AnnotatedPassage:
    text: str
    annotations: list[SpellingAnnotation]
