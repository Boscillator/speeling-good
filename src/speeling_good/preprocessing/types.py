from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class RawPassage:
    """Minimally processed data from a dataset"""

    text: str


@dataclass(frozen=True, slots=True)
class EarlyDecision:
    """Spelling judgment and the endpoint's native Choice confidence (0–1)."""

    contains_spelling_error: bool
    confidence: float


@dataclass(frozen=True, slots=True)
class EarlyDecisionFailure:
    """An invalid decision response; attempt numbers start at one."""

    attempt: int
    error: str
    response: str | None


@dataclass(frozen=True, slots=True)
class PassageWithEarlyDecision:
    """A spelling judgment; None means no usable decision was obtained."""

    text: str
    decision: EarlyDecision | None
    failures: list[EarlyDecisionFailure] = field(default_factory=list)


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
    stage: Literal["decoding", "resolution", "transport"]
    error: str
    response: str | None


@dataclass(frozen=True, slots=True)
class AnnotatedPassage:
    """Retain the decision and diagnostics alongside full or partial annotations.

    None means no usable annotation result; an empty list also covers clean
    passages whose full annotation was skipped by the preparation pipeline.
    """

    text: str
    annotations: list[SpellingAnnotation] | None
    failures: list[AnnotationFailure] = field(default_factory=list)
    decision: EarlyDecision | None = None
    early_decision_failures: list[EarlyDecisionFailure] = field(default_factory=list)
