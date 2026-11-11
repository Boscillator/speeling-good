"""Structured spelling annotations from an OpenAI-compatible chat endpoint."""

import math
import random
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from threading import BoundedSemaphore
from typing import Annotated, Literal

from loguru import logger
from openai import (
    APIConnectionError,
    APIStatusError,
    OpenAI,
    omit,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from speeling_good.preprocessing.types import (
    AnnotatedPassage,
    AnnotationFailure,
    PassageWithEarlyDecision,
    SpellingAnnotation,
)


class AnnotationConfig(BaseModel):
    """Validated annotation settings shared by the CLI and provider."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    model: str
    prompt: str
    temperature: Annotated[float, Field(ge=0, le=2)] | None = 0.0
    top_p: Annotated[float, Field(ge=0, le=1)] | None = None
    max_completion_tokens: Annotated[int, Field(gt=0)] | None = None
    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None
    ) = None
    seed: int | None = None
    max_attempts: Annotated[int, Field(gt=0)] = 3
    timeout: Annotated[float, Field(gt=0)] = 120.0

    @field_validator("model", "prompt")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class AnnotationError(ValueError):
    """Corrections cannot be resolved exactly against the passage."""


class Correction(BaseModel):
    """A proposed spelling correction with verbatim adjacent context."""

    model_config = ConfigDict(extra="forbid", strict=True)

    original: str
    correction: str
    before: str
    after: str


class Corrections(BaseModel):
    """Structured spelling corrections for a passage."""

    model_config = ConfigDict(extra="forbid", strict=True)

    corrections: list[Correction]


class _CorrectionEntries(BaseModel):
    """Validate the envelope without discarding independently valid entries."""

    model_config = ConfigDict(extra="forbid", strict=True)

    corrections: list[object]


def _resolve_correction(text: str, item: Correction) -> SpellingAnnotation:
    """Match a single edit exactly, raising an actionable resolution error."""
    if not item.original or not item.correction:
        raise AnnotationError("original and correction must be nonempty")
    if item.original == item.correction:
        raise AnnotationError("correction is unchanged")
    context = item.before + item.original + item.after
    match = text.find(context)
    if match < 0:
        raise AnnotationError(f"exact context {context!r} is missing from passage")
    if text.find(context, match + 1) >= 0:
        raise AnnotationError(f"exact context {context!r} is ambiguous; extend it")
    start = match + len(item.before)
    return SpellingAnnotation(
        item.original, item.correction, start, start + len(item.original)
    )


def _exclude_conflicts(
    annotations: list[tuple[int, SpellingAnnotation]],
) -> tuple[list[SpellingAnnotation], list[str]]:
    """Sort spans and exclude every edit involved in a duplicate or overlap."""
    annotations = sorted(annotations, key=lambda item: item[1].span_start)
    conflicts: set[int] = set()
    errors: list[str] = []
    for position, (left_index, left) in enumerate(annotations):
        for right_index, right in annotations[position + 1 :]:
            if right.span_start >= left.span_end:
                break
            conflicts.update((left_index, right_index))
            reason = (
                "duplicate span"
                if (left.span_start, left.span_end)
                == (right.span_start, right.span_end)
                else "overlapping spans"
            )
            errors.append(
                f"corrections.{left_index} and corrections.{right_index}: {reason}"
            )
    return [item for index, item in annotations if index not in conflicts], errors


def resolve_partial(
    text: str, corrections: list[tuple[int, Correction]]
) -> tuple[list[SpellingAnnotation], list[str]]:
    """Return sorted valid spans and diagnostics without discarding unrelated edits.

    Each correction is paired with its unique, zero-based index in the original
    response, preserving diagnostic locations when schema-invalid entries were
    skipped. Every correction involved in a duplicate or overlap is excluded.
    Neither the text nor the supplied corrections are modified.
    """
    resolved: list[tuple[int, SpellingAnnotation]] = []
    errors: list[str] = []
    for index, item in corrections:
        try:
            resolved.append((index, _resolve_correction(text, item)))
        except AnnotationError as error:
            errors.append(f"corrections.{index}: {error}")
    annotations, conflicts = _exclude_conflicts(resolved)
    return annotations, errors + conflicts


def _response_content(completion: ChatCompletion) -> tuple[str | None, list[str]]:
    """Read content and diagnose missing, refused, or incomplete completions."""
    if not completion.choices:
        return None, ["The response must contain a completion"]
    choice = completion.choices[0]
    errors: list[str] = []
    if choice.finish_reason != "stop":
        errors.append(
            f"The response must be complete (finish_reason={choice.finish_reason})"
        )
    if choice.message.refusal:
        errors.append(f"The model refused: {choice.message.refusal}")
    if choice.message.content is None and not errors:
        errors.append("The response must contain corrections; content is missing")
    return choice.message.content, errors


def _decode_corrections(
    response: str,
) -> tuple[list[tuple[int, Correction]], list[str]]:
    """Validate the JSON envelope and salvage independently valid corrections."""
    try:
        entries = _CorrectionEntries.model_validate_json(response).corrections
    except ValidationError as error:
        return [], [str(error)]
    corrections: list[tuple[int, Correction]] = []
    errors: list[str] = []
    for index, entry in enumerate(entries):
        try:
            corrections.append((index, Correction.model_validate(entry)))
        except ValidationError as error:
            for issue in error.errors(include_url=False):
                location = ".".join(map(str, ("corrections", index, *issue["loc"])))
                errors.append(f"{location}: {issue['msg']} ({issue['type']})")
    return corrections, errors


def _evaluate_response(
    text: str, completion: ChatCompletion, attempt: int
) -> AnnotatedPassage:
    """Decode and resolve output, retaining the original response for each failure."""
    response, errors = _response_content(completion)
    if errors or response is None:
        failures = [
            AnnotationFailure(attempt, "decoding", error, response) for error in errors
        ]
        return AnnotatedPassage(text, None, failures)

    corrections, decoding_errors = _decode_corrections(response)
    annotations, resolution_errors = resolve_partial(text, corrections)
    failures = [
        AnnotationFailure(attempt, "decoding", error, response)
        for error in decoding_errors
    ]
    failures.extend(
        AnnotationFailure(attempt, "resolution", error, response)
        for error in resolution_errors
    )
    return AnnotatedPassage(
        text, annotations if annotations or not failures else None, failures
    )


def _retry_delay(attempt: int, error: Exception) -> float:
    if isinstance(error, APIStatusError):
        header = error.response.headers.get("retry-after")
        if header:
            try:
                delay = float(header)
            except ValueError:
                try:
                    date = parsedate_to_datetime(header)
                    delay = (date - datetime.now(UTC)).total_seconds()
                except (ValueError, TypeError, OverflowError):
                    delay = math.nan
            if math.isfinite(delay):
                return min(30.0, max(0.0, delay))
    return random.uniform(0.5, 1.0) * 2 ** min(attempt, 3)


class LlmAnnotationProvider:
    """Annotate passages synchronously; one instance may be shared across threads.

    ``max_inflight`` limits active requests per provider, not queued callers.
    ``max_attempts`` includes the initial request; SDK retries are disabled on a
    client copy. Each request has a timeout in seconds. Retryable transport errors
    and invalid output use jittered exponential backoff (1–8 seconds before jitter),
    or Retry-After capped at 30 seconds. Other HTTP errors fail immediately.

    Context is matched exactly; ambiguous, missing, duplicate, or overlapping spans
    retry the entire passage with validation feedback. Invalid responses are retained
    in the result. Exhaustion returns the best partial annotations (latest on ties),
    or None if none are usable. Transport exhaustion raises the last SDK exception.
    The caller owns the supplied client and its underlying connection pool.
    """

    def __init__(
        self,
        *,
        client: OpenAI,
        config: AnnotationConfig,
        max_inflight: int = 4,
    ) -> None:
        if (
            isinstance(max_inflight, bool)
            or not isinstance(max_inflight, int)
            or max_inflight < 1
        ):
            raise ValueError("max_inflight must be a positive integer")
        self._config = config
        self._client = client.with_options(max_retries=0, timeout=config.timeout)
        self._semaphore = BoundedSemaphore(max_inflight)

    def annotate(self, passage: PassageWithEarlyDecision) -> AnnotatedPassage:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self._config.prompt},
            {"role": "user", "content": passage.text},
        ]
        failures: list[AnnotationFailure] = []
        best: list[SpellingAnnotation] | None = None
        for attempt in range(self._config.max_attempts):
            try:
                completion = self._request_completion(messages)
            except (APIConnectionError, APIStatusError) as error:
                self._retry_or_raise(error, attempt)
                continue
            result = _evaluate_response(passage.text, completion, attempt + 1)
            failures.extend(result.failures)
            if not result.failures:
                return AnnotatedPassage(
                    passage.text,
                    result.annotations,
                    failures,
                    decision=passage.decision,
                    early_decision_failures=passage.failures,
                )
            if result.annotations and (
                best is None or len(result.annotations) >= len(best)
            ):
                best = result.annotations
            feedback = "; ".join(failure.error for failure in result.failures)
            if attempt + 1 == self._config.max_attempts:
                logger.warning(
                    "Annotation exhausted {} attempts; retaining {} annotations",
                    attempt + 1,
                    len(best or []),
                )
                return AnnotatedPassage(
                    passage.text,
                    best,
                    failures,
                    decision=passage.decision,
                    early_decision_failures=passage.failures,
                )
            messages.append(
                {
                    "role": "user",
                    "content": f"Validation failed: {feedback}. Try again.",
                }
            )
            self._retry_or_raise(AnnotationError(feedback), attempt)
        raise AssertionError("Unreachable: max_attempts is positive")

    def _request_completion(
        self, messages: list[ChatCompletionMessageParam]
    ) -> ChatCompletion:
        """Request schema-constrained output, occupying a slot only during the call."""
        with self._semaphore:
            return self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Corrections",
                        "strict": True,
                        "schema": Corrections.model_json_schema(),
                    },
                },
                temperature=(
                    self._config.temperature
                    if self._config.temperature is not None
                    else omit
                ),
                top_p=self._config.top_p if self._config.top_p is not None else omit,
                max_completion_tokens=(
                    self._config.max_completion_tokens
                    if self._config.max_completion_tokens is not None
                    else omit
                ),
                seed=self._config.seed if self._config.seed is not None else omit,
                extra_body=(
                    {"reasoning": {"effort": self._config.reasoning_effort}}
                    if self._config.reasoning_effort is not None
                    else None
                ),
            )

    def _retry_or_raise(
        self,
        error: AnnotationError | APIConnectionError | APIStatusError,
        attempt: int,
    ) -> None:
        """Apply the retry policy and wait without occupying a request slot."""
        if isinstance(error, APIStatusError) and not (
            error.status_code in (408, 409, 429) or error.status_code >= 500
        ):
            logger.error("Annotation request failed ({})", type(error).__name__)
            raise error
        if attempt + 1 == self._config.max_attempts:
            logger.error("Annotation failed after {} attempts ({})", attempt + 1, error)
            raise error

        delay = _retry_delay(attempt, error)
        logger.warning(
            "Annotation attempt {}/{} failed ({}); retrying in {:.2f}s",
            attempt + 1,
            self._config.max_attempts,
            error,
            delay,
        )
        time.sleep(delay)
