"""Fast spelling judgments through OpenRouter's shared Decisions API."""

import time
from threading import BoundedSemaphore
from typing import Annotated, Literal

from loguru import logger
from openai import APIConnectionError, APIStatusError, OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from speeling_good.preprocessing.llm_annotation_provider import _retry_delay
from speeling_good.preprocessing.types import (
    EarlyDecision,
    EarlyDecisionFailure,
    PassageWithEarlyDecision,
    RawPassage,
)


class EarlyDecisionConfig(BaseModel):
    """Decision settings; prompt supplies the spelling-only classification rules."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    model: str
    prompt: str
    max_attempts: Annotated[int, Field(gt=0)] = 3
    timeout: Annotated[float, Field(gt=0)] = 120.0

    @field_validator("model", "prompt")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class _ChoiceAnswer(BaseModel):
    """Validate the fields consumed by this provider, allowing API metadata."""

    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    type: Literal["choice"]
    choice: Literal["error", "clean"]
    confidence: Annotated[float, Field(ge=0, le=1)]


class _Answers(BaseModel):
    model_config = ConfigDict(strict=True)

    spelling: _ChoiceAnswer


class _DecisionResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    answers: _Answers


class EarlyDecisionProvider:
    """Decide synchronously; one provider may be shared across threads.

    Supply an OpenAI client with OpenRouter's ``https://openrouter.ai/api/alpha``
    base URL. Both Jev and Luna Decisions use the same state/questions contract.
    Choice confidence is copied from the endpoint without applying a threshold.

    ``max_attempts`` includes the first request; SDK retries are disabled on a
    client copy. Invalid output exhausts to an unknown decision with diagnostics.
    Transient transport errors retry, then raise; other HTTP errors raise at once.
    Backoff matches the annotation provider. Only active HTTP calls occupy a
    concurrency slot. The caller owns the client and its connection pool.
    """

    def __init__(
        self,
        *,
        client: OpenAI,
        config: EarlyDecisionConfig,
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

    def decide(self, passage: RawPassage) -> PassageWithEarlyDecision:
        failures: list[EarlyDecisionFailure] = []
        for attempt in range(self._config.max_attempts):
            try:
                response = self._request_decision(passage.text)
            except (APIConnectionError, APIStatusError) as error:
                self._retry_or_raise(error, attempt)
                continue

            try:
                answer = _DecisionResponse.model_validate_json(
                    response
                ).answers.spelling
            except ValidationError as error:
                failures.append(EarlyDecisionFailure(attempt + 1, str(error), response))
                if attempt + 1 == self._config.max_attempts:
                    logger.warning(
                        "Early decision exhausted {} attempts; returning unknown",
                        attempt + 1,
                    )
                    return PassageWithEarlyDecision(passage.text, None, failures)
                self._retry_or_raise(error, attempt)
                continue

            decision = EarlyDecision(answer.choice == "error", answer.confidence)
            return PassageWithEarlyDecision(passage.text, decision, failures)
        raise AssertionError("Unreachable: max_attempts is positive")

    def _request_decision(self, text: str) -> str:
        """Read raw JSON so malformed responses can be retained and retried."""
        with self._semaphore:
            return self._client.post(
                "/decisions",
                cast_to=str,
                body={
                    "model": self._config.model,
                    "state": text,
                    "questions": {
                        "spelling": {
                            "type": "choice",
                            "instructions": self._config.prompt,
                            "criteria": {
                                "error": "The passage contains at least one spelling error.",
                                "clean": "The passage contains no spelling errors.",
                            },
                        }
                    },
                },
            )

    def _retry_or_raise(
        self,
        error: ValidationError | APIConnectionError | APIStatusError,
        attempt: int,
    ) -> None:
        if isinstance(error, APIStatusError) and not (
            error.status_code in (408, 409, 429) or error.status_code >= 500
        ):
            logger.error("Early decision request failed ({})", type(error).__name__)
            raise error
        if attempt + 1 == self._config.max_attempts:
            logger.error(
                "Early decision failed after {} attempts ({})", attempt + 1, error
            )
            raise error

        delay = _retry_delay(attempt, error)
        logger.warning(
            "Early decision attempt {}/{} failed ({}); retrying in {:.2f}s",
            attempt + 1,
            self._config.max_attempts,
            error,
            delay,
        )
        time.sleep(delay)
