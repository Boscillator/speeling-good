`EarlyDecisionProvider` uses a decision model to quickly and cheaply evaluate whether
a passage contains any spelling errors. The preparation pipeline uses it as a gate before full annotation.

The important api surface is:

```python
class EarlyDecisionProvider:
    def __init__(
        self, *, client: OpenAI, config: EarlyDecisionConfig, max_inflight: int = 4
    ): ...

    def decide(self, passage: RawPassage) -> PassageWithEarlyDecision: ...
```

## Behaviour

- Use OpenRouter's shared Decisions contract for models such as
  `typesafe/jev-1.13` and `openai/gpt-6-luna-decisions`.
- Supply an `OpenAI` client with base URL `https://openrouter.ai/api/alpha`.
  Call `client.post("/decisions", ...)`; the SDK's typed `decisions.create()`
  uses OpenAI's different native schema.
- Send passage text unchanged as `state`, with one `spelling` Choice question.
  Use the configured prompt as its instructions and `error` / `clean` as criteria.
- Map the selected choice to `EarlyDecision.contains_spelling_error`. Copy the
  native Choice `confidence` unchanged; it measures certainty of the distribution,
  not the probability of a spelling error. Do not apply a confidence threshold.
- The caller supplies spelling-only instructions, preserving grammar, proper nouns,
  technical terms, and other accepted spellings.

## Configuration

`EarlyDecisionConfig` requires nonempty `model` and `prompt`. Optional fields are
`max_attempts=3` and `timeout=120.0` (seconds), both positive. Configuration is strict
and rejects unknown fields and nonfinite numbers. Chat sampling controls do not apply.
The constructor separately requires a positive integer `max_inflight=4`.

## Retries, parallelism, and failures

- Disable SDK retries on a client copy. The caller owns the client and connection pool.
- Match the annotation provider's backoff, including jitter and `Retry-After` capped
  at 30 seconds. The attempt budget includes the initial request.
- Retry connection failures, HTTP 408/409/429 and server errors. Other HTTP errors
  raise immediately; exhausted transport errors raise the last SDK exception.
- Validate the expected named Choice answer, known choice, and finite confidence in
  `[0, 1]`. Unrelated API metadata is allowed. Malformed JSON, missing data, refusals,
  and unexpected answer types are invalid output and retry the same request.
- Record invalid output in `EarlyDecisionFailure(attempt, error, response)`, with
  1-based attempts and the raw response when available. Transport errors are not
  included in this history. Preserve diagnostics after a successful retry.
- On invalid-output exhaustion return `PassageWithEarlyDecision` with `decision=None`.
  A valid clean result has `contains_spelling_error=False`; it differs from unknown.
- Each call owns its failure history. A shared provider supports multiple threads;
  only active HTTP calls occupy semaphore slots, not validation or backoff.
- Caching and gate sampling belong to the preparation CLI, not this provider.

## API references

- [OpenRouter Decisions](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request)
- [TypeSafe confidence](https://docs.typesafe.ai/confidence)

