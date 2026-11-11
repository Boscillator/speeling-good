`LlmAnnotationProvider` is a class that lives in `src/speeling_good/preprocessing/llm_annotation_provider`.
The goal is to use an LLM to annotate `PassageWithEarlyDecision` inputs.

The important API surface is as follows
```python
class LlmAnnotationProvider:
    def __init__(
        self, *, client: OpenAI, config: AnnotationConfig, max_inflight: int = 4
    ): ...

    def annotate(self, passage: PassageWithEarlyDecision) -> AnnotatedPassage: ...
```

## Configuration

The constructor requires `client` and a validated `config: AnnotationConfig`.
The config requires `model` and `prompt`. The CLI's default system prompt is
defined in the repository’s `config/prepare_fineweb.yaml`; other callers supply their
own prompt or load that configuration. The provider retains the supplied config
without serializing or validating it again.

Optional config fields are `temperature=0.0`, `top_p=None`,
`max_completion_tokens=None`, `seed=None`, `max_attempts=3`, and
`timeout=120.0`. Sampling controls apply to every request, including retries; null
optional controls are omitted from the request. Structured output remains fixed.
`AnnotationConfig` validates nonempty model/prompt, finite sampling ranges and
positive timeout, completion-token limit, and attempt budget. Request concurrency
is controlled separately by the constructor argument `max_inflight=4`.

## Behaviour
- The class calls out to an OpenAI compatible LLM endpoint via the `openai` library.
- It asks the LLM to identify all spelling mistakes in the provided passage.
- It uses structured generation with Pydantic models to constrain the output.
- The output should have the following fields per correction: `original`, `correction`, `before`, `after`.
- The model should be prompted to correct spelling mistakes only, not grammar mistakes. Furthermore,
  correctly spelled proper nouns, technical terms, etc, are not spelling mistakes.
- The client should use the context returned to find the actual span in the passage.

## Retries, caching and parallelism
- The client retries invalid output and transient connection/HTTP failures within
  `max_attempts`, including the initial request. Invalid output receives validation
  feedback. Other HTTP errors fail immediately; transport exhaustion still raises.
- Caching belongs to the preparation CLI; the provider does not cache results.
- The client should expect to be called from multiple python threads. Furthermore, it should limit the number
  of inflight requests to a configurable number.

## Results and failure diagnostics

Every result preserves the input `decision` and copies its diagnostics to
`early_decision_failures`, including partial and exhausted annotation results. Only
passage text is sent to the model; early judgments do not influence its prompt.

`AnnotatedPassage.failures` defaults to an empty list. Each `AnnotationFailure`
contains the 1-based request attempt, stage (`decoding` or `resolution`), diagnostic
`error`, and full raw model `response` content when available. Diagnostics identify
invalid correction indices and fields, missing or ambiguous context, unchanged or
empty edits, and duplicate or overlapping spans. Refusal text and unexpected finish
reasons are also retained. Transport errors are not added to this history.

Decode and validate entries independently when the JSON envelope is valid. Resolve
valid corrections independently, excluding every correction involved in a duplicate
or overlap. Retry the passage if any entry fails. A fully valid response replaces
earlier partial results, including when it contains no corrections; failure history
is retained even after recovery.

On output exhaustion, return the attempt with the most recoverable annotations,
preferring the latest on ties. Never merge attempts. `annotations=None` indicates
that no usable result was obtained; `annotations=[]` indicates a successful response
with no mistakes. Partial results contain both annotations and failures. Failures
describe all invalid attempts, not necessarily the selected attempt alone, so a
nonempty failure history can accompany either a partial or recovered result.

The public `resolve_partial(text, corrections)` function accepts
`(original_index, Correction)` pairs and returns `(annotations, errors)`.
Indices are unique, zero-based positions in the original response, allowing gaps
when schema-invalid entries were skipped. It preserves unrelated valid corrections
and excludes all duplicates or overlaps. Neither response histories nor annotations
are shared across concurrent calls.

