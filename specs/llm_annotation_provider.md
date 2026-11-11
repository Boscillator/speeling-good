`LlmAnnotationProvider` is a class that lives in `src/speeling_good/preprocessing/llm_annotation_provider`.
The goal is to use an LLM to annotated `RawPassages`.

The important API surfaces is as follows
```python
class LlmAnnotationProvider:
  
  def __init__(self, *,
    client: OpenAI, # or OpenAIAsync
    other_config_as_needed...): ...

  def annotate(self, passage: RawPassage) -> AnnotatedPassage: ...
```

## Behaviour
- The class calls out to an open-ai compatible LLM endpoint via the `openai` library.
- It asks the LLM to identify all spelling mistakes in the provided passage.
- It uses structured generation with pydanic models to constrain the output.
- The output should have the following fields per correction: `original`, `correction`, `before`, `after`.
- The model should be prompted to correct spelling mistakes only, not grammer mistakes. Furthermore,
  correctly spelled proper nouns, techical terms, etc, are not spelling mistakes.
- The client should use the context return to find the actual span in the passage.

## Retries, caching and parrallelism
- The client retries invalid output and transient connection/HTTP failures within
  `max_attempts`, including the initial request. Invalid output receives validation
  feedback. Other HTTP errors fail immediately; transport exhaustion still raises.
- Caching is defered for now.
- The client should expect to be called from multiple python threads. Furthermore, it should limit the number
  of inflight requests to a configurable number.

## Results and failure diagnostics

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

