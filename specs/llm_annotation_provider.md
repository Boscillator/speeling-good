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
- The client should automatically retry failed attempts with resonable policies (todo: document policies)
- Caching is defered for now.
- The client should expect to be called from multiple python threads. Furthermore, it should limit the number
  of inflight requests to a configurable number.

