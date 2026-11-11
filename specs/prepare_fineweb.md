# prepare-fineweb

Convert a streaming Hugging Face dataset into local, annotated examples using
`LlmAnnotationProvider`.

## Interface

`prepare-fineweb` accepts:

- `--dataset`: dataset identifier (default: `HuggingFaceFW/fineweb`).
- `--subset`: dataset `name` (default: `sample-10BT`).
- `--split`: dataset split (default: `train`).
- `--take`: positive number of input entries to attempt, including validation
  failures (default: `100`); `all` removes the limit.
- `--model`: annotation model (default: `qwen/qwen3.8-27b`).
- `--output`: output directory (default: `data/fineweb`).
- `--workers`: positive number of annotation threads (default: `4`).
- `--shard-size`: positive number of input entries per shard (default: `100`).

Set `MODEL_ENDPOINT` and `MODEL_TOKEN`. The endpoint may include `/v1/` or omit
it. Credentials are never written to the configuration or cache.

## Pipeline and sharding

Load the dataset with `streaming=True` and read consecutive batches of at most
`shard-size` entries. Check once that the dataset schema declares a string `text`
field, then rely on that schema when reading rows. Joblib threads
annotate each batch concurrently through a shared provider, with at most `workers`
active model requests. Results retain input order, and only one batch is held in
memory at a time.

Write one `part-00000.jsonl`, `part-00001.jsonl`, etc. per input batch. Each line
contains `text`, `annotations`, and `failures`; each annotation contains `original`,
`corrected`, `span_start`, and `span_end` using half-open Python string offsets.
Keep passages with no spelling mistakes, partial annotations, and exhausted output
failures. `annotations: null` indicates no usable result, while `annotations: []`
indicates a successful response with no mistakes. Failure records retain attempt
numbers, decoding/resolution stages, errors, and raw response content. Transport
and other unexpected errors stop processing with a nonzero exit status.

Write resolved CLI options and the normalized endpoint to `config.yaml` beside
the shards. Log per-shard and final counts of written passages, usable results,
failed results, and passages with failure history (including recovered results).
Each completed file replaces its destination atomically. Every invocation requires
a new output directory; an existing directory raises `FileExistsError` before
loading data. An interrupted run may leave completed shards and requires a new
output directory for another invocation.

PyArrow is capped at 24.0.0 to retain datasets' workaround for crashes when
stopping Parquet streams early (huggingface/datasets#8793).

## Caching

Joblib `Memory` stores returned annotation results under `<output>/.cache`, keyed by
passage text, model, normalized endpoint, and an explicit annotation-cache version.
Provider objects, clients, and credentials are excluded from cache identity.
Partial and failed output results are cached with their diagnostics; raised errors
are not cached. Bump the cache version when annotation
prompts, resolution rules, or output types change. Replacing model weights behind
the same endpoint/model identifier requires removing the cache or choosing a new
output directory.

Repeated passages within a run reuse cached results. Cache artifacts are
retained in the output directory, but the command does not resume existing runs.
