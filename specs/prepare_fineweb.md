# prepare-fineweb

Convert a streaming Hugging Face dataset into local, annotated examples using
`LlmAnnotationProvider`.

## Interface

`prepare-fineweb` loads the repository’s `config/prepare_fineweb.yaml` using Hydra.
Edit the default YAML or supply overrides using `key=value` syntax:

```sh
uv run prepare-fineweb take=20 output=data/experiment-1 annotation.temperature=0.5
uv run prepare-fineweb take=null annotation.model=other-model output=data/experiment-2
uv run prepare-fineweb --cfg job --resolve
uv run prepare-fineweb --config-dir=/absolute/path/to/configs --config-name=experiment
```

The default configuration is loaded from the source checkout, independently of
the working directory. For an installation without the checkout, supply
`--config-path=/absolute/path/to/config` to select an external configuration folder.

For a custom configuration, copy the default YAML into the config directory as
`experiment.yaml` and edit it. Previous argparse flags are replaced by Hydra
options; `take=null` replaces `--take all`.

Top-level settings use these defaults:

- `dataset`: `HuggingFaceFW/fineweb`.
- `subset`: dataset `name`, `sample-10BT`.
- `split`: `train`.
- `take`: `100`; a positive input count including failures, or `null` for all.
- `output`: `data/fineweb`; each invocation requires a fresh directory.
- `workers`: `2` positive annotation threads.
- `shard_size`: `10` positive input entries per shard.

The `annotation` section controls model requests:

- `model`: `z-ai/glm-5.3-flash:floor`.
- `prompt`: the evaluated strict spelling-only system prompt, stored as multiline YAML.
- `temperature`: `0.0`, within `[0, 2]`, or `null` to omit it for models that do not
  support temperature.
- `top_p`: `null`, or a value within `[0, 1]`.
- `max_completion_tokens`: `8192`, or `null` to omit the cap; otherwise a positive
  integer.
- `reasoning_effort`: `low` by default; accepts `null`, `none`, `minimal`, `low`,
  `medium`, `high`, `xhigh`, or `max`. An explicit value is sent as OpenRouter's
  `reasoning.effort`; support depends on the model. Null keeps the server default,
  which may incur reasoning token charges.
- `seed`: `null`, or an integer; reproducibility depends on the server.
- `max_attempts`: `3`, including the initial request.
- `timeout`: `180.0` positive, finite seconds per request.

Null optional sampling controls are omitted from requests. The output schema and
validation feedback remain fixed. Counts, model settings, and unknown fields are
validated before output creation or external requests. `--help` and `--cfg job`
work without endpoint credentials.

The sole built-in profile uses the winning settings evaluated on the first 100
FineWeb entries. Its spelling policy, measured costs, remaining non-spelling edits,
and evaluation limits are documented in the
[profile evaluation](fineweb_annotation_profile_evaluation.md).

Hydra keeps the working directory unchanged. Ordinary runs create no separate
Hydra snapshots or log files; application logging continues through Loguru.
For Hydra multiruns, supply distinct output paths, for example:

```sh
uv run prepare-fineweb -m annotation.temperature=0,0.5 'output=data/sweep-${hydra:job.num}'
```

Set `MODEL_ENDPOINT` and `MODEL_TOKEN`. The endpoint may include `/v1/` or omit
it. Credentials are never written to the configuration or cache.

## Pipeline and sharding

Load the dataset with `streaming=True` and read consecutive batches of at most
`shard_size` entries. Check once that the dataset schema declares a string `text`
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

Write the resolved application configuration and normalized endpoint to
`config.yaml` beside the shards. Log per-shard and final counts of written passages, usable results,
failed results, and passages with failure history (including recovered results).
Each completed file replaces its destination atomically. Every invocation requires
a new output directory; an existing directory raises `FileExistsError` before
loading data. An interrupted run may leave completed shards and requires a new
output directory for another invocation.

PyArrow is capped at 24.0.0 to retain datasets' workaround for crashes when
stopping Parquet streams early (huggingface/datasets#8793).

## Caching

Joblib `Memory` stores returned annotation results under `<output>/.cache`, keyed by
passage text, the full annotation configuration (including prompt, sampling, timeout,
and attempt budget), normalized endpoint, and an explicit annotation-cache version.
Provider objects, clients, and credentials are excluded from cache identity.
Partial and failed output results are cached with their diagnostics; raised errors
are not cached. Bump the cache version when annotation resolution rules or output
types change; prompt changes are hashed directly. Replacing model weights behind
the same endpoint/model identifier requires removing the cache or choosing a new
output directory.

Repeated passages within a run reuse cached results. Cache artifacts are
retained in the output directory, but the command does not resume existing runs.
