# prepare-fineweb

Convert a streaming Hugging Face dataset into local, annotated examples using
`EarlyDecisionProvider` followed by `LlmAnnotationProvider`.

## Interface

`prepare-fineweb` loads the repository’s `config/prepare_fineweb.yaml` using Hydra.
Edit the default YAML or supply overrides using `key=value` syntax:

```sh
uv run prepare-fineweb take=20 output=data/experiment-1 annotation.temperature=0.5
uv run prepare-fineweb take=null annotation.model=other-model output=data/experiment-2
uv run prepare-fineweb skip_early_decision=true output=data/annotation-only
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
- `p_annotated_anyway`: `1.0`, a finite probability in `[0, 1]` for fully annotating
  passages classified as clean. Errors and unknown decisions always receive full annotation.
- `early_decision_endpoint`: `https://openrouter.ai/api/alpha`, using `MODEL_TOKEN`.
- `skip_early_decision`: `false`. Set to `true` to bypass the decision client,
  provider, cache, and sampling gate. Every passage receives full annotation,
  regardless of `p_annotated_anyway`, with `decision: null` and empty
  `early_decision_failures`.

The `early_decision` section uses `EarlyDecisionConfig`: `model` defaults to
`typesafe/jev-1.13`, `max_attempts` to `3`, and `timeout` to `120.0`. Its spelling-only
prompt matches the annotation policy. Confidence is retained unchanged without a
threshold. The decision client is separate from the annotation client, allowing
`MODEL_ENDPOINT` to point at another annotation service.

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

Load the dataset with `streaming=True`. Check once that the dataset schema declares a string `text`
field, then rely on that schema when reading rows. A thread pool
processes passages concurrently through shared decision and annotation providers,
with at most `workers` active model requests. Each passage gets a decision first.
Unknown decisions proceed to full annotation with their diagnostics retained.
Clean decisions are independently sampled for full annotation using
`p_annotated_anyway`; at `1.0`, every passage is fully annotated for comparison.
Skipped clean passages remain in output with `annotations: []` and their decision.
Sampling occurs outside both caches; no random seed or cross-run reproducibility
is promised. The annotation model receives only the text, without the decision.
Results retain input order. Requests continue across shard boundaries, with a
bounded window of `shard_size + 2 * workers` queued, active, or out-of-order results,
plus the shard being assembled. An unusually slow passage can eventually fill
this window and pause further submission, keeping memory bounded.
Worker errors are retrieved in input order, so complete shards preceding a failed
passage are yielded and written before the error stops processing.

Write one `part-00000.jsonl`, `part-00001.jsonl`, etc. per input batch. Each line
contains `text`, `annotations`, `failures`, `decision`, and
`early_decision_failures`; each annotation contains `original`,
`corrected`, `span_start`, and `span_end` using half-open Python string offsets.
Keep passages with no spelling mistakes, partial annotations, and exhausted output
failures. `annotations: null` indicates no usable result, while `annotations: []`
indicates either a successful response with no mistakes or a skipped clean passage.
There is no per-passage skipped status. `decision` is null for unknowns, otherwise
it contains `contains_spelling_error` and native `confidence`. Early-decision
diagnostics retain attempts, errors and raw responses separately from annotation
failures. Annotation failure records retain attempt numbers, decoding/resolution
stages, errors, and raw response content. Exhausted transient annotation transport
errors retain the best partial result and diagnostics, or `annotations: null` if
none is usable. Nonretryable HTTP errors, early-decision transport exhaustion,
and other unexpected errors stop processing with a nonzero exit status.

Write the resolved application configuration and normalized endpoint to
`config.yaml` beside the shards. Log per-shard and final counts of written passages, usable results,
failed results, passages with annotation or decision failure history (including
recovered results), and skipped annotations.
`usage.json` sums reported input, output, total, cached, and reasoning tokens,
response counts, reported cost, and responses missing token or cost metadata,
separately per stage and overall. Cached and reasoning tokens are subsets of the
token totals. Missing costs are not estimated or treated as known zero charges.
Because requests overlap shard writes, usage snapshots can include responses for
passages in later shards; the final snapshot includes all completed responses.
Each completed file replaces its destination atomically. Every invocation requires
a new output directory; an existing directory raises `FileExistsError` before
loading data. An interrupted run may leave completed shards and requires a new
output directory for another invocation.

PyArrow is capped at 24.0.0 to retain datasets' workaround for crashes when
stopping Parquet streams early (huggingface/datasets#8793).

## Caching

Joblib `Memory` stores decision and annotation results separately under
`<output>/.cache`. Decision keys include text, the full decision configuration,
normalized decision endpoint and decision-cache version. Annotation keys include
the passage with its decision and decision diagnostics, the full annotation
configuration (including prompt, sampling, timeout and attempt budget), normalized
annotation endpoint and annotation-cache version.
Provider objects, clients, and credentials are excluded from cache identity.
Partial and failed output results are cached with their diagnostics; raised errors
are not cached. Bump the cache version when annotation resolution rules or output
types change; prompt changes are hashed directly. Replacing model weights behind
the same endpoint/model identifier requires removing the cache or choosing a new
output directory.

Repeated passages within a run reuse cached results. Cache artifacts are
retained in the output directory, but the command does not resume existing runs.
