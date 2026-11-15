# Recursive correction and annotation repeatability

On November 14, 2026, 200 original passages were reservoir-sampled from all
50,000 rows in `data/fineweb-2`, using seed `20261114`. Stored annotations were
ignored. All sampled texts were distinct; lengths ranged from 276 to 42,211
characters. This samples the locally available FineWeb prefix, not all FineWeb.

The production annotation provider ran with the settings frozen from
`config/prepare_fineweb.yaml`: `z-ai/glm-5.3-flash:floor`, temperature zero, low
reasoning, an 8,192-token completion limit, three attempts, and eight workers.
Every call used a fresh conversation, the existing prompt, schema, exact resolver,
canonicalization and scope filtering. No early-decision calls were made. The
configured OpenRouter `/api/` URL returned 404, so the experiment used `/api/v1`.

## Passes until the model returns no accepted edits

After each nonempty result, all accepted corrections were applied simultaneously,
and the resulting passage was annotated again. The final empty-result pass is
included in the count; validation retries within a provider call are not.

| Pipeline passes | Passages | Share of the 200 sampled passages |
|---:|---:|---:|
| 1 | 65 | 32.5% |
| 2 | 81 | 40.5% |
| 3 | 35 | 17.5% |
| 4 | 13 | 6.5% |
| 5 | 3 | 1.5% |
| 6 | 0 | 0.0% |
| 7 | 2 | 1.0% |
| No usable annotation: token limit | 1 | 0.5% |

**199/200 reached a clean result**, averaging **2.08 passes**, with median two and
maximum seven. Of the 134 successfully processed passages that initially had
edits, **53 (39.6%) still had edits after the first correction round**. There were
no repeated text states, and none reached the 30-pass guard.

Later rounds can find different kinds of edits. For example, passage 8 changed
`David’s → David` in pass one, `second generation → second-generation` and
`business critical → business-critical` in pass two, and
`re-negotiations → renegotiations` in pass three, before returning empty in pass
four. These are model proposals, not adjudicated corrections. Returning empty
does not establish that the text is correct or that earlier edits improved it.

## Independent runs on the original text

Each original passage was independently annotated twice. The first result also
served as the first recursive pass. Corrections from one independent run were
never applied before the other.

For location comparison, character diffs within each accepted edit identify the
source words actually changed, excluding unchanged context in a broader span.
Whitespace-delimited source words are the location anchors; a whitespace-only
change between words touches both neighbors. An edit counts as a different
location only when **none** of the other run's edits touches any of its changed
source words. Different replacements, different span boundaries, and splitting
one overlapping edit into several therefore do not by themselves count. This is
a conservative overlap comparison, not exact edit equality.

**86/199 usable pairs (43.2%) found at least one completely different edit
location.** Among the 145 pairs where either run proposed edits, this was
**59.3%**.

| Pair outcome | Passages |
|---|---:|
| Both return no edits | 54 |
| Both propose edits, with matching locations under the overlap rule | 59 |
| Both propose edits, with at least one different location | 64 |
| Exactly one returns no edits | 22 |
| Both fail at the completion-token limit | 1 |

Run one proposed 350 edits and run two proposed 353. Of these, **75/350 (21.4%)**
and **79/353 (22.4%)**, respectively, had no matching location in the other run.
These count proposals in each direction, not distinct underlying errors.
For example, passage 61 yielded `aint → ain't` only in the first run and
`your → you're` only in the second.

## Diagnostics and retained artifacts

The standard experiment used **614 pipeline passes and 684 API completions**,
including retries. Sixty passes retained validation diagnostics; two initial
results retained partial corrections after exhausting all three attempts.
Excluding those two passages changes the location-disagreement rate only to
**84/197 (42.6%)**. A 25,809-character OCR newspaper page, passage 62, used its
entire completion budget on reasoning in both independent runs and returned no
annotations. It is counted as failed, never clean.

The standard experiment took **236 seconds**, with **$0.06597** in reported API
charges, 1,022,118 input tokens and 128,276 output tokens. All 684 completions
reported usage and cost. These costs exclude the subsequent OCR follow-up.

**OCR follow-up:** raising only this passage's completion limit to 32,768 allowed
its first annotation to finish, returning 137 accepted edits after a validation
retry. That took more than five minutes; the subsequent independent call was
interrupted to keep this a quick experiment. No convergence or repeatability
result was obtained for this passage. Its two completed follow-up responses cost
another $0.00363, bringing all recorded completion charges to **$0.06960**, plus
any unreported charge for the interrupted request. The follow-up configuration,
partial responses, completed first pass, interruption metadata and runner are
retained in the `long-passage-recovery/` subdirectory.

The runner is [tools/annotation_experiment.py](../tools/annotation_experiment.py).
Frozen inputs, configuration, raw completions, resolved passes, per-passage
results and summaries are retained under
`data/recursive-annotation-experiment-20261114/` (ignored by Git). The failed URL
probe is retained separately with suffix `-endpoint-probe`.

The derived comparisons were recomputed using the final location-normalization
code. Every recursive correction chain was checked against its recorded next
input, including exact span matching and nonoverlap. REPL checks covered broad
spans, different replacements, adjacent-word suffix changes, word joins/splits,
disjoint locations and clean runs. Ruff formatting/linting and ty checks passed.
