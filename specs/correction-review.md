# Reviewing spelling proposals

Run commands from the repository root, inside its Nix shell. This uses local
Label Studio **Community Edition 1.23.2**, Python 3.12, SQLite, and built-in controls.
The service has its own locked uv environment: its OpenAI dependency is version 1,
whereas this research project uses version 3. No model backend is configured.

## Start

```bash
uv sync
uv sync --project tools/label-studio --locked
bash tools/label-studio/start.sh
```

Open <http://127.0.0.1:8080>, create a local account, and obtain a personal access
token from Account & Settings. Export it in the terminal used for review commands:

```bash
printf 'Label Studio API token: '
IFS= read -rs LABEL_STUDIO_API_TOKEN
printf '\n'
export LABEL_STUDIO_API_TOKEN
# Optional: export LABEL_STUDIO_URL=http://127.0.0.1:8080
```

Both personal access (JWT refresh) tokens and enabled legacy tokens work.
`start.sh` binds only localhost, disables usage analytics, and stores the database,
media, and generated secret in ignored `data/label-studio`. Keep this directory
between runs; back it up with the service stopped. Stop with Ctrl-C. Override the
port using `LABEL_STUDIO_PORT`; set the matching `LABEL_STUDIO_URL` for commands.

The implementation smoke run initialized this checkout with the account
`review@localhost.test`. Its randomly generated password and legacy token are in
ignored, mode-0600 `data/label-studio-smoke-credentials.json`. Use that password to
log in and obtain a personal token, or read its `token` field locally for API use.
Do not commit this file. The smoke projects and exports are retained for inspection.

## Configure and import

```bash
uv run review-corrections setup --project spelling-baseline
uv run review-corrections import --project spelling-baseline \
  --source-id fineweb-baseline data/fineweb-annotated
```

Input is an existing prepared JSONL shard or a directory of `part-*.jsonl` shards.
No generation runs. Null/empty annotation lists create no review tasks. Source
records, including failures and early decisions, are retained with every proposal.
The XML template lives in `src/speeling_good/review/interface.xml`; enum choices
are filled from the repository's `CorrectionType` at setup time.

Choose a stable, unique `--source-id` for each generation run. Passage IDs hash
that namespace, shard filename, and one-based row number. Proposal IDs additionally
hash the zero-based annotation index. Do not rename shards or reorder rows within
a source namespace. Source hashes cover the complete source record. A changed
prediction or passage at an existing proposal ID causes import to fail; use a new
source namespace for regenerated proposals.

Setup reuses a matching project and applies compatible layout changes without changing saved reviews. Identical imports skip existing tasks, including
reviewed tasks. Incompatible control schemas, changed existing task data, and duplicate IDs
fail rather than overwrite. If a request is interrupted, rerun the same import:
the command reads server state first. Commands serialize through a local lock;
use one checkout for imports and avoid simultaneous manual/API task imports.
Import validates all input before sending batches, but is not a server transaction.
Never delete tasks to resolve a conflict without first exporting reviews.

## Review

Open the project and select **Label All Tasks**. Each task reviews one proposal;
passages with several proposals appear in several tasks sharing a passage ID.
The upper passage highlights the current proposal. The span judgment follows it;
original text and replacement precede inclusion and replacement judgments; the
predicted correction type precedes the type judgment.
Answer replacement correctness only when inclusion is `in_scope`. Its question
and correction fields are hidden for `outside_scope`, `not_an_error`, and `unsure`.
Complete the span and type judgments for every inclusion choice.

For an incorrect type, choose a corrected enum value. For an incorrect span,
activate `corrected_span` and select exactly one region in the unchanged
passage. For an incorrect replacement, enter and submit text in its field, or
select `delete` for an empty replacement. Do not combine text and deletion.
Notes are optional. Submit the task to save the review.

The read-only highlighted rendering uses HTML escaping and preserves whitespace.
The plain passage only appears when the span is marked incorrect, because built-in
span selection needs a Text object.
Built-in controls cannot enforce every conditional cross-field rule: export
requires one corrected span and replacement text or deletion when applicable.
A failed export identifies the invalid task or annotation; repair it in Label Studio and
retry. Adding missed proposals and zero-width insertion spans are outside this
initial workflow. The baseline measures proposal review quality, not recall of
missed errors.

## Export

```bash
uv run review-corrections export --project spelling-baseline \
  --output data/reviews/baseline.jsonl
```

UTF-8 JSONL, one record per proposal, with `schema_version: 1`:

| Field | Meaning |
| --- | --- |
| `passage_id`, `proposal_id` | Stable source identities, independent of Label Studio IDs |
| `source` | Source ID, shard filename, one-based row, zero-based annotation index |
| `source_hash` | SHA-256 of sorted-key, unescaped-Unicode JSON source record |
| `source_record` | Complete original parsed source record |
| `text`, `proposal` | Exact passage and unchanged original prediction |
| `reviews` | All saved annotations; empty for unreviewed proposals |

Each review stores `annotation_id`, `reviewer`, timestamps, `status`, `judgments`,
`corrected`, `notes`, and original Label Studio `raw_results`. The four judgments
are `inclusion`, `type`, `span`, and `replacement`; values follow the interface.
For completed reviews whose inclusion is not `in_scope`, `replacement` is `null`
(not applicable), and no corrected replacement is exported. Previously saved or
hidden replacement answers remain available in `raw_results`.
`corrected` only contains adjudicated changes: `correction_type`, `span`
(`span_start`, `span_end`, `original`), and/or `replacement`. Correct and unsure
judgments do not fabricate corrected values. An empty replacement means deletion.
Canceled annotations retain their raw results and partial judgments. Unsubmitted
browser drafts are not completed reviews and are not exported.

All baseline offsets are zero-based **Unicode code-point indices, end exclusive**,
matching Python slicing. Label Studio Text offsets count UTF-16 code units; export
converts them and verifies the selected substring against the original passage.
Text and replacements are not stripped or normalized. Rejected, uncertain, and
unreviewed proposals remain present. Multiple reviews remain separate; no majority
vote or automatic accepted-only filtering is applied. Export uses atomic file
replacement, so an invalid review leaves an existing output file intact. Avoid
editing tasks while exporting if you need a consistent snapshot.

## Validation

```bash
uv run python tools/label-studio/smoke.py
uv run pytest -q tests/test_review_baseline.py
```

The live smoke check creates a uniquely named project, imports a synthetic passage
with multiple proposals, saves reviews through the API, and exports them under
`data/reviews`. It checks outside-scope and unsure judgments, incorrect type/span/
replacement, exact whitespace and non-ASCII preservation, unchanged predictions,
idempotent import, and conflicting-source rejection. It leaves its project for
manual inspection. Unit tests cover pure identity and review conversion logic;
service checks are deliberately separate from pytest.

The implementation validation passed against the running service: all 62 existing
local proposals imported, a second import added none, and all 62 exported as
unreviewed. Synthetic reviews passed the Unicode/whitespace and corrected-value
checks above. The full pytest suite passed (243 tests), as did Ruff and ty checks.
Browser inspection could not run because Chromium's system libraries are missing
in this Nix environment; rendered controls and mouse-based span selection remain
unverified. XML acceptance and saved-review behavior were checked through the API.
