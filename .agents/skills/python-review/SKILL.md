---
name: python-review
description: Review Python changes and propose or apply focused fixes for readability, function responsibilities, testable business logic, and justified safeguards. Maintain the learned preferences after human review feedback.
---

# Python Review

Review Python as a maintainer who values direct, readable code and meaningful tests.
Use the preferences below alongside correctness and repository instructions. They
are judgment criteria, not a requirement to rewrite every change into one style.

## Review and fix

Read the diff, surrounding code, callers, and relevant tests before commenting.
Distinguish new problems from pre-existing code and respect unrelated working-tree
changes. Follow the repository's tooling and test conventions.

For each actionable finding, identify the location, explain the concrete reading,
maintenance, or correctness cost, and propose the smallest useful fix. Readability
findings are valid even when behavior is correct; label them accordingly rather
than inventing a runtime failure. Avoid speculative findings and cosmetic churn.

A review request calls for findings and suggested fixes. When asked to implement
fixes, make them, preserve intended behavior and required public contracts, and
run relevant checks. Report the result and any unverified parts.

## Learned preferences

### Keep orchestration easy to read

When a function mixes requesting data, parsing, business rules, error translation,
and retries, separate cohesive responsibilities into named functions or methods.
The main function should make the overall flow apparent. Extract where a name and
contract reduce cognitive load; avoid arbitrary line limits, one-line wrappers,
or a framework of tiny helpers that makes readers chase the implementation.
Keep per-call state local and preserve concurrency and retry semantics when
refactoring those concerns.

Keep dataset loading, cleaning, and train/validation splitting in the dataset or
dataloader layer. Training entrypoints should configure and consume loaders rather
than duplicate their preparation logic. Expose meaningful data options through
that layer when callers need them.

Separate per-item validation or matching from collection-level conflict handling,
and response validation from decoding and result assembly, when those concerns
obscure the overall flow. Prefer explicit local results over helpers that mutate
captured output state.

### Retain functionality for actual callers

Check production callers before preserving an API or compatibility wrapper. In a
greenfield application, remove unused functionality and adapt or remove tests
that exist only to exercise it; tests alone do not justify retaining a feature.
Keep compatibility only when real consumers or explicit requirements need it.

### Prefer explicit code over artificial uniformity

For a small fixed set of distinct configuration checks or operations, explicit
statements can be clearer than a loop over name/value tuples or a dispatch table.
Unroll when the abstraction forces readers to reconstruct which rule applies to
which field. Keep loops for genuinely repeated operations over data or sufficiently
large uniform sets. Preserve meaningful validation and field-specific errors;
readability is the reason to change the structure, not to weaken the rules.

Prefix unused parameters with an underscore and document why they are needed.
If an unused parameter must keep its name, document that reason too.

For ordered matching, prefer an explicit advancing index over a nested
next/generator search when it makes progress and boundary handling easier to
follow. Use an indexable input when needed, while retaining iterator-based code
when traversal state or random access would add complexity without helping.

### Test pure business logic directly

A pure function that performs meaningful decisions, matching, transformations, or
validation deserves direct unit tests rather than coverage only through a service
wrapper. Expose it as a public function when it represents an independently useful
operation; make its required input types usable without private imports. Do not
make every incidental implementation helper public just to test it.

Test meaningful contracts with in-memory inputs and independently specified
expected results: normal cases, ambiguous or invalid inputs, boundaries, ordering,
and invariants relevant to the actual algorithm. Exercise the function without
HTTP clients, mocked service responses, or filesystem fixtures when it needs none.
Keep service-level tests for integration responsibilities such as retries; avoid
needlessly duplicating the full business-rule matrix there. Do not add trivial
assertions or tests tied to private structure, incidental wording, or external
services. Use separate manual or integration checks where external resources are
needed, following repository policy.

### Justify safeguards and preserve useful errors

Avoid unnecessary defensive coding. Each safeguard should prevent a concrete
failure at a clear boundary in the actual data flow.

For example, validate shared schema invariants once at the dataset or service
boundary, then trust them in per-item processing. Let built-in exceptions surface
broken upstream invariants unless translation adds useful context.

Keep input types aligned with accepted values: if None is always an error, omit
it from the type rather than accepting it only to reject it at runtime. Retain
optional types when absence is meaningful or handling it is a boundary's job.
Trust structural guarantees from established libraries, such as ordered offsets,
instead of validating hypothetical violations on each item. Keep validation for
untrusted inputs or invariants the provider does not guarantee.

When a simple precondition such as requiring a new output directory suffices,
prefer it over inspecting and reconciling individual artifacts. Add resume or
compatibility logic only when that behavior is required.

A warning about sensitive content in logs is not by itself a reason to hide
validation details from the same model that already received the input. Keep
useful field and validation information for debugging and recovery.

Choose error detail according to the recipient and actual project requirements.
Do not infer a blanket ban on Pydantic errors, or a blanket permission to log raw
inputs. Preserve real credential protection and privacy requirements. Replace
unjustified restrictions and alarmist comments with the simpler appropriate
behavior or a comment explaining the actual reason.

## Learn after each human review

When using this skill with new human review feedback, update this skill as part of
handling that feedback. If the feedback is unavailable, do not invent preferences
or claim to have learned from a review you have not read.

- Identify the underlying preference and its applicable context from the human's
  explanation and accepted fix. Distinguish human feedback from automated findings
  and the agent's own assumptions.
- Merge a concise decision criterion into the relevant section. Include the usual
  fix and when it would be inappropriate; do not encode project identifiers,
  exact snippets, one-off incidents, or every review comment as universal rules.
- Refine or replace an existing rule instead of accumulating near duplicates. If
  feedback conflicts, follow explicit current instructions for the task and clarify
  whether the durable preference changed when that distinction matters.
- Keep repository-specific conventions in repository instructions. This skill
  records reusable review judgment, not one project's dependency or tooling list.
- Validate changed skill metadata using the available skill validator and summarize
  the preference update. Keep this entrypoint short; add supporting material only
  when it has a concrete recurring use. Updating the skill does not authorize
  changes outside the user's task or bypass installation permissions.
