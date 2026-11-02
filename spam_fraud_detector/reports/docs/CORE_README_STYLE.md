# `core/README.md` Style Guide

This is the canonical reference for editing `spam_fraud_detector/core/README.md`.
If a rule here conflicts with a code comment, the code comment wins; open a PR to
align them.

## Voice

- Third person, present tense. "The comparer fits..." not "We fit..."
- No marketing adjectives ("powerful", "robust", "state-of-the-art") unless the
  next sentence names the mechanism that justifies it.
- Every claim about behavior must be falsifiable by reading the relevant `.py`
  file. If you cannot point to the line, delete the claim.

## Structure (non-negotiable)

1. `# Core Modules` — top heading.
2. One-line module table.
3. `## Overview` — 150-300 words, one mermaid diagram.
4. `## How to read this document` — four bullets.
5. One `## \`<module>\`` section per `core/*.py` (except `__init__.py`), in
   the fixed order: `exceptions`, `utils`, `data_loader`, `split`, `quality`,
   `features`, `data_pipeline`, `comparer`, `result_store`, `selector`, `stats`,
   `visualizer`. New modules are appended at the end.
6. `## Cross-module flows` — three mermaid sequence diagrams.
7. `## Provenance & invariants (reference)` — table.
8. `## Author & license`.

## Per-module section (four subheadings, in this order)

### `### What it does`

One full paragraph (>= 40 words) in plain language. Bullets may follow the
paragraph but must not replace it.

### `### How it does it`

Prose describing the algorithm or flow, followed by a bullet list of guarantees.
Every guarantee names its mechanism. Example: "atomic via staging dir +
`os.replace`", not "atomic".

### `### What it produces`

A markdown table with columns `Artifact | Path | Shape | When`. Every path
fragment must appear verbatim in the module's source (checked by
`scripts/check_core_readme.py`).

### `### How to use it`

At least one fenced ` ```python ` block that:

- imports from `spam_fraud_detector.core.<module>`;
- constructs the object with real arguments;
- calls at least one public method;
- ends with a comment showing the expected artifact path or value.

Snippets must parse with `ast.parse` and must not contain `...` placeholders.

## Migration notes

If the code does not yet match the spec, add a short `> **Migration note ...**`
blockquote **inside** the relevant section, not in a separate appendix. The note
must say what the code does today, what the target is, and whether the API
changes. Never delete a working behavior description because the target is
different — describe both.

## Length

- Overview: 150-300 words.
- Per module: 250-600 words.
- Cross-module flows: 200-400 words total.

## Forbidden

- HTML anchors, owner blocks, or hash blocks — those were removed in v3.5.
- External images; mermaid only.
- Placeholder task items without an owner and a tracking issue link.
- Placeholders like `{{value}}` in code or JSON snippets.

## When you add a module

1. Add `<module>.py` under `core/`.
2. Add a row to the module table.
3. Add a `## \`<module>\`` section at the **end** of the module list.
4. Run `make docs-check-core`.