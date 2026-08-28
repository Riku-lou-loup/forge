# Behavioral tests

The first six cases cover recording integrity, preserved sampling gaps,
separation of labels from sensor fields, duplicate/reversed timestamps, and
invalid labels or non-finite measurements. Small synthetic test fixtures are
used only for software correctness, never as research performance evidence.

Run `.\.venv\Scripts\python.exe -m pytest -q` from the project root.

Add tests as features are implemented, particularly for invalid schemas,
train/test leakage, unknown documents, unsupported citations, exhausted agent
budgets, and duplicate report writes.

Foundation checks are available through `scripts/check.ps1`; these verify
dependency compatibility, imports, configuration, formatting, and page rendering.
