# skdr-eval v0.9.0 Release Notes

**M2 User-Facing Tooling Release — EvaluationCard Schema, `doctor()`, Tracker, CLI, Security Hardening**

This release adds the full M2 user-facing tooling layer: a machine-readable `EvaluationCard` Pydantic v2 schema, a `doctor()` preflight diagnostic surface, a `Tracker` protocol with `NullTracker` and `FileTracker` implementations, and a `skdr-eval` CLI with stable CI-gate exit codes. It also delivers a repositioning sweep (README, notebooks, use-case gallery, citability), and hardens the CLI against path-traversal vulnerabilities.

There are **no breaking changes** in this release. All additions are additive. The tracker `tracker=` kwarg defaults to `None` (no tracking), so existing callers are unaffected.

---

## Added

### `EvaluationCard` Pydantic v2 schema (#88)

New `skdr_eval.EvaluationCard` — the machine-readable sibling of the HTML evaluation card. Bundles headline result, trust signals, diagnostics, sensitivity, and provenance for a single `(model, estimator)` row into a typed payload that is JSON/YAML round-trippable.

```python
from skdr_eval import EvaluationCard

card = artifact.card_schema("my_model", estimator="SNDR")
card.to_yaml("card.yaml")
card.to_json("card.json")
card2 = EvaluationCard.from_yaml("card.yaml")
schema = EvaluationCard.json_schema()
```

Block types exported: `HeadlineBlock`, `TrustBlock`, `DiagnosticsBlock`, `SensitivityBlock`, `ProvenanceBlock`, `CoverageSimBlock`. The constant `CARD_SCHEMA_VERSION = "1.0.0"` is exported from the top-level package. `ConfigDict(extra="allow")` keeps the schema forward-compatible: unknown fields are preserved on round-trip.

### `skdr_eval.doctor()` (#91)

New preflight diagnostic surface that composes `validate_logs` / `validate_pairwise_inputs` and `get_capabilities()` into a single non-raising `DoctorReport`.

```python
from skdr_eval import doctor

report = doctor(logs_df)
report.print(color=True)          # colored text to stdout
print(report.ok)                   # True if no failures
print(report.to_markdown())        # markdown table
```

Each row is a `Check(name, status, message, fix_hint, category)`. The doctor never mutates its input DataFrame and never raises — even on a non-DataFrame input the report surfaces the failure as a `Check` with `status="fail"`.

### `Tracker` protocol + `NullTracker` + `FileTracker` (#93)

New `skdr_eval.trackers` package. Both `evaluate_sklearn_models` and `evaluate_pairwise_models` accept a new `tracker=` kwarg.

```python
from skdr_eval import evaluate_sklearn_models
from skdr_eval.trackers import FileTracker

with FileTracker("./run_output/") as tracker:
    artifact = evaluate_sklearn_models(logs, models, tracker=tracker)
# run_output/ now contains:
#   metrics.jsonl, tags.json, artifacts/, cards/<model>_<estimator>.card.yaml
```

Auto-logged per `(model, estimator)` row: `V_hat`, `ci_lower`, `ci_upper`, `ESS`, `match_rate`, `pareto_k`. Stubs `MLflowTracker`, `WandbTracker`, `AimTracker` raise `NotImplementedError` on construction — full adapters tracked by umbrella issue #73.

### `skdr-eval` CLI (#89)

New Typer-based CLI installed via the `[cli]` extra (`pip install 'skdr-eval[cli]'`).

```bash
# Evaluate models and write cards
skdr-eval evaluate logs.parquet --model my_model=model.pkl --out ./results/

# Pairwise evaluation
skdr-eval pairwise logs.parquet op_daily.parquet --model my_model=model.pkl --out ./results/

# Re-render a card from a saved artifact
skdr-eval card artifact.json --model my_model --estimator SNDR --out card.yaml --format yaml

# Validate schema
skdr-eval validate-schema logs.parquet --kind standard

# Run preflight doctor
skdr-eval doctor logs.parquet --json

# Print version
skdr-eval version
```

Input format auto-detection covers `.parquet` / `.csv` / `.tsv` / `.feather`. Stable CI-gate exit codes: `0` ok, `1` data/schema error, `2` env error, `3` `do_not_deploy` on at least one row.

### New optional extras

- `[cli]`: `typer>=0.12`, `joblib>=1.3`, `pyarrow>=14`
- `[notebooks]`: `jupyter`, `matplotlib`, `nbformat`, `nbclient`
- `[mlflow]`, `[wandb]`, `[aim]`: placeholder extras for tracker adapters (umbrella issue #73)

### Repositioning, onboarding, and citability sweep (#67, #69, #77, #78, #90, #98)

- **README** rewritten for general-purpose offline policy evaluation with a plain-language "When should I use this?" section, Open-in-Colab badge table for five quickstart notebooks, and a use-case gallery.
- **`examples/notebooks/`** (new) — five nbmake-tested Colab notebooks: `01_quickstart`, `02_pairwise_quickstart`, `03_ecommerce_ranking`, `04_ad_targeting`, `05_healthcare_cate`.
- **`examples/use_cases/`** (new) — four runnable domain walk-throughs: e-commerce ranking, ad targeting, healthcare CATE, call routing.
- **CI** — new `notebooks-smoke` and `use-cases-smoke` jobs gate on the `test` job so notebook/example rot is caught at PR time.
- **`CITATION.cff`** and **`.zenodo.json`** (new) — citability and Zenodo metadata for the next tagged release.

---

## Fixed

### CLI path-traversal and filename-sanitization (#101)

Output paths in the `evaluate`, `pairwise`, and `card` subcommands are now validated to stay inside the declared `--out` directory. Model filenames are sanitized before use in filesystem operations. `doctor` column handling is corrected. Notebook `pip install` calls are guarded behind a Colab-detection check to avoid unintended side-effects in non-Colab environments.

---

## Tests

- New `tests/test_card_schema.py` (22 tests): `card_schema()`, YAML/JSON round-trip, `json_schema()`, forward-compatibility, schema-version stability guard.
- New `tests/test_doctor.py` (18 tests): per-check pass/warn/fail paths, never-raise contract, idempotence, `to_dict`/`to_markdown`/`to_text`.
- New `tests/test_trackers.py` (19 tests): `NullTracker` no-op semantics, `FileTracker` writes, evaluator wiring, pairwise `tracker=` kwarg, stub contracts.
- New `tests/test_cli.py` (15 tests): `--help`, `version`, `doctor`, `validate-schema`, `evaluate`, `card` (YAML and JSON), `EXIT_DO_NOT_DEPLOY == 3` stability guard.
- Additional coverage for `_build_card_from_row` fallback paths, `OSError` branch in `reporting.py`, CLI path guards, `coerce_float`, and the path-resolver helper (PRs #101, #102).
