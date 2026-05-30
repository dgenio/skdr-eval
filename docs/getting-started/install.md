# Install

`skdr-eval` requires Python ≥ 3.11.

```bash
pip install skdr-eval
```

## Optional extras

The core install is intentionally lean (numpy, pandas, scikit-learn, scipy,
jinja2, pydantic, pyyaml). Heavier or specialised dependencies are opt-in:

| Extra | Installs | Use it for |
|-------|----------|------------|
| `[viz]` | matplotlib | Rendering charts in the evaluation card. |
| `[speed]` | pyarrow, polars | Columnar I/O / faster pairwise grouping. |
| `[cli]` | typer, joblib, pyarrow | The `skdr-eval` command-line tool. |
| `[mlflow]` / `[wandb]` / `[aim]` | the respective tracker | Logging artifacts to an experiment tracker. |
| `[notebooks]` | jupyter, matplotlib | Running the example notebooks. |
| `[docs]` | mkdocs-material, mkdocstrings | Building this documentation site. |
| `[dev]` | pytest, mypy, ruff, … | Contributing. |

```bash
pip install "skdr-eval[viz,cli]"
```

## Verify the install

```python
import skdr_eval

# Which optional extras are active?
print(skdr_eval.get_capabilities())
# -> {'viz': True, 'speed': False, 'cli': False, 'missing_extras': [...]}
```

## From source

```bash
git clone https://github.com/dgenio/skdr-eval
cd skdr-eval
pip install -e ".[dev]"
make check        # lint + typecheck + test + smoke
```

The [`Makefile`](https://github.com/dgenio/skdr-eval/blob/main/Makefile) is the
authoritative source of development workflows.
