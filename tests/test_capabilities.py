"""Tests for skdr_eval.capabilities."""

import importlib
import re
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import skdr_eval
from skdr_eval import capabilities as cap_module

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_documented_extras_are_declared_in_pyproject():
    """#107: every 'pip install skdr-eval[<extra>]' in the README is a real extra.

    Guards against doc/packaging drift such as the removed ``[choice]`` extra,
    where the README advertised an install target that pip could not satisfy.
    """
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    pyproject = tomllib.loads(
        (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    declared = set(pyproject["project"]["optional-dependencies"])

    documented = set(re.findall(r"(?:skdr-eval|\.)\[([a-z0-9_-]+)\]", readme))
    assert documented, "expected at least one documented extra in the README"

    undeclared = documented - declared
    assert not undeclared, (
        f"README documents pip extras that are not declared in pyproject.toml: "
        f"{sorted(undeclared)} (declared: {sorted(declared)})"
    )


def test_get_capabilities_schema():
    caps = skdr_eval.get_capabilities()
    assert isinstance(caps, dict)
    for key in ("viz", "speed"):
        assert key in caps
        assert isinstance(caps[key], bool)
    # 'choice' is intentionally absent: scipy is a mandatory dep.
    assert "choice" not in caps
    assert "missing_extras" in caps
    assert isinstance(caps["missing_extras"], list)
    assert all(isinstance(x, str) for x in caps["missing_extras"])
    assert caps["missing_extras"] == sorted(caps["missing_extras"])


def test_viz_capability_matches_matplotlib_presence():
    caps = skdr_eval.get_capabilities()
    has_matplotlib = importlib.util.find_spec("matplotlib") is not None
    assert caps["viz"] is has_matplotlib


def test_missing_extras_listed_when_module_absent(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name):
        if name == "matplotlib":
            return None
        return real_find_spec(name)

    monkeypatch.setattr(cap_module.importlib.util, "find_spec", fake_find_spec)
    caps = cap_module.get_capabilities()
    assert caps["viz"] is False
    assert "viz" in caps["missing_extras"]


def test_no_missing_extras_when_all_present(monkeypatch):
    def all_present(_name):
        class _Spec:
            pass

        return _Spec()

    monkeypatch.setattr(cap_module.importlib.util, "find_spec", all_present)
    caps = cap_module.get_capabilities()
    assert caps["missing_extras"] == []
    assert caps["viz"] is True
    assert caps["speed"] is True


def test_find_spec_value_error_treated_as_missing(monkeypatch):
    def raises(_name):
        raise ValueError("simulated namespace package")

    monkeypatch.setattr(cap_module.importlib.util, "find_spec", raises)
    caps = cap_module.get_capabilities()
    assert caps["viz"] is False
    assert caps["missing_extras"] == sorted(["speed", "viz"])


class TestCapabilityMatrix:
    """#215: the full optional-dependency capability matrix."""

    def test_matrix_covers_every_declared_extra(self):
        pyproject = tomllib.loads(
            (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        declared = set(pyproject["project"]["optional-dependencies"])
        # The matrix intentionally omits dev-only / docs / notebooks extras,
        # but must cover every *runtime feature* extra.
        feature_extras = {
            "viz",
            "speed",
            "cli",
            "boosting",
            "mlflow",
            "wandb",
            "aim",
        }
        assert feature_extras <= declared, (
            "matrix references extras not declared in pyproject.toml"
        )
        matrix = skdr_eval.get_capability_matrix()
        assert {c.extra for c in matrix} == feature_extras

    def test_matrix_entries_are_well_formed(self):
        for cap in skdr_eval.get_capability_matrix():
            assert isinstance(cap.installed, bool)
            assert cap.feature
            assert cap.install_hint == f"pip install 'skdr-eval[{cap.extra}]'"
            assert cap.modules
            assert cap.to_dict()["extra"] == cap.extra

    def test_installed_reflects_module_presence(self, monkeypatch):
        # Simulate an environment where only matplotlib (viz) is importable.
        def only_matplotlib(name):
            class _Spec:
                pass

            return _Spec() if name == "matplotlib" else None

        monkeypatch.setattr(cap_module.importlib.util, "find_spec", only_matplotlib)
        by_extra = {c.extra: c for c in cap_module.get_capability_matrix()}
        assert by_extra["viz"].installed is True
        # boosting needs all three of xgboost/lightgbm/catboost → absent.
        assert by_extra["boosting"].installed is False
        # speed needs pyarrow AND polars → absent.
        assert by_extra["speed"].installed is False
