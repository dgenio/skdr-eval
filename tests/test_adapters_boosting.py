"""Tests for non-sklearn model adapters (#71)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

import skdr_eval
from skdr_eval.adapters import (
    CallableModelAdapter,
    CatBoostRegressorAdapter,
    LGBMRegressorAdapter,
    XGBRegressorAdapter,
)
from skdr_eval.exceptions import ModelValidationError, OptionalDependencyError

_HAS_XGB = importlib.util.find_spec("xgboost") is not None
_HAS_LGBM = importlib.util.find_spec("lightgbm") is not None

requires_xgb = pytest.mark.skipif(not _HAS_XGB, reason="xgboost not installed")
requires_lgbm = pytest.mark.skipif(not _HAS_LGBM, reason="lightgbm not installed")


def _evaluate_with(model: object) -> skdr_eval.EvaluationArtifact:
    logs, _, _ = skdr_eval.make_synth_logs(n=600, n_ops=3, seed=0)
    return skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models={"candidate": model},
        fit_models=True,
        n_splits=3,
        random_state=0,
        policy_train="pre_split",
    )


@requires_xgb
class TestXGBAdapter:
    def test_fit_predict_shape(self) -> None:
        rng = np.random.RandomState(0)
        X = rng.normal(size=(120, 4))
        y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.normal(size=120)
        model = XGBRegressorAdapter(n_estimators=20, max_depth=3, random_state=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (120,)
        assert np.all(np.isfinite(preds))

    def test_satisfies_models_dict_contract(self) -> None:
        art = _evaluate_with(XGBRegressorAdapter(n_estimators=20, random_state=0))
        assert "V_hat" in art.report.columns
        assert not art.report.empty


@requires_lgbm
class TestLGBMAdapter:
    def test_fit_kwargs_forwarded(self) -> None:
        rng = np.random.RandomState(1)
        X = rng.normal(size=(150, 3))
        y = X[:, 0] * 2.0 + rng.normal(size=150)
        # ``fit_kwargs`` should reach LGBMRegressor.fit without error.
        model = LGBMRegressorAdapter(
            n_estimators=20,
            random_state=0,
            verbose=-1,
            fit_kwargs={"feature_name": ["a", "b", "c"]},
        )
        model.fit(X, y)
        assert model.predict(X).shape == (150,)


def test_boosting_adapter_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing backend raises OptionalDependencyError with the extra hint."""
    import importlib  # noqa: PLC0415

    real_import_module = importlib.import_module

    def fake_import_module(name: str, *args: object, **kwargs: object) -> object:
        if name == "xgboost":
            raise ImportError("no xgboost")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    with pytest.raises(OptionalDependencyError) as exc:
        XGBRegressorAdapter()
    assert exc.value.package == "xgboost"
    assert exc.value.extra == "boosting"
    # CatBoost backend is genuinely absent in the test env → same error type.
    with pytest.raises(OptionalDependencyError):
        CatBoostRegressorAdapter()


class TestCallableModelAdapter:
    def test_predict_only_is_a_regressor(self) -> None:
        adapter = CallableModelAdapter(predict_fn=lambda X: np.zeros(len(X)))
        # No predict_proba bound when none supplied.
        assert not hasattr(adapter, "predict_proba")
        assert hasattr(adapter, "fit")
        out = adapter.predict(np.ones((5, 2)))
        assert out.shape == (5,)
        # fit is a no-op returning self when fit_fn is omitted.
        assert adapter.fit(np.ones((5, 2)), np.zeros(5)) is adapter

    def test_predict_proba_bound_when_supplied(self) -> None:
        adapter = CallableModelAdapter(
            predict_fn=lambda X: np.zeros(len(X)),
            predict_proba_fn=lambda X: np.tile([0.5, 0.5], (len(X), 1)),
        )
        assert hasattr(adapter, "predict_proba")
        proba = adapter.predict_proba(np.ones((3, 2)))
        assert proba.shape == (3, 2)

    def test_fit_fn_invoked(self) -> None:
        calls: list[tuple] = []
        adapter = CallableModelAdapter(
            predict_fn=lambda X: np.zeros(len(X)),
            fit_fn=lambda X, y, **kw: calls.append((X.shape, y.shape)),
        )
        adapter.fit(np.ones((4, 2)), np.zeros(4))
        assert calls == [((4, 2), (4,))]

    def test_non_callable_rejected(self) -> None:
        with pytest.raises(ModelValidationError, match="predict_fn must be callable"):
            CallableModelAdapter(predict_fn=123)  # type: ignore[arg-type]

    def test_callable_adapter_as_model(self) -> None:
        # Wrap an sklearn model behind bare fit / predict functions; the
        # evaluator drives fit on its internal policy-feature matrix.
        logs, _, _ = skdr_eval.make_synth_logs(n=500, n_ops=3, seed=2)
        inner = HistGradientBoostingRegressor(max_iter=20, random_state=0)
        adapter = CallableModelAdapter(predict_fn=inner.predict, fit_fn=inner.fit)
        art = skdr_eval.evaluate_sklearn_models(
            logs=logs,
            models={"wrapped": adapter},
            fit_models=True,
            n_splits=3,
            random_state=0,
            policy_train="pre_split",
        )
        assert "V_hat" in art.report.columns
