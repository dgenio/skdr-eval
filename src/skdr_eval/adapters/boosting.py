"""Adapters for non-sklearn outcome / surrogate models (#71).

``evaluate_sklearn_models`` and ``evaluate_pairwise_models`` accept any object
satisfying the sklearn ``fit`` / ``predict`` (and optionally ``predict_proba``)
protocol — the name is historical, not a restriction. In practice production
outcome models are gradient-boosted trees (XGBoost / LightGBM / CatBoost) or a
bespoke callable. These adapters make that first-class:

* :class:`XGBRegressorAdapter`, :class:`LGBMRegressorAdapter`,
  :class:`CatBoostRegressorAdapter` — thin wrappers that construct the native
  regressor and forward its preferred fit kwargs (``early_stopping_rounds``,
  ``categorical_feature`` / ``cat_features``, GPU flags). Each exposes the
  ``fit`` / ``predict`` surface the evaluators expect, so the wrapped model
  drops straight into a ``models`` dict or the ``outcome_estimator`` slot.
* :class:`CallableModelAdapter` — wraps a plain ``predict`` function (plus
  optional ``predict_proba`` / ``fit``) for the "I already have a function"
  case.

The GBDT libraries are optional: install via ``pip install 'skdr-eval[boosting]'``.
Importing this module never imports them — each adapter imports its backend
lazily in ``__init__`` and raises :class:`OptionalDependencyError` when the
package is missing.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import numpy as np

from ..exceptions import ModelValidationError, OptionalDependencyError

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

__all__ = [
    "CallableModelAdapter",
    "CatBoostRegressorAdapter",
    "LGBMRegressorAdapter",
    "XGBRegressorAdapter",
]


class _BoostingRegressorAdapter:
    """Shared base for GBDT regressor adapters.

    Subclasses set :attr:`_package` (importable module), :attr:`_extra` (the
    matching ``skdr-eval`` extra) and implement :meth:`_make_estimator`.
    Construction imports the backend lazily and builds the native estimator;
    :meth:`fit` forwards stored fit-time kwargs and :meth:`predict` delegates.
    """

    _package: str = ""
    _extra: str = "boosting"

    def __init__(self, *, fit_kwargs: dict[str, Any] | None = None, **params: Any):
        self._backend = self._import_backend()
        self._fit_kwargs: dict[str, Any] = dict(fit_kwargs or {})
        self.estimator = self._make_estimator(params)

    @classmethod
    def _import_backend(cls) -> Any:
        try:
            return importlib.import_module(cls._package)
        except ImportError as exc:
            raise OptionalDependencyError(
                f"{cls.__name__}", cls._package, extra=cls._extra
            ) from exc

    def _make_estimator(self, params: dict[str, Any]) -> Any:  # pragma: no cover
        raise NotImplementedError

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        **kwargs: Any,
    ) -> _BoostingRegressorAdapter:
        """Fit the wrapped estimator, merging construction-time fit kwargs.

        Per-call ``kwargs`` take precedence over the ``fit_kwargs`` supplied at
        construction. Returns ``self`` to match the sklearn fit contract.
        """
        merged = {**self._fit_kwargs, **kwargs}
        self.estimator.fit(X, y, **merged)
        return self

    def predict(self, X: npt.ArrayLike) -> np.ndarray:
        """Delegate to the wrapped estimator's ``predict``."""
        return np.asarray(self.estimator.predict(X))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.estimator!r})"


class XGBRegressorAdapter(_BoostingRegressorAdapter):
    """Adapter around :class:`xgboost.XGBRegressor`.

    Parameters
    ----------
    fit_kwargs : dict, optional
        Forwarded to ``XGBRegressor.fit`` on every call — e.g.
        ``{"verbose": False}``. (XGBoost reads ``early_stopping_rounds`` as a
        *constructor* argument in modern versions, so pass it via ``**params``.)
    **params
        Forwarded to the ``XGBRegressor`` constructor (e.g. ``n_estimators``,
        ``max_depth``, ``device="cuda"``, ``early_stopping_rounds``).
    """

    _package = "xgboost"

    def _make_estimator(self, params: dict[str, Any]) -> Any:
        return self._backend.XGBRegressor(**params)


class LGBMRegressorAdapter(_BoostingRegressorAdapter):
    """Adapter around :class:`lightgbm.LGBMRegressor`.

    Parameters
    ----------
    fit_kwargs : dict, optional
        Forwarded to ``LGBMRegressor.fit`` — e.g.
        ``{"categorical_feature": ["region"]}`` or early-stopping callbacks.
    **params
        Forwarded to the ``LGBMRegressor`` constructor (e.g. ``n_estimators``,
        ``num_leaves``, ``device="gpu"``).
    """

    _package = "lightgbm"

    def _make_estimator(self, params: dict[str, Any]) -> Any:
        return self._backend.LGBMRegressor(**params)


class CatBoostRegressorAdapter(_BoostingRegressorAdapter):
    """Adapter around :class:`catboost.CatBoostRegressor`.

    Parameters
    ----------
    fit_kwargs : dict, optional
        Forwarded to ``CatBoostRegressor.fit`` — e.g.
        ``{"cat_features": [0, 3], "early_stopping_rounds": 50}``.
    **params
        Forwarded to the ``CatBoostRegressor`` constructor (e.g.
        ``iterations``, ``depth``, ``task_type="GPU"``). ``verbose=False`` is
        applied by default to keep evaluation output quiet; override it via
        ``**params``.
    """

    _package = "catboost"

    def _make_estimator(self, params: dict[str, Any]) -> Any:
        params.setdefault("verbose", False)
        return self._backend.CatBoostRegressor(**params)


class CallableModelAdapter:
    """Wrap plain prediction function(s) as an sklearn-compatible estimator.

    For the "I already have a model, just call it" case. The adapter exposes
    ``fit`` / ``predict`` (and ``predict_proba`` when ``predict_proba_fn`` is
    given) so a bare function plugs into a ``models`` dict or the
    ``outcome_estimator`` slot without a wrapper class.

    Parameters
    ----------
    predict_fn : callable
        ``predict_fn(X) -> array`` of shape ``(n_samples,)`` (regression /
        decision values) or ``(n_samples,)`` class labels.
    predict_proba_fn : callable, optional
        ``predict_proba_fn(X) -> array`` of shape ``(n_samples, n_classes)``.
        When omitted, the adapter has no ``predict_proba`` attribute, matching
        how a regressor presents itself.
    fit_fn : callable, optional
        ``fit_fn(X, y, **kwargs) -> Any``. When omitted, :meth:`fit` is a
        no-op returning ``self`` — appropriate for an already-fitted model.

    Notes
    -----
    ``predict_proba_fn`` is wired conditionally: omitting it means the instance
    genuinely lacks ``predict_proba`` (``hasattr(..., "predict_proba")`` is
    ``False``), so downstream code that branches on that attribute behaves the
    same as it would for a native regressor.
    """

    def __init__(
        self,
        predict_fn: Callable[..., npt.ArrayLike],
        predict_proba_fn: Callable[..., npt.ArrayLike] | None = None,
        fit_fn: Callable[..., Any] | None = None,
    ):
        if not callable(predict_fn):
            raise ModelValidationError("predict_fn must be callable")
        if predict_proba_fn is not None and not callable(predict_proba_fn):
            raise ModelValidationError("predict_proba_fn must be callable")
        if fit_fn is not None and not callable(fit_fn):
            raise ModelValidationError("fit_fn must be callable")
        self._predict_fn = predict_fn
        self._fit_fn = fit_fn
        if predict_proba_fn is not None:
            self._predict_proba_fn = predict_proba_fn
            # Bind ``predict_proba`` only when supplied so ``hasattr`` reflects
            # the true capability.
            self.predict_proba = self._predict_proba

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        **kwargs: Any,
    ) -> CallableModelAdapter:
        """Call ``fit_fn`` if provided; otherwise a no-op. Returns ``self``."""
        if self._fit_fn is not None:
            self._fit_fn(X, y, **kwargs)
        return self

    def predict(self, X: npt.ArrayLike) -> np.ndarray:
        """Return ``predict_fn(X)`` as a NumPy array."""
        return np.asarray(self._predict_fn(X))

    def _predict_proba(self, X: npt.ArrayLike) -> np.ndarray:
        return np.asarray(self._predict_proba_fn(X))
