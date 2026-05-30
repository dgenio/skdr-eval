"""Simulation proof for the #106 propensity-calibration switch (isotonic→sigmoid).

#106 switched :func:`skdr_eval.core.fit_propensity_timecal` from
``CalibratedClassifierCV(method="isotonic", cv=2)`` to
``method="sigmoid", cv=3``. Isotonic regression is a monotone *step* function:
on the small per-fold samples produced by time-aware CV it pins calibrated
probabilities to hard zeros, driving ``min_pscore ≈ 0`` and tripping
``POOR_OVERLAP`` / ``MISCAL_PROP`` even on well-overlapped data (the alarming
``support_health=high_risk`` newcomers saw on the library's own demos). Sigmoid
(Platt) calibration is smooth and stays bounded away from zero.

This is the simulation proof AGENTS.md §6 requires for the change: on a known
DGP with full overlap (true propensities bounded away from zero) the sigmoid
path recovers non-degenerate propensities that track the truth, and isotonic
collapses to hard zeros on the same fold-sized data.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from skdr_eval.core import fit_propensity_timecal


def _dgp(
    seed: int, n: int = 1500, n_actions: int = 3, dim: int = 4, temp: float = 0.8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A well-overlapped softmax-logging DGP with known propensities.

    ``temp < 1`` flattens the softmax so every action keeps appreciable
    probability — there is no genuine overlap problem to find, so any
    ``min_pscore ≈ 0`` is a calibration artefact, not a data property.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim))
    w = rng.normal(size=(dim, n_actions))
    logits = temp * (x @ w)
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    a = np.array([rng.choice(n_actions, p=p[i]) for i in range(n)], dtype=int)
    return x, a, p


def test_timecal_sigmoid_is_nondegenerate_and_tracks_truth_simulation() -> None:
    """The shipped (sigmoid) path: non-degenerate propensities that track truth."""
    for seed in range(5):
        x, a, p_true = _dgp(seed)
        ts = np.arange(len(a))
        p_hat, _ = fit_propensity_timecal(x, a, ts, n_splits=3, random_state=0)
        rows = np.arange(len(a))
        # No *hard-zero* propensities — isotonic pins probabilities to an exact
        # zero on these fold sizes (< 1e-6, see the contrast test); sigmoid
        # stays strictly positive even where it is small.
        assert p_hat.min() > 1e-6, (seed, float(p_hat.min()))
        # Observed-action propensities correlate with the ground truth.
        corr = float(np.corrcoef(p_hat[rows, a], p_true[rows, a])[0, 1])
        assert corr > 0.3, (seed, corr)


def test_isotonic_cv2_collapses_where_sigmoid_does_not() -> None:
    """Contrast pinning *why* the switch was needed (#106).

    On a fold-sized slice, isotonic(cv=2) calibration produces hard-zero class
    probabilities (``min ≈ 0``); sigmoid(cv=3) on identical data stays bounded
    away from zero. This is the degenerate-``min_pscore`` mechanism #106 fixed.
    """
    x, a, _ = _dgp(0)
    sl = slice(0, 500)  # mimics a small time-aware training fold
    base = LogisticRegression(max_iter=500)
    iso = CalibratedClassifierCV(base, method="isotonic", cv=2).fit(x[sl], a[sl])
    sig = CalibratedClassifierCV(base, method="sigmoid", cv=3).fit(x[sl], a[sl])
    iso_p = iso.predict_proba(x[sl])
    sig_p = sig.predict_proba(x[sl])
    # Isotonic pins probabilities to a hard zero; sigmoid does not.
    assert iso_p.min() < 1e-6, float(iso_p.min())
    assert sig_p.min() > 1e-3, float(sig_p.min())
