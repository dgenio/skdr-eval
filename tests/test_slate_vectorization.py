"""Parity + performance tests for the vectorized slate estimators (#137).

The vectorized estimators must produce bit-for-bit (to float64 tolerance) the
same output as the original pure-Python nested-loop implementation. We pin that
contract by re-implementing the reference loops inline here and asserting parity
across all three click models and several target policies.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from skdr_eval.slate import (
    make_slate_synth,
    pseudo_inverse_ips,
    reward_interaction_ips,
    slate_cascade_dr,
)


def _ref_ess(w: np.ndarray) -> float:
    if (w**2).sum() == 0:
        return 0.0
    return float(w.sum() ** 2 / (w**2).sum())


def _ref_se(contribs: np.ndarray) -> float:
    return (
        float(contribs.std(ddof=1) / np.sqrt(contribs.size))
        if contribs.size > 1
        else 0.0
    )


def _ref_rips(logs, target, log_policy=None):
    slates = [list(map(int, s)) for s in logs["slate"]]
    clicks = [list(map(int, c)) for c in logs["clicks"]]
    n_items = int(max(max(s) for s in slates)) + 1
    if log_policy is None:

        def log_policy(_r, _i):
            return 1.0 / n_items

    contribs = np.empty(len(slates))
    weights = np.empty(len(slates))
    for i, (slate, click_row) in enumerate(zip(slates, clicks, strict=False)):
        per_rank, sw = 0.0, 1.0
        for k, item in enumerate(slate):
            p_t, p_l = float(target(k, item)), float(log_policy(k, item))
            if p_l <= 0:
                sw = 0.0
                break
            ratio = p_t / p_l
            sw *= ratio
            per_rank += ratio * float(click_row[k])
        contribs[i] = per_rank if sw > 0 else 0.0
        weights[i] = sw
    return float(contribs.mean()), _ref_se(contribs), _ref_ess(weights)


def _ref_piips(logs, target, n_items):
    slates = [list(map(int, s)) for s in logs["slate"]]
    clicks = [list(map(int, c)) for c in logs["clicks"]]
    slate_size = len(slates[0])
    pi_t = np.zeros((slate_size, n_items))
    for k in range(slate_size):
        for j in range(n_items):
            pi_t[k, j] = float(target(k, j))
    pinv = np.linalg.pinv(pi_t)
    contribs = np.empty(len(slates))
    weights = np.empty(len(slates))
    for i, (slate, click_row) in enumerate(zip(slates, clicks, strict=False)):
        per_rank, norm = 0.0, 0.0
        for k, item in enumerate(slate):
            w = float(pinv[item, k])
            per_rank += w * float(click_row[k])
            norm += abs(w)
        contribs[i] = per_rank
        weights[i] = norm
    return float(contribs.mean()), _ref_se(contribs), _ref_ess(weights)


def _ref_cascade(logs, target, q_hat, log_policy=None):
    slates = [list(map(int, s)) for s in logs["slate"]]
    clicks = [list(map(int, c)) for c in logs["clicks"]]
    n_items = int(max(max(s) for s in slates)) + 1
    if log_policy is None:

        def log_policy(_r, _i):
            return 1.0 / n_items

    contribs = np.empty(len(slates))
    weights = np.empty(len(slates))
    for i, (slate, click_row) in enumerate(zip(slates, clicks, strict=False)):
        per_rank, max_w = 0.0, 0.0
        for k, item in enumerate(slate):
            q_pi = sum(
                float(target(k, j)) * float(q_hat[i, k, j]) for j in range(n_items)
            )
            p_t, p_l = float(target(k, item)), float(log_policy(k, item))
            w = p_t / p_l if p_l > 0 else 0.0
            per_rank += q_pi + w * (float(click_row[k]) - float(q_hat[i, k, item]))
            max_w = max(max_w, w)
        contribs[i] = per_rank
        weights[i] = max_w
    return float(contribs.mean()), _ref_se(contribs), _ref_ess(weights)


def _sharp_target(n_items: int, seed: int):
    pref = np.random.default_rng(seed).random(n_items)

    def target(rank: int, item: int) -> float:
        w = np.exp(1.5 * pref + 0.3 * rank)
        return float(w[item] / w.sum())

    return target


@pytest.mark.parametrize("click_model", ["cascade", "position_bias", "linear"])
def test_rips_parity(click_model: str) -> None:
    logs, _, _ = make_slate_synth(
        n_impressions=120, n_items=7, slate_size=3, click_model=click_model, seed=4
    )
    target = _sharp_target(7, 1)
    res = reward_interaction_ips(logs, target)
    ref_v, ref_se, ref_ess = _ref_rips(logs, target)
    assert res.V_hat == pytest.approx(ref_v, rel=0, abs=1e-12)
    assert pytest.approx(ref_se, rel=0, abs=1e-12) == res.SE
    assert pytest.approx(ref_ess, rel=0, abs=1e-9) == res.ESS


@pytest.mark.parametrize("click_model", ["cascade", "position_bias", "linear"])
def test_piips_parity(click_model: str) -> None:
    logs, _, _ = make_slate_synth(
        n_impressions=120, n_items=7, slate_size=3, click_model=click_model, seed=5
    )
    target = _sharp_target(7, 2)
    res = pseudo_inverse_ips(logs, target, n_items=7)
    ref_v, ref_se, ref_ess = _ref_piips(logs, target, 7)
    assert res.V_hat == pytest.approx(ref_v, rel=0, abs=1e-12)
    assert pytest.approx(ref_se, rel=0, abs=1e-12) == res.SE
    assert pytest.approx(ref_ess, rel=0, abs=1e-9) == res.ESS


@pytest.mark.parametrize("click_model", ["cascade", "position_bias", "linear"])
def test_cascade_parity(click_model: str) -> None:
    logs, _, _ = make_slate_synth(
        n_impressions=120, n_items=7, slate_size=3, click_model=click_model, seed=6
    )
    target = _sharp_target(7, 3)
    rng = np.random.default_rng(9)
    q_hat = rng.random((len(logs), 3, 7))
    res = slate_cascade_dr(logs, target, q_hat)
    ref_v, ref_se, ref_ess = _ref_cascade(logs, target, q_hat)
    assert res.V_hat == pytest.approx(ref_v, rel=0, abs=1e-12)
    assert pytest.approx(ref_se, rel=0, abs=1e-12) == res.SE
    assert pytest.approx(ref_ess, rel=0, abs=1e-9) == res.ESS


def test_empty_logs_returns_zero() -> None:
    empty = pd.DataFrame({"slate": [], "clicks": [], "reward": [], "logging_prob": []})
    target = _sharp_target(5, 0)
    assert reward_interaction_ips(empty, target).n == 0
    assert pseudo_inverse_ips(empty, target, n_items=5).n == 0


def test_cascade_vectorized_is_not_slower_on_large_catalogue() -> None:
    """The vectorized path must scale to large catalogues (#137).

    A pure-Python triple loop over (impression, rank, item) would be painfully
    slow here; the vectorized estimator should finish comfortably under a
    generous wall-clock budget. This guards against an accidental regression
    back to per-item Python loops.
    """
    logs, _, _ = make_slate_synth(n_impressions=400, n_items=200, slate_size=5, seed=7)
    target = _sharp_target(200, 4)
    q_hat = np.zeros((len(logs), 5, 200), dtype=np.float64)
    start = time.perf_counter()
    slate_cascade_dr(logs, target, q_hat)
    assert time.perf_counter() - start < 5.0
