"""Edge-branch coverage for the slate-OPE module (#75).

The recovery simulation in ``test_estimator_recovery_simulation.py`` only
exercises the ``cascade`` click model and the happy path of each estimator.
This module covers the remaining branches: the ``position_bias`` / ``linear``
click models and their validation, the slate-size guard, and the estimator
edge cases (empty logs, custom logging policies, zero-weight ESS, and the
Cascade-DR ``q_hat`` shape guard).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skdr_eval.slate import (
    make_slate_synth,
    pseudo_inverse_ips,
    reward_interaction_ips,
    slate_cascade_dr,
    slate_standard_ips,
)
from skdr_eval.slate.estimators import _ess_from_weights, _matched_slate_indicator


def _empty_logs() -> pd.DataFrame:
    return pd.DataFrame({"slate": [], "clicks": [], "reward": [], "logging_prob": []})


class TestSynthClickModels:
    @pytest.mark.parametrize("model", ["cascade", "position_bias", "linear"])
    def test_each_click_model_runs(self, model: str) -> None:
        logs, attractiveness, truth = make_slate_synth(
            n_impressions=50, n_items=6, slate_size=3, click_model=model, seed=1
        )
        assert truth.click_model == model
        assert len(logs) == 50
        assert attractiveness.shape == (50, 6)
        assert np.isfinite(truth.V_oracle_target)
        assert np.isfinite(truth.V_logging)

    def test_invalid_click_model_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown click model"):
            make_slate_synth(
                n_impressions=10,
                n_items=5,
                slate_size=2,
                click_model="bogus",  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("slate_size", [0, 99])
    def test_invalid_slate_size_raises(self, slate_size: int) -> None:
        with pytest.raises(ValueError, match="slate_size must be"):
            make_slate_synth(n_impressions=10, n_items=5, slate_size=slate_size)


class TestSlateHelpers:
    def test_matched_slate_indicator(self) -> None:
        assert _matched_slate_indicator([1, 2, 3], [1, 2, 3]) == 1
        assert _matched_slate_indicator([1, 2, 3], [1, 2, 4]) == 0
        # Length mismatch short-circuits to 0.
        assert _matched_slate_indicator([1, 2], [1, 2, 3]) == 0

    def test_ess_zero_when_weights_vanish(self) -> None:
        assert _ess_from_weights(np.zeros(5)) == 0.0


class TestSlateEstimatorEdges:
    def _logs(self) -> pd.DataFrame:
        logs, _attr, _truth = make_slate_synth(
            n_impressions=40, n_items=6, slate_size=3, seed=2
        )
        return logs

    def test_standard_ips_zero_target_yields_zero_value_and_ess(self) -> None:
        logs = self._logs()
        # A target that never selects the logged slate zeros every weight.
        result = slate_standard_ips(logs, target_policy=lambda s: 0.0)
        assert result.V_hat == 0.0
        assert result.ESS == 0.0
        assert result.n == len(logs)

    def test_rips_zero_logging_prob_breaks_slate(self) -> None:
        logs = self._logs()
        # A custom logging policy returning 0 forces the per-slate weight to
        # collapse (the ``p_l <= 0`` break path).
        result = reward_interaction_ips(
            logs,
            target_policy_per_rank=lambda _k, _item: 1.0 / 6,
            logging_policy_per_rank=lambda _k, _item: 0.0,
        )
        assert result.V_hat == 0.0

    def test_pseudo_inverse_empty_logs(self) -> None:
        result = pseudo_inverse_ips(
            _empty_logs(), target_policy_per_rank=lambda _k, _item: 0.1
        )
        assert result.n == 0
        assert result.V_hat == 0.0

    def test_pseudo_inverse_infers_n_items(self) -> None:
        logs = self._logs()
        # n_items=None forces the catalogue size to be inferred from the logs.
        result = pseudo_inverse_ips(
            logs, target_policy_per_rank=lambda _k, _item: 1.0 / 6
        )
        assert np.isfinite(result.V_hat)

    def test_cascade_dr_empty_logs(self) -> None:
        result = slate_cascade_dr(
            _empty_logs(),
            target_policy_per_rank=lambda _k, _item: 0.1,
            q_hat_per_rank=np.zeros((0, 3, 6)),
        )
        assert result.n == 0

    def test_cascade_dr_bad_qhat_shape_raises(self) -> None:
        logs, attractiveness, _truth = make_slate_synth(
            n_impressions=30, n_items=6, slate_size=3, seed=3
        )
        n_items = attractiveness.shape[1]
        slate_size = len(logs["slate"].iloc[0])
        bad_q = np.zeros((len(logs), slate_size, n_items + 1))
        with pytest.raises(ValueError, match="q_hat_per_rank must have shape"):
            slate_cascade_dr(
                logs,
                target_policy_per_rank=lambda _k, _item: 1.0 / n_items,
                q_hat_per_rank=bad_q,
            )
