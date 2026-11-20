"""Unit tests for clustered standard errors, paired deltas, and the 2-window decision rule."""

from __future__ import annotations

import pytest

from agentnerf.analysis import (
    cusum_series,
    evaluate_series_decisions,
    mean_and_clustered_se,
    paired_window_delta,
)


def test_paired_window_delta_cancels_between_item_variance() -> None:
    # Items vary widely in difficulty (0.2..0.9), yet a uniform -0.10 drop per item has SE=0!
    base = [
        {"item_id": f"it-{i}", "status": "ok", "score": 0.2 + 0.1 * i}
        for i in range(8)
    ]
    win = [
        {"item_id": f"it-{i}", "status": "ok", "score": 0.1 + 0.1 * i}
        for i in range(8)
    ]
    unpaired = mean_and_clustered_se(base)
    assert unpaired["se"] > 0.08

    paired = paired_window_delta(base, win)
    assert paired["delta"] == pytest.approx(-0.10)
    assert paired["se"] == pytest.approx(0.0)
    assert paired["paired_items"] == 8


def test_two_window_decision_rule_and_control_guard() -> None:
    w_shift_1 = {"delta": -0.09, "sig99": True}
    w_shift_2 = {"delta": -0.11, "sig99": True}
    w_noise = {"delta": -0.02, "sig99": False}

    decisions = evaluate_series_decisions([w_shift_1, w_shift_2, w_noise])
    assert decisions == ["watch (1 of 2 windows)", "regression", "no change"]

    # If control arm also drops significantly in window 2, attribute to harness/platform shift
    ctrl = [{"delta": 0.0, "sig99": False}, {"delta": -0.08, "sig99": True}]
    decisions_ctrl = evaluate_series_decisions([w_shift_1, w_shift_2], control_deltas=ctrl)
    assert decisions_ctrl == ["watch (1 of 2 windows)", "harness/platform shift (control moved)"]


def test_cusum_detects_sustained_negative_drift() -> None:
    res = cusum_series([-0.06, -0.07, -0.08, -0.06], slack=0.02, threshold=0.12)
    assert len(res["alarms"]) >= 2
