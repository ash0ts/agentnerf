"""Statistical analysis following Evan Miller (arXiv:2411.00640) & livenerf pre-registered rules.

- Paired item-level differences between a comparison window and the baseline window:
  d_i = mean(y_{i, window}) - mean(y_{i, baseline}), averaged equally across paired items i = 1..K.
- Clustered standard errors at the item level: SE(d_bar) = sqrt(s^2(d_1..d_K) / K).
- 95% CI (z = 1.96) for reporting and 99% CI (z = 2.5758) for the pre-registered drift decision rule.
- Pre-registered decision rule (matching livenerf):
  A model is flagged as drifted ('regression' or 'improvement') iff:
  1. Two consecutive post-baseline windows exclude 0 at 99% confidence in the same direction;
  2. |delta| >= min_effect (default 0.05 = 5.0 percentage points on partial-credit score);
  3. The provider's control arm does not exhibit a matching significant shift in the same direction.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import median
from typing import Any

Z_95 = 1.959963984540054
Z_99 = 2.5758293035489004
MIN_EFFECT = 0.05  # 5.0 percentage points


def mean_and_clustered_se(records: list[dict[str, Any]], value_key: str = "score") -> dict[str, Any]:
    """Compute item-clustered mean and standard error across sample records."""
    by_item: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r.get("status") != "ok":
            continue
        val = r.get(value_key)
        if val is not None:
            by_item[str(r["item_id"])].append(float(val))
    item_means = [sum(vals) / len(vals) for vals in by_item.values() if vals]
    k = len(item_means)
    n = sum(len(vals) for vals in by_item.values())
    if k == 0:
        return {"mean": 0.0, "se": 0.0, "ci95": 0.0, "ci99": 0.0, "items": 0, "samples": 0}
    mu = sum(item_means) / k
    if k == 1:
        se = 0.0
    else:
        var = sum((x - mu) ** 2 for x in item_means) / (k - 1)
        se = math.sqrt(var / k)
    return {
        "mean": round(mu, 6),
        "se": round(se, 6),
        "ci95": round(Z_95 * se, 6),
        "ci99": round(Z_99 * se, 6),
        "items": k,
        "samples": n,
    }


def paired_window_delta(
    baseline_records: list[dict[str, Any]],
    window_records: list[dict[str, Any]],
    value_key: str = "score",
) -> dict[str, Any]:
    """Paired per-item difference (window - baseline) with item-clustered standard error."""
    base_by_item: dict[str, list[float]] = defaultdict(list)
    win_by_item: dict[str, list[float]] = defaultdict(list)
    for r in baseline_records:
        if r.get("status") == "ok" and r.get(value_key) is not None:
            base_by_item[str(r["item_id"])].append(float(r[value_key]))
    for r in window_records:
        if r.get("status") == "ok" and r.get(value_key) is not None:
            win_by_item[str(r["item_id"])].append(float(r[value_key]))

    shared_items = sorted(set(base_by_item) & set(win_by_item))
    k = len(shared_items)
    if k == 0:
        return {
            "delta": 0.0,
            "se": 0.0,
            "ci95": 0.0,
            "ci99": 0.0,
            "paired_items": 0,
            "sig95": False,
            "sig99": False,
            "item_deltas": {},
        }

    item_deltas: dict[str, float] = {}
    for item_id in shared_items:
        b_mu = sum(base_by_item[item_id]) / len(base_by_item[item_id])
        w_mu = sum(win_by_item[item_id]) / len(win_by_item[item_id])
        item_deltas[item_id] = w_mu - b_mu

    diffs = list(item_deltas.values())
    delta = sum(diffs) / k
    if k == 1:
        se = 0.0
    else:
        var = sum((d - delta) ** 2 for d in diffs) / (k - 1)
        se = math.sqrt(var / k)

    ci95 = Z_95 * se
    ci99 = Z_99 * se
    sig95 = k >= 2 and se > 0 and abs(delta) > ci95
    sig99 = k >= 2 and se > 0 and abs(delta) > ci99
    return {
        "delta": round(delta, 6),
        "se": round(se, 6),
        "ci95": round(ci95, 6),
        "ci99": round(ci99, 6),
        "paired_items": k,
        "sig95": sig95,
        "sig99": sig99,
        "item_deltas": {k_id: round(v, 6) for k_id, v in item_deltas.items()},
    }


def token_and_latency_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r.get("status") == "ok"]
    if not ok:
        return {
            "output_tokens_median": 0,
            "reasoning_tokens_median": 0,
            "output_tokens_mean": 0.0,
            "reasoning_tokens_mean": 0.0,
            "latency_ms_median": 0,
            "tool_calls_mean": 0.0,
            "tool_errors_total": 0,
            "pass_rate": 0.0,
            "echoed_models": [],
        }
    out_toks = [int(r.get("output_tokens", 0)) for r in ok]
    reas_toks = [int(r.get("reasoning_tokens", 0)) for r in ok]
    lats = [int(r.get("duration_ms", 0)) for r in ok]
    tools = [int(r.get("tool_calls", 0)) for r in ok]
    errs = sum(int(r.get("tool_errors", 0)) for r in ok)
    passes = sum(1 for r in ok if r.get("passed"))
    echoed = sorted({str(r.get("echoed_model", "")) for r in ok if r.get("echoed_model")})
    return {
        "output_tokens_median": round(median(out_toks)),
        "reasoning_tokens_median": round(median(reas_toks)),
        "output_tokens_mean": round(sum(out_toks) / len(ok), 2),
        "reasoning_tokens_mean": round(sum(reas_toks) / len(ok), 2),
        "latency_ms_median": round(median(lats)),
        "tool_calls_mean": round(sum(tools) / len(ok), 2),
        "tool_errors_total": errs,
        "pass_rate": round(passes / len(ok), 6),
        "echoed_models": echoed,
    }


def evaluate_series_decisions(
    window_deltas: list[dict[str, Any]],
    *,
    control_deltas: list[dict[str, Any]] | None = None,
    min_effect: float = MIN_EFFECT,
) -> list[str]:
    """Apply the pre-registered 2-consecutive-window 99%-CI + control-arm exclusion rule."""
    decisions: list[str] = []
    for idx, wd in enumerate(window_deltas):
        if idx == 0:
            prev_dir = 0
        else:
            prev = window_deltas[idx - 1]
            prev_sig = bool(prev.get("sig99")) and abs(float(prev.get("delta", 0.0))) >= min_effect
            prev_dir = (1 if prev["delta"] > 0 else -1) if prev_sig else 0

        cur_sig = bool(wd.get("sig99")) and abs(float(wd.get("delta", 0.0))) >= min_effect
        cur_dir = (1 if wd["delta"] > 0 else -1) if cur_sig else 0

        ctrl_same_dir = False
        if control_deltas and idx < len(control_deltas):
            cd = control_deltas[idx]
            if cd.get("sig99") and abs(float(cd.get("delta", 0.0))) >= min_effect:
                ctrl_dir = 1 if cd["delta"] > 0 else -1
                if ctrl_dir == cur_dir:
                    ctrl_same_dir = True

        if cur_dir != 0 and cur_dir == prev_dir and not ctrl_same_dir:
            decisions.append("improvement" if cur_dir > 0 else "regression")
        elif cur_dir != 0 and ctrl_same_dir:
            decisions.append("harness/platform shift (control moved)")
        elif cur_dir != 0:
            decisions.append("watch (1 of 2 windows)")
        else:
            decisions.append("no change")
    return decisions


def cusum_series(daily_paired_deltas: list[float], *, slack: float = 0.025, threshold: float = 0.12) -> dict[str, Any]:
    """Two-sided tabular CUSUM sequential drift detector on daily paired deltas."""
    pos = 0.0
    neg = 0.0
    pos_trace: list[float] = []
    neg_trace: list[float] = []
    alarm_indices: list[int] = []
    for idx, d in enumerate(daily_paired_deltas):
        pos = max(0.0, pos + d - slack)
        neg = max(0.0, neg - d - slack)
        pos_trace.append(round(pos, 6))
        neg_trace.append(round(neg, 6))
        if pos >= threshold or neg >= threshold:
            alarm_indices.append(idx)
    return {
        "pos_cusum": pos_trace,
        "neg_cusum": neg_trace,
        "threshold": threshold,
        "slack": slack,
        "alarms": alarm_indices,
    }


def predicted_mde_99(baseline_records: list[dict[str, Any]], *, days_per_window: int = 10, power_z: float = 0.8416) -> float:
    """Estimate minimum detectable effect (99% two-sided, 80% power) from observed item variance."""
    stats = mean_and_clustered_se(baseline_records)
    k = stats["items"]
    if k <= 1:
        return 0.0
    # Conservative Bernoulli/partial-credit upper bound scaled by window resampling factor
    mu = min(0.95, max(0.05, stats["mean"]))
    within_item_sd = math.sqrt(mu * (1.0 - mu) * (2.0 / max(1, days_per_window)))
    se_window = within_item_sd / math.sqrt(k)
    return round((Z_99 + power_z) * se_window, 4)
