"""Publish paired EvalTables, window summaries, and JSONL receipts to a CoreWeave Forge Workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import wandb

from .analysis import mean_and_clustered_se, paired_window_delta, token_and_latency_summary
from .common import SUITE_VERSION, harness_content_hash


def publish_forge_workspace(
    records: list[dict[str, Any]],
    *,
    project: str,
    run_name: str,
    job_type: str = "agentnerf-window",
    validation_records: list[dict[str, Any]] | None = None,
    log_files: list[Path] | None = None,
) -> dict[str, Any]:
    """Log side-by-side wandb.EvalTable + summary metrics + evidence artifact to Forge."""
    entity, project_name = project.split("/", 1)
    run = wandb.init(
        entity=entity,
        project=project_name,
        name=run_name,
        job_type=job_type,
        config={
            "suite_version": SUITE_VERSION,
            "harness_sha": harness_content_hash(),
            "records_count": len(records),
            "validation_records_count": len(validation_records or []),
        },
        settings=wandb.Settings(init_timeout=60),
    )

    try:
        # 1. Multi-model side-by-side EvalTable on shared items
        models = sorted({f"{r['agent_name']}@{r['effort']}" for r in records if r.get("status") == "ok"})
        by_item: dict[str, dict[str, Any]] = {}
        for r in records:
            if r.get("status") != "ok":
                continue
            iid = r["item_id"]
            entry = by_item.setdefault(
                iid,
                {
                    "item_id": iid,
                    "family": r["family"],
                    "window": r["window"],
                    "item_sha256": r["item_sha256"][:12],
                },
            )
            key = f"{r['agent_name']}@{r['effort']}"
            outs = r.get("turn_outputs") or [""]
            entry[f"{key}_trace"] = (
                f"conv={r['conversation_id']} | failure={r['failure_mode']} | "
                f"tools={r['tool_calls']} (err={r['tool_errors']}) | "
                f"out_tok={r['output_tokens']} (think={r['reasoning_tokens']})\n"
                + ("\n---\n".join(o[:300] for o in outs if o))
            )
            entry[f"{key}_score"] = float(r["score"])
            entry[f"{key}_passed"] = bool(r["passed"])

        input_cols = ["item_id", "family", "window", "item_sha256"]
        output_cols = [f"{m}_trace" for m in models]
        score_cols = [col for m in models for col in (f"{m}_score", f"{m}_passed")]
        eval_rows = []
        for iid in sorted(by_item):
            e = by_item[iid]
            row = [e["item_id"], e["family"], e["window"], e["item_sha256"]]
            row.extend(e.get(c, "") for c in output_cols)
            row.extend(e.get(c) for c in score_cols)
            eval_rows.append(row)

        if eval_rows:
            panel_eval = wandb.EvalTable(
                columns=input_cols + output_cols + score_cols,
                data=eval_rows,
                input_columns=input_cols,
                output_columns=output_cols,
                score_columns=score_cols,
            )
            run.log({"agentnerf/panel_eval": panel_eval})

        # 2. Per-model summary table
        summary_rows = []
        summary_metrics: dict[str, Any] = {}
        by_model_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for r in records:
            by_model_key.setdefault((r["agent_name"], r["effort"], r["arm"]), []).append(r)

        for (agent_name, effort, arm), recs in sorted(by_model_key.items()):
            st = mean_and_clustered_se(recs)
            tl = token_and_latency_summary(recs)
            echoed = ",".join(tl["echoed_models"])
            summary_rows.append(
                [
                    agent_name,
                    arm,
                    effort,
                    echoed,
                    st["samples"],
                    st["mean"],
                    st["se"],
                    st["ci95"],
                    tl["pass_rate"],
                    tl["output_tokens_median"],
                    tl["reasoning_tokens_median"],
                    tl["tool_calls_mean"],
                    tl["tool_errors_total"],
                    tl["latency_ms_median"],
                ]
            )
            prefix = f"models/{agent_name}_{effort}"
            summary_metrics[f"{prefix}/score_mean"] = st["mean"]
            summary_metrics[f"{prefix}/score_se"] = st["se"]
            summary_metrics[f"{prefix}/pass_rate"] = tl["pass_rate"]
            summary_metrics[f"{prefix}/output_tokens_median"] = tl["output_tokens_median"]
            summary_metrics[f"{prefix}/reasoning_tokens_median"] = tl["reasoning_tokens_median"]

        model_table = wandb.Table(
            columns=[
                "model",
                "arm",
                "effort",
                "echoed_model",
                "samples",
                "score_mean",
                "clustered_se",
                "ci95",
                "exact_pass_rate",
                "output_tok_median",
                "reasoning_tok_median",
                "tool_calls_mean",
                "tool_errors",
                "latency_ms_p50",
            ],
            data=summary_rows,
        )
        run.log({"agentnerf/model_summary": model_table})

        # 3. Positive-control validation paired comparison (effort low vs high) if provided
        if validation_records:
            all_pool = records + validation_records
            val_models = sorted(
                {r["agent_name"] for r in validation_records if r.get("status") == "ok"}
            )
            val_rows = []
            paired_eval_rows = []
            for m_name in val_models:
                hi = [
                    r
                    for r in all_pool
                    if r["agent_name"] == m_name and r["effort"] == "high" and r.get("status") == "ok"
                ]
                lo = [
                    r
                    for r in all_pool
                    if r["agent_name"] == m_name and r["effort"] == "low" and r.get("status") == "ok"
                ]
                if not hi or not lo:
                    continue
                hi_st = mean_and_clustered_se(hi)
                lo_st = mean_and_clustered_se(lo)
                pd = paired_window_delta(hi, lo)
                hi_tl = token_and_latency_summary(hi)
                lo_tl = token_and_latency_summary(lo)
                tok_pct = (
                    round(
                        100.0 * (lo_tl["output_tokens_mean"] - hi_tl["output_tokens_mean"]) / hi_tl["output_tokens_mean"],
                        1,
                    )
                    if hi_tl["output_tokens_mean"] > 0
                    else 0.0
                )
                val_rows.append(
                    [
                        m_name,
                        pd["paired_items"],
                        hi_st["mean"],
                        lo_st["mean"],
                        pd["delta"],
                        pd["se"],
                        pd["sig95"],
                        hi_tl["output_tokens_median"],
                        lo_tl["output_tokens_median"],
                        tok_pct,
                        hi_tl["reasoning_tokens_median"],
                        lo_tl["reasoning_tokens_median"],
                    ]
                )
                summary_metrics[f"validation/{m_name}/score_delta_low_vs_high"] = pd["delta"]
                summary_metrics[f"validation/{m_name}/score_delta_se"] = pd["se"]
                summary_metrics[f"validation/{m_name}/output_tokens_pct_change"] = tok_pct

                hi_by_id = {r["item_id"]: r for r in hi}
                lo_by_id = {r["item_id"]: r for r in lo}
                for iid in sorted(set(hi_by_id) & set(lo_by_id)):
                    rh, rl = hi_by_id[iid], lo_by_id[iid]
                    paired_eval_rows.append(
                        [
                            m_name,
                            iid,
                            rh["family"],
                            f"conv={rh['conversation_id']} | mode={rh['failure_mode']} | tok={rh['output_tokens']} (think={rh['reasoning_tokens']})",
                            f"conv={rl['conversation_id']} | mode={rl['failure_mode']} | tok={rl['output_tokens']} (think={rl['reasoning_tokens']})",
                            float(rh["score"]),
                            float(rl["score"]),
                            round(float(rl["score"]) - float(rh["score"]), 4),
                        ]
                    )

            if val_rows:
                run.log(
                    {
                        "agentnerf/positive_control_summary": wandb.Table(
                            columns=[
                                "model",
                                "paired_items",
                                "high_effort_score",
                                "low_effort_score",
                                "paired_delta",
                                "clustered_se",
                                "sig95",
                                "high_out_tok_med",
                                "low_out_tok_med",
                                "out_tok_pct_delta",
                                "high_think_tok_med",
                                "low_think_tok_med",
                            ],
                            data=val_rows,
                        )
                    }
                )
            if paired_eval_rows:
                run.log(
                    {
                        "agentnerf/positive_control_eval": wandb.EvalTable(
                            columns=[
                                "model",
                                "item_id",
                                "family",
                                "high_effort_trace",
                                "low_effort_trace",
                                "high_score",
                                "low_score",
                                "score_delta",
                            ],
                            data=paired_eval_rows,
                            input_columns=["model", "item_id", "family"],
                            output_columns=["high_effort_trace", "low_effort_trace"],
                            score_columns=["high_score", "low_score", "score_delta"],
                        )
                    }
                )

        run.summary.update(summary_metrics)

        if log_files:
            art = wandb.Artifact(
                "agentnerf-evidence",
                type="benchmark-logs",
                description="Append-only JSONL probe logs and summary CSVs (no secrets).",
                metadata={"harness_sha": harness_content_hash(), "suite_version": SUITE_VERSION},
            )
            for lf in log_files:
                if lf.is_file():
                    art.add_file(str(lf))
            run.log_artifact(art)

        run_url, run_id = run.url, run.id
    finally:
        run.finish()

    return {
        "run_url": run_url,
        "run_id": run_id,
        "run_path": f"{project}/{run_id}",
        "agent_lens_url": f"https://forge.coreweave.com/agent-lens/{entity}/{project_name}",
    }
