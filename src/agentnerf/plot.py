"""Generate light and dark SVG benchmark figures directly from measured JSONL logs."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .analysis import mean_and_clustered_se, paired_window_delta, token_and_latency_summary
from .common import MEDIA_DIR

PALETTES: dict[str, dict[str, str]] = {
    "light": {
        "bg": "#FAF9F6",
        "panel": "#FFFFFF",
        "border": "#E2DFD7",
        "grid": "#ECEAE3",
        "text": "#18181B",
        "muted": "#65645E",
        "band": "#EEF4F0",
        "band_stroke": "#C6DDD0",
        "anthropic": "#D97757",
        "openai": "#10A37F",
        "control": "#71717A",
        "neg": "#C2410C",
        "pos": "#0F766E",
        "think": "#6366F1",
        "out": "#94A3B8",
    },
    "dark": {
        "bg": "#111215",
        "panel": "#181A1F",
        "border": "#2A2E37",
        "grid": "#232730",
        "text": "#F4F4F5",
        "muted": "#9CA3AF",
        "band": "#16241E",
        "band_stroke": "#244235",
        "anthropic": "#E68A6B",
        "openai": "#34D399",
        "control": "#9CA3AF",
        "neg": "#FB923C",
        "pos": "#2DD4BF",
        "think": "#818CF8",
        "out": "#475569",
    },
}


def _esc(text: Any) -> str:
    return html.escape(str(text))


def render_baseline_and_series_svg(
    records: list[dict[str, Any]],
    *,
    theme: str = "light",
    out_path: Path | None = None,
) -> str:
    """Render Day-0 / multi-day frontier model baseline + token/tool telemetry strip."""
    pal = PALETTES[theme]
    ok = [
        r
        for r in records
        if r.get("status") == "ok" and r.get("arm") in {"measured", "control"}
    ]

    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for r in ok:
        key = (r["arm"], r["provider"], r["agent_name"], r["effort"])
        groups.setdefault(key, []).append(r)

    # Sort measured first (anthropic then openai), then controls
    ordered_keys = sorted(
        groups.keys(),
        key=lambda k: (0 if k[0] == "measured" else 1, k[1], k[2]),
    )

    row_h = 46
    top_y = 112
    n_rows = max(1, len(ordered_keys))
    height = top_y + n_rows * row_h + 62
    width = 980

    # Left chart x-scale: score [0.50, 1.00] -> [235, 505]
    s_min, s_max = 0.50, 1.00
    x0, x1 = 235.0, 505.0

    def sx(val: float) -> float:
        clamped = max(s_min, min(s_max, val))
        return x0 + ((clamped - s_min) / (s_max - s_min)) * (x1 - x0)

    # Right chart x-scale: median output tokens -> [620, 790] (leaves 170px for right-hand label)
    tx0, tx1 = 620.0, 790.0
    max_tok = max(
        [token_and_latency_summary(groups[k])["output_tokens_median"] for k in ordered_keys] + [500]
    )

    def stx(tok: float) -> float:
        return tx0 + min(1.0, tok / (max_tok * 1.08)) * (tx1 - tx0)

    days = sorted({r["day"] for r in ok})
    day_label = f"{days[0]} (day 1 baseline)" if len(days) == 1 else f"{days[0]} .. {days[-1]} ({len(days)} days)"

    parts: list[str] = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'font-family="ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">'
        ),
        f'<rect width="{width}" height="{height}" fill="{pal["bg"]}" rx="10"/>',
        f'<rect x="16" y="16" width="{width - 32}" height="{height - 32}" fill="{pal["panel"]}" stroke="{pal["border"]}" rx="8"/>',
        f'<text x="36" y="48" fill="{pal["text"]}" font-size="16" font-weight="700">agentnerf · Frontier Agent Capability &amp; Token Baseline</text>',
        f'<text x="36" y="68" fill="{pal["muted"]}" font-size="12">Multi-turn tool probes traced in CoreWeave Agent Lens · { _esc(day_label) } · pure-function partial-credit score ± 95% clustered CI</text>',
        f'<text x="{(x0 + x1) / 2:.1f}" y="{top_y - 24}" fill="{pal["muted"]}" font-size="10" text-anchor="middle">partial-credit task score ± 95% clustered CI</text>',
        f'<text x="{tx0:.1f}" y="{top_y - 24}" fill="{pal["muted"]}" font-size="10">median output &amp; reasoning tokens / item</text>',
    ]

    for tick in (0.50, 0.75, 0.90, 1.00):
        tx = sx(tick)
        parts.append(
            f'<line x1="{tx:.1f}" y1="{top_y - 14}" x2="{tx:.1f}" y2="{top_y + n_rows * row_h - 6}" stroke="{pal["grid"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{tx:.1f}" y="{top_y + n_rows * row_h + 12}" fill="{pal["muted"]}" font-size="10" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{tick:.2f}</text>'
        )

    for idx, key in enumerate(ordered_keys):
        arm, provider, agent_name, effort = key
        recs = groups[key]
        st = mean_and_clustered_se(recs)
        tl = token_and_latency_summary(recs)
        cy = top_y + idx * row_h + 16

        if idx > 0 and ordered_keys[idx - 1][0] != arm:
            sep_y = top_y + idx * row_h - 6
            parts.append(
                f'<line x1="36" y1="{sep_y}" x2="{width - 36}" y2="{sep_y}" stroke="{pal["border"]}" stroke-dasharray="4,4"/>'
            )

        color = pal["control"] if arm == "control" else pal.get(provider, pal["text"])
        badge = "CTRL" if arm == "control" else provider.upper()[:4]
        parts.append(
            f'<text x="36" y="{cy + 1}" fill="{pal["text"]}" font-size="13" font-weight="600" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{_esc(agent_name)}</text>'
        )
        parts.append(
            f'<text x="36" y="{cy + 16}" fill="{pal["muted"]}" font-size="10">{_esc(badge)} · effort={_esc(effort)} · n={st["samples"]}</text>'
        )

        # Score CI bar + point + numeric readout
        lo = max(0.0, st["mean"] - st["ci95"])
        hi = min(1.0, st["mean"] + st["ci95"])
        cx = sx(st["mean"])
        parts.append(
            f'<line x1="{sx(lo):.1f}" y1="{cy}" x2="{sx(hi):.1f}" y2="{cy}" stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>'
        )
        parts.append(
            f'<line x1="{sx(lo):.1f}" y1="{cy - 4}" x2="{sx(lo):.1f}" y2="{cy + 4}" stroke="{color}" stroke-width="1.5"/>'
        )
        parts.append(
            f'<line x1="{sx(hi):.1f}" y1="{cy - 4}" x2="{sx(hi):.1f}" y2="{cy + 4}" stroke="{color}" stroke-width="1.5"/>'
        )
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy}" r="5" fill="{color}"/>')
        parts.append(
            f'<text x="{x1 + 10:.1f}" y="{cy + 4}" fill="{pal["text"]}" font-size="11" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{st["mean"]:.3f} ±{st["ci95"]:.2f}</text>'
        )

        # Token bar (total output in slate, reasoning subset overlaid in indigo)
        out_med = tl["output_tokens_median"]
        think_med = tl["reasoning_tokens_median"]
        out_w = max(2.0, stx(out_med) - tx0)
        parts.append(
            f'<rect x="{tx0:.1f}" y="{cy - 6}" width="{out_w:.1f}" height="12" fill="{pal["out"]}" opacity="0.45" rx="3"/>'
        )
        if think_med > 0:
            th_w = max(2.0, stx(think_med) - tx0)
            parts.append(
                f'<rect x="{tx0:.1f}" y="{cy - 6}" width="{min(out_w, th_w):.1f}" height="12" fill="{pal["think"]}" rx="3"/>'
            )
        tok_label = f"{out_med:,} tok" + (f" ({think_med:,} think)" if think_med > 0 else "")
        parts.append(
            f'<text x="{tx0 + out_w + 8:.1f}" y="{cy + 4}" fill="{pal["muted"]}" font-size="10.5" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{_esc(tok_label)}</text>'
        )

    # Legend footer
    fy = height - 28
    parts.append(f'<circle cx="42" cy="{fy - 4}" r="4.5" fill="{pal["anthropic"]}"/>')
    parts.append(f'<text x="52" y="{fy}" fill="{pal["muted"]}" font-size="11">Anthropic measured</text>')
    parts.append(f'<circle cx="178" cy="{fy - 4}" r="4.5" fill="{pal["openai"]}"/>')
    parts.append(f'<text x="188" y="{fy}" fill="{pal["muted"]}" font-size="11">OpenAI measured</text>')
    parts.append(f'<circle cx="305" cy="{fy - 4}" r="4.5" fill="{pal["control"]}"/>')
    parts.append(f'<text x="315" y="{fy}" fill="{pal["muted"]}" font-size="11">Control arm</text>')
    parts.append(f'<rect x="415" y="{fy - 9}" width="10" height="10" fill="{pal["think"]}" rx="2"/>')
    parts.append(f'<text x="430" y="{fy}" fill="{pal["muted"]}" font-size="11">Reasoning / thinking tokens</text>')
    parts.append("</svg>\n")

    svg = "\n".join(parts)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(svg, encoding="utf-8")
    return svg


def render_positive_control_svg(
    high_records: list[dict[str, Any]],
    low_records: list[dict[str, Any]],
    *,
    theme: str = "light",
    out_path: Path | None = None,
) -> str:
    """Render paired positive-control chart (effort=low vs effort=high) on score and tokens."""
    pal = PALETTES[theme]
    models = sorted(
        {r["agent_name"] for r in low_records if r.get("status") == "ok"}
        & {r["agent_name"] for r in high_records if r.get("status") == "ok"}
    )
    row_h = 54
    top_y = 114
    n_rows = max(1, len(models))
    width = 980
    height = top_y + n_rows * row_h + 58

    # Left panel: paired score delta [-0.12, +0.06] -> [235, 480]
    dx0, dx1 = 235.0, 480.0
    d_min, d_max = -0.12, 0.06

    def sdx(d: float) -> float:
        clamped = max(d_min, min(d_max, d))
        return dx0 + ((clamped - d_min) / (d_max - d_min)) * (dx1 - dx0)

    # Right panel: reasoning token % change [-85%, +15%] -> [605, 795] (leaves 165px on right)
    px0, px1 = 605.0, 795.0

    def spx(pct: float) -> float:
        clamped = max(-85.0, min(15.0, pct))
        return px0 + ((clamped - (-85.0)) / 100.0) * (px1 - px0)

    parts: list[str] = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'font-family="ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">'
        ),
        f'<rect width="{width}" height="{height}" fill="{pal["bg"]}" rx="10"/>',
        f'<rect x="16" y="16" width="{width - 32}" height="{height - 32}" fill="{pal["panel"]}" stroke="{pal["border"]}" rx="8"/>',
        f'<text x="36" y="48" fill="{pal["text"]}" font-size="16" font-weight="700">Positive-Control Validation · Effort Low vs. Effort High</text>',
        f'<text x="36" y="68" fill="{pal["muted"]}" font-size="12">Known effort downgrade tested on identical paired items · validates Agent Lens version &amp; token sensitivity before daily monitoring</text>',
        f'<text x="{(dx0 + dx1) / 2:.1f}" y="{top_y - 22}" fill="{pal["muted"]}" font-size="10" text-anchor="middle">paired score shift Δ (low − high) ± SE</text>',
        f'<text x="{(px0 + px1) / 2:.1f}" y="{top_y - 22}" fill="{pal["muted"]}" font-size="10" text-anchor="middle">median reasoning token change (%)</text>',
    ]

    for dtick in (-0.10, -0.05, 0.0, 0.05):
        tx = sdx(dtick)
        stroke_w = "1.5" if dtick == 0.0 else "1"
        parts.append(
            f'<line x1="{tx:.1f}" y1="{top_y - 12}" x2="{tx:.1f}" y2="{top_y + n_rows * row_h - 8}" stroke="{pal["border"] if dtick == 0.0 else pal["grid"]}" stroke-width="{stroke_w}"/>'
        )
        parts.append(
            f'<text x="{tx:.1f}" y="{top_y + n_rows * row_h + 10}" fill="{pal["muted"]}" font-size="10" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{dtick:+.2f}</text>'
        )

    zero_px = spx(0.0)
    parts.append(
        f'<line x1="{zero_px:.1f}" y1="{top_y - 12}" x2="{zero_px:.1f}" y2="{top_y + n_rows * row_h - 8}" stroke="{pal["border"]}" stroke-width="1.5"/>'
    )

    for idx, m_name in enumerate(models):
        hi = [r for r in high_records if r["agent_name"] == m_name and r.get("status") == "ok"]
        lo = [r for r in low_records if r["agent_name"] == m_name and r.get("status") == "ok"]
        pd = paired_window_delta(hi, lo)
        hi_tl = token_and_latency_summary(hi)
        lo_tl = token_and_latency_summary(lo)
        out_pct = (
            100.0 * (lo_tl["output_tokens_median"] - hi_tl["output_tokens_median"]) / hi_tl["output_tokens_median"]
            if hi_tl["output_tokens_median"] > 0
            else 0.0
        )
        think_pct = (
            100.0 * (lo_tl["reasoning_tokens_median"] - hi_tl["reasoning_tokens_median"]) / hi_tl["reasoning_tokens_median"]
            if hi_tl["reasoning_tokens_median"] > 0
            else out_pct
        )
        cy = top_y + idx * row_h + 18

        parts.append(
            f'<text x="36" y="{cy}" fill="{pal["text"]}" font-size="13" font-weight="600" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{_esc(m_name)}</text>'
        )
        parts.append(
            f'<text x="36" y="{cy + 16}" fill="{pal["muted"]}" font-size="10">{pd["paired_items"]} paired items · high={mean_and_clustered_se(hi)["mean"]:.2f} → low={mean_and_clustered_se(lo)["mean"]:.2f}</text>'
        )

        col = pal["neg"] if pd["delta"] < 0 else pal["pos"]
        lo_x = sdx(pd["delta"] - pd["se"])
        hi_x = sdx(pd["delta"] + pd["se"])
        cx = sdx(pd["delta"])
        parts.append(
            f'<line x1="{sdx(0.0):.1f}" y1="{cy}" x2="{cx:.1f}" y2="{cy}" stroke="{col}" stroke-width="2" stroke-dasharray="3,2"/>'
        )
        parts.append(
            f'<line x1="{lo_x:.1f}" y1="{cy}" x2="{hi_x:.1f}" y2="{cy}" stroke="{col}" stroke-width="3.5" stroke-linecap="round"/>'
        )
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy}" r="5.5" fill="{col}"/>')
        parts.append(
            f'<text x="{dx1 + 8:.1f}" y="{cy + 4}" fill="{pal["text"]}" font-size="11" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{pd["delta"]:+.3f} ±{pd["se"]:.2f}</text>'
        )

        # Reasoning token % bar from zero_px
        px = spx(think_pct)
        bar_x = min(zero_px, px)
        bar_w = max(2.0, abs(zero_px - px))
        t_col = pal["think"] if think_pct < 0 else pal["pos"]
        parts.append(
            f'<rect x="{bar_x:.1f}" y="{cy - 7}" width="{bar_w:.1f}" height="14" fill="{t_col}" rx="3"/>'
        )
        parts.append(
            f'<text x="{px1 + 8:.1f}" y="{cy + 4}" fill="{pal["text"]}" font-size="10.5" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{think_pct:+.0f}% think ({out_pct:+.0f}% out)</text>'
        )

    parts.append("</svg>\n")
    svg = "\n".join(parts)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(svg, encoding="utf-8")
    return svg


def render_all_media(
    baseline_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]] | None = None,
    *,
    media_dir: Path = MEDIA_DIR,
) -> list[Path]:
    written: list[Path] = []
    for theme, suffix in (("light", ""), ("dark", "-dark")):
        p1 = media_dir / f"agentnerf{suffix}.svg"
        render_baseline_and_series_svg(baseline_records, theme=theme, out_path=p1)
        written.append(p1)
        if validation_records:
            p2 = media_dir / f"positive-control{suffix}.svg"
            render_positive_control_svg(
                baseline_records, validation_records, theme=theme, out_path=p2
            )
            written.append(p2)
    return written
