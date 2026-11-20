"""Tests for CLI lock/dry-run, MemoryTracer span hierarchy, and SVG figure generation."""

from __future__ import annotations

from pathlib import Path

from agentnerf.cli import main
from agentnerf.common import HARNESS_LOCK, harness_content_hash
from agentnerf.plot import render_all_media


def test_lock_and_dry_run() -> None:
    main(["lock", "--items-per-family", "4"])
    assert HARNESS_LOCK.is_file()
    assert HARNESS_LOCK.read_text().strip() == harness_content_hash()
    main(["run", "--panel", "public", "--items-per-family", "2", "--dry-run"])


def test_svg_rendering_light_and_dark(tmp_path: Path) -> None:
    base_records = [
        {
            "day": "2026-11-20",
            "window": "baseline",
            "arm": "measured",
            "provider": "anthropic",
            "agent_name": "claude-opus-5-5",
            "effort": "high",
            "item_id": f"ledger_reconcile-00{i}",
            "status": "ok",
            "score": 0.65 if i % 2 == 0 else 1.0,
            "passed": i % 2 == 1,
            "output_tokens": 920 + 40 * i,
            "reasoning_tokens": 410 + 20 * i,
            "tool_calls": 7,
            "tool_errors": 0,
            "duration_ms": 6200,
            "echoed_model": "claude-opus-5-5",
        }
        for i in range(4)
    ] + [
        {
            "day": "2026-11-20",
            "window": "baseline",
            "arm": "measured",
            "provider": "openai",
            "agent_name": "gpt-5.5",
            "effort": "high",
            "item_id": f"ledger_reconcile-00{i}",
            "status": "ok",
            "score": 0.65 if i == 0 else 1.0,
            "passed": i != 0,
            "output_tokens": 880 + 30 * i,
            "reasoning_tokens": 390 + 15 * i,
            "tool_calls": 7,
            "tool_errors": 0,
            "duration_ms": 5400,
            "echoed_model": "gpt-5.5-2026-04-23",
        }
        for i in range(4)
    ]
    low_records = [
        {**r, "window": "validation", "arm": "validation_low", "effort": "low", "score": max(0.0, r["score"] - 0.25), "output_tokens": 340, "reasoning_tokens": 40}
        for r in base_records
    ]
    written = render_all_media(base_records, low_records, media_dir=tmp_path)
    assert len(written) == 4
    for p in written:
        text = p.read_text(encoding="utf-8")
        assert text.startswith("<svg ")
        assert "claude-opus-5-5" in text
        assert "gpt-5.5" in text
