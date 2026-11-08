"""Read-only loaders for Streamlit. Does not run or change the pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.config import ROOT_DIR, get_settings


def outputs_root() -> Path:
    return ROOT_DIR / "outputs"


def list_runs(root: Path | None = None) -> list[Path]:
    base = root or outputs_root()
    if not base.exists():
        return []
    runs = [p for p in base.iterdir() if p.is_dir() and (p / "promos.json").exists()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except OSError:
        return rows
    return rows


def run_mode_label(meta: dict[str, Any]) -> str:
    if meta.get("mock_llm") is True or str(meta.get("model") or "").lower() == "mock":
        return "MOCK"
    return "REAL RUN"


def describe_run(run_dir: Path, meta: dict[str, Any], quality: dict[str, Any]) -> str:
    """Human label from stored metadata. Does not assume pack size."""
    input_dir = str(meta.get("input_dir") or "")
    episodes = quality.get("episodes_parsed")
    limit = meta.get("limit_episodes")
    bits = [run_dir.name]
    if "fixtures" in input_dir.replace("\\", "/").lower():
        bits.append("fixture inputs")
    elif episodes is not None:
        bits.append(f"{episodes} episodes parsed")
    if limit:
        bits.append(f"limit {limit}")
    return " · ".join(bits)


def classify_status(
    record: dict[str, Any],
    *,
    final_ids: set[str],
) -> str:
    window_id = (record.get("candidate") or {}).get("window_id") or record.get("window_id")
    if record.get("llm_failed"):
        return "LLM failed"
    if window_id in final_ids:
        return "Final promo"
    if record.get("validation_passed") is False:
        return "Hard validation rejected"
    if record.get("validation_passed") is True:
        return "Passed validation, not shortlisted"
    return "Scored, validation not recorded"


def _read_scored_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                wid = row.get("window_id")
                if wid:
                    out[wid] = row
    except OSError:
        return out
    return out


def _as_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def _merge_csv(record: dict[str, Any], row: dict[str, str]) -> None:
    """Checkpoint is written before validation. CSV holds the later result."""
    passed = _as_bool(row.get("validation_passed"))
    if passed is not None:
        record["validation_passed"] = passed
    failures = (row.get("validation_failures") or "").strip()
    if failures:
        record["validation_failures"] = [part.strip() for part in failures.split(";") if part.strip()]
    if record.get("composite_score") in (None, "") and row.get("composite_score"):
        try:
            record["composite_score"] = float(row["composite_score"])
        except ValueError:
            pass
    extended = _as_bool(row.get("extended"))
    if extended is not None and record.get("extended") is None:
        record["extended"] = extended
    llm_failed = _as_bool(row.get("llm_failed"))
    if llm_failed is True:
        record["llm_failed"] = True
    if row.get("llm_error") and not record.get("llm_error"):
        record["llm_error"] = row["llm_error"]


def load_run(run_dir: Path) -> dict[str, Any]:
    """Load one run folder. Missing files become empty structures, not invented rows."""
    promos_doc = _read_json(run_dir / "promos.json") or {}
    quality = _read_json(run_dir / "quality_report.json") or {}
    run_log = _read_json(run_dir / "run_log.json") or {}
    meta = promos_doc.get("run") or run_log.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    promos = promos_doc.get("promos") if isinstance(promos_doc.get("promos"), list) else []
    scored = _read_jsonl(run_dir / "checkpoint_scored.jsonl")
    csv_by_window = _read_scored_csv(run_dir / "all_scored_candidates.csv")

    final_ids = {str(p.get("window_id")) for p in promos if p.get("window_id")}
    for rec in scored:
        wid = str((rec.get("candidate") or {}).get("window_id") or rec.get("window_id") or "")
        if wid in csv_by_window:
            _merge_csv(rec, csv_by_window[wid])
        rec["_window_id"] = wid
        rec["_status"] = classify_status(rec, final_ids=final_ids)

    candidate_stats = run_log.get("candidate_stats") if isinstance(run_log.get("candidate_stats"), dict) else {}
    per_episode = candidate_stats.get("per_episode") if isinstance(candidate_stats.get("per_episode"), list) else []

    settings = get_settings()
    weights = {
        "hook": settings.scoring.hook_weight,
        "cliffhanger": settings.scoring.cliffhanger_weight,
        "standalone": settings.scoring.standalone_weight,
    }

    return {
        "run_dir": str(run_dir),
        "run_id": meta.get("run_id") or run_dir.name,
        "meta": meta,
        "mode": run_mode_label(meta),
        "description": describe_run(run_dir, meta, quality if isinstance(quality, dict) else {}),
        "quality": quality if isinstance(quality, dict) else {},
        "candidate_stats": candidate_stats,
        "prefilter_per_episode": per_episode,
        "promos": promos,
        "scored": scored,
        "weights": weights,
        "failures_md": (run_dir / "failures.md").read_text(encoding="utf-8")
        if (run_dir / "failures.md").exists()
        else "",
        "files_present": {
            "promos.json": (run_dir / "promos.json").exists(),
            "quality_report.json": (run_dir / "quality_report.json").exists(),
            "checkpoint_scored.jsonl": (run_dir / "checkpoint_scored.jsonl").exists(),
            "all_scored_candidates.csv": (run_dir / "all_scored_candidates.csv").exists(),
            "run_log.json": (run_dir / "run_log.json").exists(),
            "failures.md": (run_dir / "failures.md").exists(),
        },
    }


def promo_segment(promo: dict[str, Any]) -> dict[str, Any]:
    segs = promo.get("segments") or []
    if segs and isinstance(segs[0], dict):
        return segs[0]
    return {}


def find_scored(bundle: dict[str, Any], window_id: str) -> dict[str, Any] | None:
    for rec in bundle.get("scored") or []:
        if rec.get("_window_id") == window_id:
            return rec
    return None


def ranking_breakdown(record: dict[str, Any], weights: dict[str, float]) -> dict[str, Any] | None:
    """Recompute the published formula from stored component scores. No invented scores."""
    ctx = record.get("contextual") or {}
    stand = record.get("standalone") or {}
    hook = (ctx.get("hook") or {}).get("score")
    cliff = (ctx.get("cliffhanger") or {}).get("score")
    alone = stand.get("standalone_score")
    if hook is None or cliff is None or alone is None:
        return None
    try:
        hook_f = float(hook)
        cliff_f = float(cliff)
        alone_f = float(alone)
    except (TypeError, ValueError):
        return None
    recomputed = (
        weights.get("hook", 0.4) * hook_f
        + weights.get("cliffhanger", 0.35) * cliff_f
        + weights.get("standalone", 0.25) * alone_f
    )
    stored = record.get("composite_score")
    return {
        "hook": hook_f,
        "cliffhanger": cliff_f,
        "standalone": alone_f,
        "weights": weights,
        "recomputed": recomputed,
        "stored": stored,
    }
