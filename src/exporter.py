"""Stage 9 — export JSON/CSV/reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.schemas import FinalPromo, RunMetadata, ScoredCandidate, ShowProfile


def export_promos_json(path: Path, promos: list[FinalPromo], meta: RunMetadata) -> None:
    payload = {
        "run": meta.model_dump(),
        "promos": [p.model_dump() for p in promos],
        "count": len(promos),
        "note": "Each promo is an independent advertisement clip recommendation.",
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_promos_csv(path: Path, promos: list[FinalPromo]) -> None:
    rows = []
    for p in promos:
        seg = p.segments[0] if p.segments else None
        rows.append(
            {
                "rank": p.rank,
                "promo_id": p.promo_id,
                "episode_number": p.episode_number,
                "start_timestamp": seg.start_timestamp if seg else "",
                "end_timestamp": seg.end_timestamp if seg else "",
                "duration_sec": p.duration_sec,
                "hook_score": p.scores.get("hook"),
                "cliffhanger_score": p.scores.get("cliffhanger"),
                "standalone_score": p.scores.get("standalone"),
                "composite_score": p.composite_score,
                "hook_quote": p.hook.get("quote"),
                "cliffhanger_quote": p.cliffhanger.get("quote"),
                "why_it_works": p.why_it_works,
                "human_review_status": p.human_review_status,
                "window_id": p.window_id,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def export_all_scored_csv(path: Path, scored: list[ScoredCandidate]) -> None:
    rows = []
    for s in scored:
        rows.append(
            {
                "window_id": s.candidate.window_id,
                "episode_number": s.candidate.episode_number,
                "start_timestamp": s.candidate.start_timestamp,
                "end_timestamp": s.candidate.end_timestamp,
                "duration_sec": s.candidate.duration_sec,
                "prefilter_score": s.candidate.prefilter_score,
                "hook_score": s.contextual.hook.score if s.contextual else None,
                "cliffhanger_score": (
                    s.contextual.cliffhanger.score if s.contextual else None
                ),
                "standalone_score": (
                    s.standalone.standalone_score if s.standalone else None
                ),
                "composite_score": s.composite_score,
                "validation_passed": s.validation_passed,
                "validation_failures": "; ".join(s.validation_failures),
                "llm_failed": s.llm_failed,
                "llm_error": s.llm_error or "",
                "extended": s.extended,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def export_failures_md(
    path: Path,
    *,
    validation_rejects: list[ScoredCandidate],
    llm_failures: list[ScoredCandidate],
) -> None:
    lines = ["# Failures and rejections", ""]
    lines.append("## LLM failures")
    if not llm_failures:
        lines.append("- None")
    for s in llm_failures:
        lines.append(
            f"- `{s.candidate.window_id}` ep{s.candidate.episode_number}: {s.llm_error}"
        )

    lines.append("")
    lines.append("## Hard validation rejects")
    if not validation_rejects:
        lines.append("- None")
    for s in validation_rejects:
        lines.append(
            f"- `{s.candidate.window_id}`: " + "; ".join(s.validation_failures)
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_quality_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def export_run_log(path: Path, meta: RunMetadata, extra: dict[str, Any]) -> None:
    payload = {"metadata": meta.model_dump(), **extra}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_supporting_json(
    run_dir: Path,
    *,
    parser_report: dict[str, Any],
    show_profile: ShowProfile,
) -> None:
    (run_dir / "parser_report.json").write_text(
        json.dumps(parser_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "show_profile.json").write_text(
        show_profile.model_dump_json(indent=2), encoding="utf-8"
    )


def build_quality_report(
    *,
    parser_report: dict[str, Any],
    candidate_stats: dict[str, Any],
    scored: list[ScoredCandidate],
    promos: list[FinalPromo],
    meta: RunMetadata,
) -> dict[str, Any]:
    validated = [s for s in scored if s.validation_passed]
    rejected_val = [s for s in scored if s.validation_passed is False]
    llm_fail = [s for s in scored if s.llm_failed]
    ep_dist: dict[str, int] = {}
    for p in promos:
        key = str(p.episode_number)
        ep_dist[key] = ep_dist.get(key, 0) + 1

    def _avg(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    return {
        "run_id": meta.run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files_discovered": parser_report.get("files_discovered"),
        "episodes_parsed": parser_report.get("files_parsed"),
        "block_totals": parser_report.get("block_totals"),
        "raw_candidates": candidate_stats.get("raw_candidates"),
        "candidates_after_nms": candidate_stats.get("after_nms"),
        "llm_success": len(scored) - len(llm_fail),
        "llm_failure": len(llm_fail),
        "hard_validation_rejects": len(rejected_val),
        "hard_validation_reject_reasons": _reason_hist(rejected_val),
        "final_promos": len(promos),
        "promo_episode_distribution": ep_dist,
        "avg_hook": _avg([p.scores["hook"] for p in promos]),
        "avg_cliffhanger": _avg([p.scores["cliffhanger"] for p in promos]),
        "avg_standalone": _avg([p.scores["standalone"] for p in promos]),
        "validated_count": len(validated),
        "limitations": [
            "Screenplay-only analysis cannot verify acting, cinematography, or music.",
            "Event-boundary windows may miss ideal cuts inside long events.",
            "Heuristic pre-filtering can miss quiet hooks.",
            "LLM scores are subjective.",
        ],
    }


def _reason_hist(rejects: list[ScoredCandidate]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for s in rejects:
        for r in s.validation_failures:
            hist[r] = hist.get(r, 0) + 1
    return hist
