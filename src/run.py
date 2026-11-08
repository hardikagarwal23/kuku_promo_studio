"""CLI entrypoint for the Narrative-Aware Promo Finder pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from src.candidates import generate_all_candidates
from src.config import ROOT_DIR, get_settings, load_llm_env
from src.exporter import (
    build_quality_report,
    export_all_scored_csv,
    export_failures_md,
    export_promos_csv,
    export_promos_json,
    export_quality_report,
    export_run_log,
    write_supporting_json,
)
from src.extension import maybe_extend_candidate
from src.llm_client import LLMClient
from src.parser import parse_screenplay_dir
from src.prefilter import prefilter_candidates
from src.reliability import (
    append_jsonl,
    ensure_run_dirs,
    load_checkpoint_ids,
    new_run_id,
    redact_secrets,
)
from src.schemas import RunMetadata, ScoredCandidate, ShowProfile
from src.scoring import score_candidate_pair
from src.show_profile import build_show_profile
from src.validation_ranking import apply_episode_cap, rank_validated, to_final_promo, validate_candidate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("kuku.run")

ProgressCallback = Callable[[str, str, dict], None]


def _emit(progress: ProgressCallback | None, stage: str, status: str, **info) -> None:
    """Report a real stage transition. Callback failures never change pipeline results."""
    if progress is None:
        return
    try:
        progress(stage, status, info)
    except Exception:  # noqa: BLE001
        logger.exception("Progress callback failed for %s/%s", stage, status)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Kuku PS1 — Narrative-Aware Promo Finder"
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT_DIR / "data" / "input",
        help="Directory of screenplay .txt files",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "outputs",
        help="Root output directory",
    )
    p.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    p.add_argument(
        "--limit-episodes",
        type=int,
        default=None,
        help="Process only the first N episodes (after sorted discovery)",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume an existing run_id (loads checkpoint_scored.jsonl)",
    )
    return p.parse_args(argv)


async def run_pipeline(
    args: argparse.Namespace,
    progress: ProgressCallback | None = None,
) -> Path:
    settings = get_settings(args.config)
    input_dir: Path = args.input_dir
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}. "
            "Place screenplay .txt files there or pass --input-dir."
        )

    resumed = bool(args.resume)
    custom_run_id = getattr(args, "run_id", None)
    run_id = args.resume or custom_run_id or new_run_id()
    run_dir = ensure_run_dirs(args.output_dir, run_id)
    checkpoint_path = run_dir / settings.reliability.checkpoint_file

    meta = RunMetadata(
        run_id=run_id,
        started_at=datetime.now(timezone.utc).isoformat(),
        input_dir=str(input_dir.resolve()),
        mock_llm=False,
        limit_episodes=args.limit_episodes,
        resumed=resumed,
        model=None,
        status="running",
    )

    logger.info("Run %s starting", run_id)
    _emit(progress, "parse", "running", run_id=run_id)

    # --- Stage 1: Parse ---
    episodes, parser_report = parse_screenplay_dir(
        input_dir, limit_episodes=args.limit_episodes
    )
    if not episodes:
        raise RuntimeError("No episodes parsed from input directory.")
    logger.info(
        "Parsed %s episodes; primary events=%s",
        len(episodes),
        parser_report["block_totals"].get("primary_events"),
    )
    _emit(
        progress,
        "parse",
        "completed",
        episodes=len(episodes),
        primary_events=parser_report["block_totals"].get("primary_events"),
        files_discovered=parser_report.get("files_discovered"),
    )

    client = LLMClient()
    meta.model = client.model

    _emit(progress, "profile", "running")
    # --- Stage 2A: Show profile ---
    show_profile = await build_show_profile(
        client, episodes, run_dir=run_dir, settings=settings
    )

    _emit(
        progress,
        "profile",
        "completed",
        lexicon_terms=len(show_profile.dynamic_lexicon),
    )

    write_supporting_json(
        run_dir,
        parser_report=parser_report,
        show_profile=show_profile,
    )

    _emit(progress, "candidates", "running")
    # --- Stage 3: Candidates ---
    raw_candidates = generate_all_candidates(episodes, settings)
    logger.info("Raw candidates: %s", len(raw_candidates))
    _emit(progress, "candidates", "completed", raw_candidates=len(raw_candidates))

    _emit(progress, "prefilter", "running", raw_candidates=len(raw_candidates))
    # --- Stage 4: Prefilter + NMS ---
    filtered, cand_stats = prefilter_candidates(
        raw_candidates,
        episodes,
        dynamic_lexicon=show_profile.dynamic_lexicon,
        settings=settings,
    )
    logger.info("After NMS: %s", len(filtered))
    _emit(
        progress,
        "prefilter",
        "completed",
        raw_candidates=len(raw_candidates),
        after_nms=len(filtered),
        rejected_by_prefilter=len(raw_candidates) - len(filtered),
    )

    ep_map = {ep.header.episode_number: ep for ep in episodes}
    done_ids = load_checkpoint_ids(checkpoint_path) if resumed else set()
    if resumed:
        logger.info("Resuming; %s candidates already checkpointed", len(done_ids))

    scored: list[ScoredCandidate] = []
    # Reload checkpointed scored objects if resuming
    if resumed and checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    scored.append(ScoredCandidate.model_validate(obj))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Bad checkpoint line: %s", exc)

    pending = [c for c in filtered if c.window_id not in done_ids]
    _emit(progress, "evaluate", "running", total=len(pending), evaluated=0)
    _emit(progress, "extension", "running", checked=0, extended=0)

    # --- Stages 5–6: Score (+ optional extension) ---
    extended_count = 0
    for index, cand in enumerate(tqdm(pending, desc="Scoring candidates"), start=1):
        episode = ep_map[cand.episode_number]
        emap = {e.event_id: e for e in episode.events}
        events = [emap[i] for i in cand.event_ids if i in emap]
        try:
            result = await score_candidate_pair(
                client,
                cand,
                events,
                show_profile=show_profile,
                episode=episode,
                run_dir=run_dir,
                settings=settings,
            )
            result = await maybe_extend_candidate(
                client,
                result,
                show_profile=show_profile,
                episode=episode,
                run_dir=run_dir,
                settings=settings,
            )
        except Exception as exc:  # noqa: BLE001 — isolate per candidate
            logger.error(
                "Candidate %s failed: %s",
                cand.window_id,
                redact_secrets(str(exc)),
            )
            result = ScoredCandidate(
                candidate=cand,
                llm_failed=True,
                llm_error=str(exc),
                events=events,
            )

        append_jsonl(
            checkpoint_path,
            {
                "window_id": result.candidate.window_id,
                **result.model_dump(),
            },
        )
        if result.extension_applied:
            extended_count += 1
        scored.append(result)
        _emit(
            progress,
            "evaluate",
            "running",
            total=len(pending),
            evaluated=index,
            llm_failed=sum(1 for item in scored if item.llm_failed),
        )
        _emit(
            progress,
            "extension",
            "running",
            checked=index,
            extended=extended_count,
        )

    _emit(
        progress,
        "evaluate",
        "completed",
        total=len(pending),
        evaluated=len(pending),
        llm_failed=sum(1 for item in scored if item.llm_failed),
    )
    _emit(
        progress,
        "extension",
        "completed",
        checked=len(pending),
        extended=extended_count,
    )

    _emit(progress, "validate", "running", candidates=len(scored))
    # --- Stage 7: Validate + rank ---
    validated_scored: list[ScoredCandidate] = []
    for s in scored:
        ep = ep_map.get(s.candidate.episode_number)
        if not ep:
            s.validation_passed = False
            s.validation_failures = ["Episode missing"]
            validated_scored.append(s)
            continue
        validated_scored.append(validate_candidate(s, ep, settings))

    provisional = rank_validated(validated_scored, settings)
    logger.info("Ranked shortlist before episode cap: %s", len(provisional))
    _emit(
        progress,
        "validate",
        "completed",
        scored=len(validated_scored),
        validation_rejects=sum(1 for item in validated_scored if item.validation_passed is False),
        provisional_pool=len(provisional),
    )

    selected = apply_episode_cap(
        provisional,
        max_per_episode=settings.app.max_promos_per_episode,
        limit=settings.app.top_n_promos,
    )
    promos = [to_final_promo(i + 1, v) for i, v in enumerate(selected)]

    _emit(progress, "export", "running")
    # --- Stage 9: Export ---
    meta.finished_at = datetime.now(timezone.utc).isoformat()
    meta.status = "completed"

    export_promos_json(run_dir / "promos.json", promos, meta)
    export_promos_csv(run_dir / "promos.csv", promos)
    export_all_scored_csv(run_dir / "all_scored_candidates.csv", validated_scored)

    llm_failures = [s for s in validated_scored if s.llm_failed]
    val_rejects = [s for s in validated_scored if s.validation_passed is False]
    export_failures_md(
        run_dir / "failures.md",
        validation_rejects=val_rejects,
        llm_failures=llm_failures,
    )

    quality = build_quality_report(
        parser_report=parser_report,
        candidate_stats=cand_stats,
        scored=validated_scored,
        promos=promos,
        meta=meta,
    )
    export_quality_report(run_dir / "quality_report.json", quality)
    export_run_log(
        run_dir / "run_log.json",
        meta,
        {
            "candidate_stats": cand_stats,
            "final_promo_count": len(promos),
            "llm_call_count": client.call_count,
            "llm_fail_count": client.fail_count,
        },
    )

    logger.info(
        "Done. Final promos=%s. Outputs at %s",
        len(promos),
        run_dir,
    )
    _emit(progress, "export", "completed", final_promos=len(promos), run_dir=str(run_dir))
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_dir = asyncio.run(run_pipeline(args))
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline failed: %s", redact_secrets(str(exc)))
        return 1
    print(f"Run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
