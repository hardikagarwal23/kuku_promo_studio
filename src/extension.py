"""Stage 6 — conditional contiguous start extension."""

from __future__ import annotations

from pathlib import Path

from src.config import AppSettings, get_settings
from src.llm_client import LLMClient
from src.schemas import (
    CandidateWindow,
    ParsedEpisode,
    ParsedEvent,
    ScoredCandidate,
    ShowProfile,
)
from src.scoring import composite_score, score_candidate_pair


def should_attempt_extension(
    scored: ScoredCandidate,
    episode: ParsedEpisode,
    settings: AppSettings | None = None,
) -> bool:
    settings = settings or get_settings()
    if not settings.extension.enabled:
        return False
    if scored.llm_failed or not scored.standalone or not scored.contextual:
        return False
    if not scored.standalone.context_needed:
        return False
    thr = settings.scoring.strong_score_threshold
    if scored.contextual.hook.score < thr:
        return False
    if scored.contextual.cliffhanger.score < thr:
        return False

    emap = {e.event_id: e for e in episode.events}
    start_id = scored.candidate.start_event_id
    ordered = sorted(episode.events, key=lambda e: (e.start_sec, e.end_sec, e.event_id))
    idx = next((i for i, e in enumerate(ordered) if e.event_id == start_id), None)
    if idx is None or idx == 0:
        return False

    max_prec = settings.extension.max_preceding_events
    # Check if at least one preceding event can be prepended within 60s
    for n in range(1, max_prec + 1):
        if idx - n < 0:
            break
        new_start = ordered[idx - n]
        new_duration = scored.candidate.end_sec - new_start.start_sec
        if new_duration <= settings.windows.max_duration_sec:
            return True
    return False


def build_extended_candidate(
    scored: ScoredCandidate,
    episode: ParsedEpisode,
    settings: AppSettings | None = None,
) -> CandidateWindow | None:
    settings = settings or get_settings()
    ordered = sorted(episode.events, key=lambda e: (e.start_sec, e.end_sec, e.event_id))
    start_id = scored.candidate.start_event_id
    idx = next((i for i, e in enumerate(ordered) if e.event_id == start_id), None)
    if idx is None or idx == 0:
        return None

    max_prec = settings.extension.max_preceding_events
    best: CandidateWindow | None = None
    for n in range(1, max_prec + 1):
        if idx - n < 0:
            break
        new_start = ordered[idx - n]
        duration = scored.candidate.end_sec - new_start.start_sec
        if duration < settings.windows.min_duration_sec:
            continue
        if duration > settings.windows.max_duration_sec:
            continue
        # Include overlapping events
        event_ids = [
            e.event_id
            for e in ordered
            if e.end_sec > new_start.start_sec and e.start_sec < scored.candidate.end_sec
        ]
        best = CandidateWindow(
            window_id=f"{scored.candidate.window_id}_ext{n}",
            episode_number=scored.candidate.episode_number,
            start_sec=new_start.start_sec,
            end_sec=scored.candidate.end_sec,
            start_timestamp=new_start.start_timestamp,
            end_timestamp=scored.candidate.end_timestamp,
            duration_sec=round(duration, 6),
            start_event_id=new_start.event_id,
            end_event_id=scored.candidate.end_event_id,
            event_ids=event_ids,
            prefilter_score=scored.candidate.prefilter_score,
            prefilter_signals=scored.candidate.prefilter_signals,
        )
    return best


def hook_within_first_three(
    scored: ScoredCandidate, settings: AppSettings | None = None
) -> bool:
    settings = settings or get_settings()
    if not scored.contextual:
        return False
    emap = {e.event_id: e for e in scored.events}
    hook_ev = emap.get(scored.contextual.hook.event_id)
    if not hook_ev:
        return False
    return hook_ev.start_sec <= scored.candidate.start_sec + settings.windows.hook_zone_sec


async def maybe_extend_candidate(
    client: LLMClient,
    scored: ScoredCandidate,
    *,
    show_profile: ShowProfile,
    episode: ParsedEpisode,
    run_dir: Path | None = None,
    settings: AppSettings | None = None,
) -> ScoredCandidate:
    """
    Attempt at most one contiguous start extension.
    Keep extension only if standalone improves, hook>=7, and hook still in first 3s.
    """
    settings = settings or get_settings()
    if not should_attempt_extension(scored, episode, settings):
        return scored

    extended_cand = build_extended_candidate(scored, episode, settings)
    if extended_cand is None:
        return scored

    emap = {e.event_id: e for e in episode.events}
    ext_events = [emap[i] for i in extended_cand.event_ids if i in emap]

    # Re-score once (both evaluators)
    from src.scoring import score_hook_cliffhanger, score_standalone_blind
    import asyncio

    (standalone, _, stand_err), (contextual, _, ctx_err) = await asyncio.gather(
        score_standalone_blind(
            client,
            extended_cand,
            ext_events,
            run_dir=run_dir,
            settings=settings,
            stage_name="extension_rescore",
        ),
        score_hook_cliffhanger(
            client,
            extended_cand,
            ext_events,
            show_profile=show_profile,
            episode=episode,
            run_dir=run_dir,
            settings=settings,
            stage_name="extension_rescore",
        ),
    )

    if standalone is None or contextual is None:
        return scored

    thr = settings.scoring.strong_score_threshold
    orig_stand = scored.standalone.standalone_score if scored.standalone else 0
    improved = standalone.standalone_score > orig_stand
    if settings.extension.require_standalone_improvement and not improved:
        return scored
    if contextual.hook.score < thr:
        return scored

    rescored = ScoredCandidate(
        candidate=extended_cand,
        standalone=standalone,
        contextual=contextual,
        extended=True,
        extension_applied=True,
        llm_failed=False,
        events=ext_events,
    )
    if not hook_within_first_three(rescored, settings):
        return scored

    rescored.composite_score = composite_score(
        contextual.hook.score,
        contextual.cliffhanger.score,
        standalone.standalone_score,
        settings,
    )
    return rescored
