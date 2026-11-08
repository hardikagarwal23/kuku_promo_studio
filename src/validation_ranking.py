"""Stage 7 — hard validation and deterministic ranking."""

from __future__ import annotations

from src.config import AppSettings, get_settings
from src.schemas import (
    FinalPromo,
    ParsedEpisode,
    PromoSegment,
    ScoredCandidate,
    ValidatedCandidate,
    normalize_quote,
)
from src.scoring import composite_score


def quote_matches_event(quote: str, event_text: str) -> bool:
    q = normalize_quote(quote)
    t = normalize_quote(event_text)
    if not q:
        return False
    return q in t


def validate_candidate(
    scored: ScoredCandidate,
    episode: ParsedEpisode,
    settings: AppSettings | None = None,
) -> ScoredCandidate:
    settings = settings or get_settings()
    failures: list[str] = []
    cand = scored.candidate
    emap = {e.event_id: e for e in episode.events}

    # Duration
    if not (
        settings.windows.min_duration_sec
        <= cand.duration_sec
        <= settings.windows.max_duration_sec
    ):
        failures.append(
            f"Duration {cand.duration_sec} outside "
            f"{settings.windows.min_duration_sec}-{settings.windows.max_duration_sec}"
        )

    # Event IDs exist and belong to candidate
    if not cand.event_ids:
        failures.append("Candidate contains no primary events")
    for eid in cand.event_ids:
        if eid not in emap:
            failures.append(f"Unknown event_id {eid} in episode {cand.episode_number}")
    if cand.start_event_id not in cand.event_ids or cand.end_event_id not in cand.event_ids:
        failures.append("Start/end event IDs not in candidate event_ids")

    # Timestamp consistency with selected events
    if cand.start_event_id in emap:
        if abs(emap[cand.start_event_id].start_sec - cand.start_sec) > 1e-6:
            failures.append("Start timestamp inconsistent with start event")
    else:
        failures.append("Missing start event")
    if cand.end_event_id in emap:
        if abs(emap[cand.end_event_id].end_sec - cand.end_sec) > 1e-6:
            failures.append("End timestamp inconsistent with end event")
    else:
        failures.append("Missing end event")

    if scored.llm_failed or not scored.standalone or not scored.contextual:
        failures.append("LLM scoring incomplete/failed")
        scored.validation_passed = False
        scored.validation_failures = failures
        return scored

    hook = scored.contextual.hook
    cliff = scored.contextual.cliffhanger

    if hook.event_id not in cand.event_ids or hook.event_id not in emap:
        failures.append("Hook evidence event_id not in clip")
    else:
        hook_ev = emap[hook.event_id]
        # Hook evidence must begin within the first hook_zone_sec of the clip.
        if hook_ev.start_sec > cand.start_sec + settings.windows.hook_zone_sec + 1e-9:
            failures.append(
                f"Hook event starts after first {settings.windows.hook_zone_sec}s"
            )
        if not quote_matches_event(hook.quote, hook_ev.source_text or hook_ev.text):
            failures.append("Hook quote does not match source event text")

    # Cliffhanger among final N primary events and near ending when possible
    included = [emap[i] for i in cand.event_ids if i in emap]
    included_sorted = sorted(
        included, key=lambda e: (e.start_sec, e.end_sec, e.event_id)
    )
    final_n = settings.windows.max_cliffhanger_events
    final_ids = {e.event_id for e in included_sorted[-final_n:]} if included_sorted else set()
    if cliff.event_id not in cand.event_ids or cliff.event_id not in emap:
        failures.append("Cliffhanger evidence event_id not in clip")
    else:
        cliff_ev = emap[cliff.event_id]
        # Hard MVP rule: among final N primary events.
        # Prefer also within final zone seconds when event lengths allow.
        zone = settings.windows.cliffhanger_final_zone_sec
        in_final_events = cliff.event_id in final_ids
        in_final_zone = cliff_ev.start_sec >= cand.end_sec - zone
        if not in_final_events:
            failures.append(
                f"Cliffhanger event not among final {final_n} primary events "
                f"(final-zone target={zone}s; in_final_zone={in_final_zone})"
            )
        if not quote_matches_event(cliff.quote, cliff_ev.source_text or cliff_ev.text):
            failures.append("Cliffhanger quote does not match source event text")

    # Ordering
    if included_sorted and included_sorted[0].start_sec > included_sorted[-1].end_sec:
        failures.append("Invalid event ordering")

    scored.validation_failures = failures
    scored.validation_passed = len(failures) == 0
    if scored.validation_passed:
        scored.composite_score = composite_score(
            hook.score,
            cliff.score,
            scored.standalone.standalone_score,
            settings,
        )
    return scored


def rank_validated(
    scored_list: list[ScoredCandidate],
    settings: AppSettings | None = None,
) -> list[ValidatedCandidate]:
    """Sort by composite then component scores and keep the top-N shortlist."""
    settings = settings or get_settings()
    valid: list[ValidatedCandidate] = []
    for s in scored_list:
        if not s.validation_passed or s.composite_score is None:
            continue
        assert s.contextual and s.standalone
        valid.append(
            ValidatedCandidate(
                scored=s,
                composite_score=s.composite_score,
                hook_score=s.contextual.hook.score,
                cliffhanger_score=s.contextual.cliffhanger.score,
                standalone_score=s.standalone.standalone_score,
            )
        )

    valid.sort(
        key=lambda v: (
            -v.composite_score,
            -v.hook_score,
            -v.cliffhanger_score,
            -v.standalone_score,
            v.scored.candidate.window_id,
        )
    )

    # Build provisional pool with max-2 per episode
    per_ep: dict[int, int] = {}
    provisional: list[ValidatedCandidate] = []
    for v in valid:
        ep = v.scored.candidate.episode_number
        if per_ep.get(ep, 0) >= settings.app.max_promos_per_episode:
            continue
        provisional.append(v)
        per_ep[ep] = per_ep.get(ep, 0) + 1
        if len(provisional) >= settings.app.top_n_promos:
            break
    return provisional


def to_final_promo(rank: int, validated: ValidatedCandidate) -> FinalPromo:
    scored = validated.scored
    candidate = scored.candidate
    assert scored.contextual and scored.standalone
    return FinalPromo(
        rank=rank,
        promo_id=f"promo_{rank:02d}",
        episode_number=candidate.episode_number,
        segments=[
            PromoSegment(
                episode_number=candidate.episode_number,
                start_timestamp=candidate.start_timestamp,
                end_timestamp=candidate.end_timestamp,
                start_sec=candidate.start_sec,
                end_sec=candidate.end_sec,
                duration_sec=candidate.duration_sec,
                start_event_id=candidate.start_event_id,
                end_event_id=candidate.end_event_id,
                event_ids=candidate.event_ids,
            )
        ],
        duration_sec=candidate.duration_sec,
        hook=scored.contextual.hook.model_dump(),
        cliffhanger=scored.contextual.cliffhanger.model_dump(),
        why_it_works=scored.contextual.why_it_works,
        scores={
            "hook": float(validated.hook_score),
            "cliffhanger": float(validated.cliffhanger_score),
            "standalone": float(validated.standalone_score),
        },
        composite_score=validated.composite_score,
        human_review_status="PENDING",
        window_id=candidate.window_id,
        archetypes=scored.contextual.archetypes,
        flags=scored.contextual.flags,
    )


def apply_episode_cap(
    items: list[ValidatedCandidate],
    *,
    max_per_episode: int,
    limit: int,
) -> list[ValidatedCandidate]:
    per_ep: dict[int, int] = {}
    out: list[ValidatedCandidate] = []
    for v in items:
        ep = v.scored.candidate.episode_number
        if per_ep.get(ep, 0) >= max_per_episode:
            continue
        out.append(v)
        per_ep[ep] = per_ep.get(ep, 0) + 1
        if len(out) >= limit:
            break
    return out
