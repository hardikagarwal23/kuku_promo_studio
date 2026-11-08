"""Stage 4 — recall-first pre-filter and temporal NMS (IoS)."""

from __future__ import annotations

import re
from collections import defaultdict

from src.config import AppSettings, get_settings
from src.schemas import CandidateWindow, ParsedEpisode, ParsedEvent

HIGH_INTENSITY_TAGS = {
    "intense",
    "demanding",
    "furious",
    "angry",
    "shouting",
    "screaming",
    "desperate",
    "terrified",
    "shocked",
    "enraged",
    "cold",
    "commanding",
    "blunt",
    "unforgiving",
    "heartbroken",
    "trembling",
    "pleading",
}

ABRUPT_MARKERS = re.compile(
    r"\b(freeze|black|gasp|shock|abrupt|slam|cliffhanger|silence|cut to black|"
    r"episode end|ominous)\b",
    re.IGNORECASE,
)
RESOLUTION_MARKERS = re.compile(
    r"\b(everything is fine|goodnight|rest well|smile warmly|happily ever|"
    r"reconcile|peacefully)\b",
    re.IGNORECASE,
)
RELATIONSHIP_NOUNS = re.compile(
    r"\b(mother|father|sister|brother|wife|husband|daughter|son|duke|duchess|"
    r"fiancé|fiance|maid|ring|poison|inheritance|wedding|altar|betrayer|"
    r"secret|truth)\b",
    re.IGNORECASE,
)
PRONOUN_OPEN = re.compile(r"\b(he|she|they|this|that|it)\b", re.IGNORECASE)
INTERRUPTION = re.compile(r"(\.\.\.|—|--|cutoff|interrupted)", re.IGNORECASE)


def intersection_over_shorter(
    a_start: float, a_end: float, b_start: float, b_end: float
) -> float:
    """IoS = overlap / min(duration_a, duration_b)."""
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    dur_a = a_end - a_start
    dur_b = b_end - b_start
    shorter = min(dur_a, dur_b)
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def _event_map(episode: ParsedEpisode) -> dict[int, ParsedEvent]:
    return {e.event_id: e for e in episode.events}


def _text_in_zone(
    events: list[ParsedEvent], clip_start: float, zone_end: float
) -> str:
    parts: list[str] = []
    for ev in events:
        if ev.start_sec < zone_end and ev.end_sec > clip_start:
            if ev.start_sec <= clip_start + (zone_end - clip_start):
                parts.append(ev.text or "")
                parts.extend(ev.delivery_tags)
                for cam in ev.camera_context:
                    parts.append(cam.get("description", ""))
    return " ".join(parts)


def score_candidate(
    candidate: CandidateWindow,
    episode: ParsedEpisode,
    *,
    dynamic_lexicon: list[str] | None = None,
    settings: AppSettings | None = None,
) -> CandidateWindow:
    """Attach weak-signal prefilter score. Never hard-gates on keywords."""
    settings = settings or get_settings()
    weights = settings.prefilter.weights
    terms = [t.lower() for t in settings.prefilter.universal_conflict_terms]
    lexicon = [t.lower() for t in (dynamic_lexicon or [])]
    if not settings.prefilter.enable_dynamic_lexicon:
        lexicon = []

    emap = _event_map(episode)
    events = [emap[i] for i in candidate.event_ids if i in emap]
    signals: dict[str, float] = {}
    score = 0.0

    hook_end = candidate.start_sec + settings.windows.hook_zone_sec
    hook_events = [
        e
        for e in events
        if e.start_sec < hook_end and e.end_sec > candidate.start_sec
    ]
    hook_text = " ".join(e.text for e in hook_events).lower()
    hook_tags = [t.lower() for e in hook_events for t in e.delivery_tags]

    if any(e.block_type in {"DIALOGUE", "ACTION"} for e in hook_events):
        signals["hook_dialogue_or_action"] = weights.hook_dialogue_or_action
        score += weights.hook_dialogue_or_action

    if any(term in hook_text for term in terms if term):
        signals["hook_universal_term"] = weights.hook_universal_term
        score += weights.hook_universal_term

    if any(term in hook_text for term in lexicon if term):
        signals["hook_dynamic_lexicon"] = weights.hook_dynamic_lexicon
        score += weights.hook_dynamic_lexicon

    if any(tag in HIGH_INTENSITY_TAGS for tag in hook_tags):
        signals["hook_intensity_tag"] = weights.hook_intensity_tag
        score += weights.hook_intensity_tag

    for e in hook_events:
        if e.block_type == "DIALOGUE" and e.text.strip().endswith(("!", "?")):
            # Prefer short punchy openings
            if len(e.text.strip()) <= 80:
                signals["hook_punchy_dialogue"] = weights.hook_punchy_dialogue
                score += weights.hook_punchy_dialogue
                break

    if INTERRUPTION.search(hook_text):
        signals["hook_interruption"] = weights.hook_interruption
        score += weights.hook_interruption

    # Cliffhanger zone — final one or two primary events
    final_n = settings.windows.max_cliffhanger_events
    final_events = events[-final_n:] if events else []
    final_text = " ".join(e.text for e in final_events)
    final_joined = final_text.lower()

    if ABRUPT_MARKERS.search(final_text) or any(
        ABRUPT_MARKERS.search(cam.get("description", ""))
        for e in final_events
        for cam in e.camera_context
    ):
        signals["cliff_abrupt_marker"] = weights.cliff_abrupt_marker
        score += weights.cliff_abrupt_marker

    if final_events:
        last = final_events[-1]
        if last.block_type == "DIALOGUE" and (
            last.text.strip().endswith("?")
            or INTERRUPTION.search(last.text)
            or last.text.strip().endswith("...")
        ):
            signals["cliff_unanswered_question"] = weights.cliff_unanswered_question
            score += weights.cliff_unanswered_question

    if any(term in final_joined for term in lexicon if term):
        signals["cliff_dynamic_lexicon"] = weights.cliff_dynamic_lexicon
        score += weights.cliff_dynamic_lexicon

    if RESOLUTION_MARKERS.search(final_text):
        signals["cliff_resolution_penalty"] = weights.cliff_resolution_penalty
        score += weights.cliff_resolution_penalty

    # Grounding
    full_text = " ".join(e.text for e in events)
    rel_hits = RELATIONSHIP_NOUNS.findall(full_text)
    if len(set(t.lower() for t in rel_hits)) >= 2:
        signals["ground_relationship_nouns"] = weights.ground_relationship_nouns
        score += weights.ground_relationship_nouns

    speakers = {e.speaker_raw for e in events if e.speaker_raw}
    if len(speakers) >= 2:
        signals["ground_named_speakers"] = weights.ground_named_speakers
        score += weights.ground_named_speakers

    opening = hook_text
    if PRONOUN_OPEN.search(opening) and len(speakers) < 1 and not rel_hits:
        signals["ground_ungrounded_pronoun_penalty"] = (
            weights.ground_ungrounded_pronoun_penalty
        )
        score += weights.ground_ungrounded_pronoun_penalty

    candidate.prefilter_score = score
    candidate.prefilter_signals = signals
    return candidate


def temporal_nms(
    candidates: list[CandidateWindow],
    *,
    threshold: float = 0.40,
    max_keep: int = 5,
) -> list[CandidateWindow]:
    """Greedy NMS using intersection-over-shorter-duration."""
    ordered = sorted(
        candidates,
        key=lambda c: (
            -c.prefilter_score,
            c.start_sec,
            c.end_sec,
            c.window_id,
        ),
    )
    kept: list[CandidateWindow] = []
    for cand in ordered:
        if len(kept) >= max_keep:
            break
        suppressed = False
        for other in kept:
            ios = intersection_over_shorter(
                cand.start_sec, cand.end_sec, other.start_sec, other.end_sec
            )
            if ios >= threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(cand)
    # Stable output order by start time for readability
    return sorted(kept, key=lambda c: (c.start_sec, -c.prefilter_score, c.window_id))


def prefilter_candidates(
    candidates: list[CandidateWindow],
    episodes: list[ParsedEpisode],
    *,
    dynamic_lexicon: list[str] | None = None,
    settings: AppSettings | None = None,
) -> tuple[list[CandidateWindow], dict]:
    """Score and NMS-filter candidates; return survivors + stats."""
    settings = settings or get_settings()
    ep_map = {ep.header.episode_number: ep for ep in episodes}
    scored: list[CandidateWindow] = []
    for cand in candidates:
        ep = ep_map.get(cand.episode_number)
        if not ep:
            continue
        scored.append(
            score_candidate(
                cand, ep, dynamic_lexicon=dynamic_lexicon, settings=settings
            )
        )

    by_ep: dict[int, list[CandidateWindow]] = defaultdict(list)
    for c in scored:
        by_ep[c.episode_number].append(c)

    kept: list[CandidateWindow] = []
    per_episode_stats: list[dict] = []
    for ep_num, group in sorted(by_ep.items()):
        survivors = temporal_nms(
            group,
            threshold=settings.prefilter.nms_threshold,
            max_keep=settings.app.max_candidates_per_episode,
        )
        kept.extend(survivors)
        per_episode_stats.append(
            {
                "episode_number": ep_num,
                "raw_candidates": len(group),
                "after_nms": len(survivors),
            }
        )

    stats = {
        "raw_candidates": len(candidates),
        "scored_candidates": len(scored),
        "after_nms": len(kept),
        "max_per_episode": settings.app.max_candidates_per_episode,
        "nms_threshold": settings.prefilter.nms_threshold,
        "nms_metric": "intersection_over_shorter",
        "per_episode": per_episode_stats,
    }
    return kept, stats
