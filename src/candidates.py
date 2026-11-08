"""Stage 3 — continuous 20–60s candidate window generation."""

from __future__ import annotations

from src.config import AppSettings, get_settings
from src.schemas import CandidateWindow, ParsedEpisode, ParsedEvent

START_TYPES = {"DIALOGUE", "ACTION"}
END_TYPES = {"DIALOGUE", "ACTION", "SOUND_FX"}


def _events_overlapping_range(
    events: list[ParsedEvent], start_sec: float, end_sec: float
) -> list[ParsedEvent]:
    out: list[ParsedEvent] = []
    for ev in events:
        if ev.end_sec > start_sec and ev.start_sec < end_sec:
            out.append(ev)
        elif ev.start_sec == start_sec or ev.end_sec == end_sec:
            out.append(ev)
    # Deduplicate by event_id while preserving order
    seen: set[int] = set()
    unique: list[ParsedEvent] = []
    for ev in out:
        if ev.event_id not in seen:
            seen.add(ev.event_id)
            unique.append(ev)
    return unique


def generate_candidates_for_episode(
    episode: ParsedEpisode,
    settings: AppSettings | None = None,
) -> list[CandidateWindow]:
    """Generate all valid continuous event-boundary windows for one episode."""
    settings = settings or get_settings()
    min_d = settings.windows.min_duration_sec
    max_d = settings.windows.max_duration_sec
    events = episode.events
    if not events:
        return []

    ep_num = episode.header.episode_number
    windows: list[CandidateWindow] = []
    counter = 0

    for i, start_ev in enumerate(events):
        if start_ev.block_type not in START_TYPES:
            continue
        for j in range(i, len(events)):
            end_ev = events[j]
            if end_ev.block_type not in END_TYPES:
                continue
            start_sec = start_ev.start_sec
            end_sec = end_ev.end_sec
            duration = end_sec - start_sec
            if duration < min_d:
                continue
            if duration > max_d:
                # Later end events only grow duration for fixed start
                break
            included = _events_overlapping_range(events, start_sec, end_sec)
            if not included:
                continue
            counter += 1
            windows.append(
                CandidateWindow(
                    window_id=f"ep{ep_num:03d}_w{counter:04d}",
                    episode_number=ep_num,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    start_timestamp=start_ev.start_timestamp,
                    end_timestamp=end_ev.end_timestamp,
                    duration_sec=round(duration, 6),
                    start_event_id=start_ev.event_id,
                    end_event_id=end_ev.event_id,
                    event_ids=[e.event_id for e in included],
                )
            )
    return windows


def generate_all_candidates(
    episodes: list[ParsedEpisode],
    settings: AppSettings | None = None,
) -> list[CandidateWindow]:
    settings = settings or get_settings()
    all_windows: list[CandidateWindow] = []
    for ep in episodes:
        all_windows.extend(generate_candidates_for_episode(ep, settings))
    return all_windows
