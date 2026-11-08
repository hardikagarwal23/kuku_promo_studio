"""Stage 1 — deterministic screenplay parser."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.schemas import (
    CameraBlock,
    EpisodeHeader,
    ParsedEpisode,
    ParsedEvent,
    ParseIssue,
)

# --- compiled patterns ---
EPISODE_NUM_RE = re.compile(
    r"EPISODE\s+NUMBER\s*:\s*(?:Episode\s+)?(\d+)", re.IGNORECASE
)
EPISODE_LOGLINE_RE = re.compile(r"EPISODE\s+LOGLINE\s*:\s*(.*)", re.IGNORECASE)
CONTINUITY_RE = re.compile(r"CONTINUITY\s+NOTES\s*:\s*(.*)", re.IGNORECASE)
SCRIPT_START_RE = re.compile(r"^---\s*SCRIPT\s+START\s*---\s*$", re.IGNORECASE)
SCENE_NUM_RE = re.compile(r"^SCENE\s+(\d+)\s*$", re.IGNORECASE)
SCENE_HEADING_RE = re.compile(r"^Scene\s+heading\s*:\s*(.*)$", re.IGNORECASE)
SCENE_START_RE = re.compile(r"^Scene\s+start\s*:\s*(.*)$", re.IGNORECASE)
SCENE_END_RE = re.compile(r"^Scene\s+end\s*:\s*(.*)$", re.IGNORECASE)

# **LABEL:** or **SPEAKER (tags):**
BLOCK_HEADER_RE = re.compile(
    r"^\*\*\s*(.+?)\s*\*\*\s*:?\s*$", re.IGNORECASE
)
TIMESTAMP_LINE_RE = re.compile(
    r"^Start:\s*\(?\s*([0-9]{1,2}:[0-9]{2}(?:\.[0-9]+)?)\s*\)?"
    r"\s*\|\s*End:\s*\(?\s*([0-9]{1,2}:[0-9]{2}(?:\.[0-9]+)?)\s*\)?\s*$",
    re.IGNORECASE,
)
TIMESTAMP_TOKEN_RE = re.compile(r"^(\d{1,2}):(\d{2})(?:\.(\d+))?$")
FILENAME_EPISODE_RE = re.compile(r"(?:episode[_\-\s]*)(\d+)", re.IGNORECASE)

CAMERA_LABELS = {"CAMERA"}
ACTION_LABELS = {"ACTION"}
SOUND_LABELS = {"SOUND FX", "SOUNDFX", "SOUND_FX", "SFX"}

GENERIC_ROLE_EXCLUDES = {
    "LORD",
    "MAID",
    "GUARD",
    "JUDGE",
    "CONSTABLE",
    "DOCTOR",
    "SERVANT",
}


def timestamp_to_seconds(ts: str) -> float:
    """Convert `MM:SS.xx` or `M:SS.xx` to float seconds."""
    raw = (ts or "").strip().strip("()")
    match = TIMESTAMP_TOKEN_RE.match(raw)
    if not match:
        raise ValueError(f"Invalid timestamp: {ts!r}")
    minutes = int(match.group(1))
    seconds = int(match.group(2))
    frac = match.group(3) or "0"
    # Preserve decimal precision from fractional digits (e.g. .40 -> 0.40)
    frac_val = int(frac) / (10 ** len(frac))
    return minutes * 60 + seconds + frac_val


def seconds_to_timestamp(sec: float) -> str:
    """Format seconds as MM:SS.xx matching screenplay style."""
    if sec < 0:
        raise ValueError("Negative seconds")
    minutes = int(sec // 60)
    rem = sec - minutes * 60
    whole = int(rem)
    frac = int(round((rem - whole) * 100))
    if frac == 100:
        whole += 1
        frac = 0
    if whole == 60:
        minutes += 1
        whole = 0
    return f"{minutes:02d}:{whole:02d}.{frac:02d}"


def extract_episode_number_from_filename(name: str) -> int | None:
    match = FILENAME_EPISODE_RE.search(name)
    if match:
        return int(match.group(1))
    return None


def discover_screenplay_files(input_dir: Path) -> list[Path]:
    """Discover `.txt` files and sort by episode number."""
    files = [
        p
        for p in Path(input_dir).rglob("*")
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".markdown"}
    ]
    files = [p for p in files if p.name.lower() != ".gitkeep"]

    def sort_key(path: Path) -> tuple[int, str]:
        num = extract_episode_number_from_filename(path.name)
        return (num if num is not None else 10**9, path.name.lower())

    return sorted(files, key=sort_key)


def _classify_block_header(header_inner: str) -> tuple[str, str | None, list[str]]:
    """
    Return (block_type, speaker_raw|None, delivery_tags).
    block_type in CAMERA|ACTION|SOUND_FX|DIALOGUE
    """
    # Headers may arrive as `ACTION:` or `GERALD (O.S.):` when the colon
    # was captured inside the **...** group.
    inner = header_inner.strip().rstrip(":").strip()
    # Split speaker and parenthetical tags: SPEAKER (Cold, Commanding)
    speaker = inner
    tags: list[str] = []
    paren = re.match(r"^(.*?)\s*\((.+)\)\s*$", inner)
    if paren:
        speaker = paren.group(1).strip().rstrip(":").strip()
        raw_tags = paren.group(2).strip()
        # Conservative: split on commas; keep O.S./V.O./Off-screen as tags
        tags = [t.strip() for t in re.split(r"[,;/]", raw_tags) if t.strip()]

    upper = speaker.upper().strip()
    if upper in CAMERA_LABELS:
        return "CAMERA", None, tags
    if upper in ACTION_LABELS:
        return "ACTION", None, tags
    if upper in SOUND_LABELS or upper.replace(" ", "") in {"SOUNDFX", "SFX"}:
        return "SOUND_FX", None, tags
    return "DIALOGUE", speaker.strip(), tags


def _intervals_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return max(a0, b0) < min(a1, b1) or (
        max(a0, b0) <= min(a1, b1) and (a1 > a0 and b1 > b0)
    )


def parse_episode_text(
    text: str,
    *,
    source_filename: str = "",
    fallback_episode_number: int | None = None,
) -> ParsedEpisode:
    """Parse one screenplay file into structured events."""
    issues: list[ParseIssue] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    episode_number = fallback_episode_number
    logline = ""
    continuity = ""
    script_idx = 0

    for i, line in enumerate(lines):
        if SCRIPT_START_RE.match(line.strip()):
            script_idx = i + 1
            break
        m = EPISODE_NUM_RE.search(line)
        if m:
            episode_number = int(m.group(1))
            continue
        m = EPISODE_LOGLINE_RE.match(line.strip())
        if m:
            logline = m.group(1).strip()
            continue
        m = CONTINUITY_RE.match(line.strip())
        if m:
            continuity = m.group(1).strip()
            continue

    if episode_number is None:
        issues.append(
            ParseIssue(
                source_filename=source_filename,
                severity="error",
                message="Could not determine episode number; defaulting to 0",
            )
        )
        episode_number = 0

    header = EpisodeHeader(
        episode_number=episode_number,
        episode_logline=logline,
        continuity_notes=continuity,
        source_filename=source_filename,
    )

    scenes: list[dict[str, Any]] = []
    current_scene: int | None = None
    cameras: list[CameraBlock] = []
    primary_raw: list[dict[str, Any]] = []
    block_counts: dict[str, int] = {
        "CAMERA": 0,
        "ACTION": 0,
        "SOUND_FX": 0,
        "DIALOGUE": 0,
        "UNKNOWN": 0,
    }

    i = script_idx
    source_order = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        scene_m = SCENE_NUM_RE.match(stripped)
        if scene_m:
            current_scene = int(scene_m.group(1))
            scenes.append(
                {
                    "scene_number": current_scene,
                    "heading": "",
                    "start_timestamp": "",
                    "end_timestamp": "",
                }
            )
            i += 1
            continue

        heading_m = SCENE_HEADING_RE.match(stripped)
        if heading_m and scenes:
            scenes[-1]["heading"] = heading_m.group(1).strip()
            i += 1
            continue

        ss = SCENE_START_RE.match(stripped)
        if ss and scenes:
            scenes[-1]["start_timestamp"] = ss.group(1).strip()
            i += 1
            continue

        se = SCENE_END_RE.match(stripped)
        if se and scenes:
            scenes[-1]["end_timestamp"] = se.group(1).strip()
            i += 1
            continue

        if stripped.lower().startswith("actions and dialogue"):
            i += 1
            continue

        header_m = BLOCK_HEADER_RE.match(stripped)
        if not header_m:
            # Unrecognised non-empty line inside script — record and continue
            if stripped and not stripped.startswith("---"):
                issues.append(
                    ParseIssue(
                        episode_number=episode_number,
                        source_filename=source_filename,
                        line_number=i + 1,
                        severity="warning",
                        message="Unrecognised line outside a labelled block",
                        raw_snippet=stripped[:200],
                    )
                )
            i += 1
            continue

        block_type, speaker_raw, delivery_tags = _classify_block_header(
            header_m.group(1)
        )
        i += 1

        # Expect timestamp line next (skip blanks)
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            issues.append(
                ParseIssue(
                    episode_number=episode_number,
                    source_filename=source_filename,
                    line_number=i,
                    severity="error",
                    message=f"Block '{block_type}' missing timestamp line",
                )
            )
            break

        ts_line = lines[i].strip()
        ts_m = TIMESTAMP_LINE_RE.match(ts_line)
        if not ts_m:
            issues.append(
                ParseIssue(
                    episode_number=episode_number,
                    source_filename=source_filename,
                    line_number=i + 1,
                    severity="error",
                    message=f"Malformed timestamp for {block_type} block",
                    raw_snippet=ts_line[:200],
                )
            )
            block_counts["UNKNOWN"] += 1
            i += 1
            continue

        start_ts = ts_m.group(1)
        end_ts = ts_m.group(2)
        i += 1

        try:
            start_sec = timestamp_to_seconds(start_ts)
            end_sec = timestamp_to_seconds(end_ts)
        except ValueError as exc:
            issues.append(
                ParseIssue(
                    episode_number=episode_number,
                    source_filename=source_filename,
                    line_number=i,
                    severity="error",
                    message=str(exc),
                    raw_snippet=ts_line[:200],
                )
            )
            block_counts["UNKNOWN"] += 1
            continue

        if start_sec < 0 or end_sec < 0 or start_sec > end_sec:
            issues.append(
                ParseIssue(
                    episode_number=episode_number,
                    source_filename=source_filename,
                    line_number=i,
                    severity="error",
                    message=f"Invalid time range {start_ts}–{end_ts}",
                )
            )

        # Collect body until next header / scene / blank-run ending at next header
        body_lines: list[str] = []
        while i < len(lines):
            nxt = lines[i]
            nxt_s = nxt.strip()
            if not nxt_s:
                # Peek ahead: blank line may separate paragraphs within a block
                # Stop only when next non-empty is a new header/scene/script end
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j >= len(lines):
                    break
                peek = lines[j].strip()
                if (
                    BLOCK_HEADER_RE.match(peek)
                    or SCENE_NUM_RE.match(peek)
                    or SCENE_HEADING_RE.match(peek)
                    or SCRIPT_START_RE.match(peek)
                ):
                    break
                body_lines.append("")
                i += 1
                continue
            if (
                BLOCK_HEADER_RE.match(nxt_s)
                or SCENE_NUM_RE.match(nxt_s)
                or SCENE_HEADING_RE.match(nxt_s)
                or SCRIPT_START_RE.match(nxt_s)
            ):
                break
            body_lines.append(nxt_s)
            i += 1

        text_body = "\n".join(body_lines).strip()
        source_order += 1
        block_counts[block_type] = block_counts.get(block_type, 0) + 1

        if block_type == "CAMERA":
            cameras.append(
                CameraBlock(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    start_timestamp=start_ts,
                    end_timestamp=end_ts,
                    description=text_body,
                    source_order=source_order,
                    scene_number=current_scene,
                )
            )
            continue

        if not text_body and block_type in {"DIALOGUE", "ACTION"}:
            issues.append(
                ParseIssue(
                    episode_number=episode_number,
                    source_filename=source_filename,
                    line_number=i,
                    severity="warning",
                    message=f"Empty {block_type} block text",
                )
            )

        primary_raw.append(
            {
                "block_type": block_type,
                "speaker_raw": speaker_raw,
                "delivery_tags": delivery_tags,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                "text": text_body,
                "source_text": text_body,
                "source_order": source_order,
                "scene_number": current_scene,
            }
        )

    # Sort primary events: start, end, source_order
    primary_raw.sort(
        key=lambda e: (e["start_sec"], e["end_sec"], e["source_order"])
    )

    events: list[ParsedEvent] = []
    for idx, raw in enumerate(primary_raw, start=1):
        cam_ctx: list[dict[str, Any]] = []
        for cam in cameras:
            if _intervals_overlap(
                raw["start_sec"], raw["end_sec"], cam.start_sec, cam.end_sec
            ):
                cam_ctx.append(
                    {
                        "start_sec": cam.start_sec,
                        "end_sec": cam.end_sec,
                        "start_timestamp": cam.start_timestamp,
                        "end_timestamp": cam.end_timestamp,
                        "description": cam.description,
                    }
                )
        events.append(
            ParsedEvent(
                episode_number=episode_number,
                scene_number=raw["scene_number"],
                event_id=idx,
                block_type=raw["block_type"],
                speaker_raw=raw["speaker_raw"],
                speaker_canonical=None,
                delivery_tags=raw["delivery_tags"],
                start_sec=raw["start_sec"],
                end_sec=raw["end_sec"],
                start_timestamp=raw["start_timestamp"],
                end_timestamp=raw["end_timestamp"],
                text=raw["text"],
                camera_context=cam_ctx,
                source_order=raw["source_order"],
                source_text=raw["source_text"],
            )
        )

    attached_orders = set()
    for ev in events:
        for cam in cameras:
            if _intervals_overlap(
                ev.start_sec, ev.end_sec, cam.start_sec, cam.end_sec
            ):
                attached_orders.add(cam.source_order)
    unattached = [c for c in cameras if c.source_order not in attached_orders]

    return ParsedEpisode(
        header=header,
        events=events,
        unattached_camera_blocks=unattached,
        scenes=scenes,
        issues=issues,
        block_counts=block_counts,
    )


def parse_screenplay_file(path: Path) -> ParsedEpisode:
    text = path.read_text(encoding="utf-8", errors="replace")
    fallback = extract_episode_number_from_filename(path.name)
    return parse_episode_text(
        text, source_filename=path.name, fallback_episode_number=fallback
    )


def parse_screenplay_dir(
    input_dir: Path,
    *,
    limit_episodes: int | None = None,
) -> tuple[list[ParsedEpisode], dict[str, Any]]:
    """Parse all episodes and return episodes + parser report."""
    files = discover_screenplay_files(input_dir)
    if limit_episodes is not None:
        files = files[:limit_episodes]

    episodes: list[ParsedEpisode] = []
    all_issues: list[dict[str, Any]] = []
    totals = {
        "CAMERA": 0,
        "ACTION": 0,
        "SOUND_FX": 0,
        "DIALOGUE": 0,
        "UNKNOWN": 0,
        "primary_events": 0,
        "unattached_cameras": 0,
    }

    for path in files:
        try:
            ep = parse_screenplay_file(path)
            episodes.append(ep)
            for k, v in ep.block_counts.items():
                totals[k] = totals.get(k, 0) + v
            totals["primary_events"] += len(ep.events)
            totals["unattached_cameras"] += len(ep.unattached_camera_blocks)
            for issue in ep.issues:
                all_issues.append(issue.model_dump())
        except Exception as exc:  # noqa: BLE001 — keep pack run alive
            all_issues.append(
                {
                    "episode_number": None,
                    "source_filename": path.name,
                    "severity": "error",
                    "message": f"Failed to parse file: {exc}",
                }
            )

    report = {
        "files_discovered": len(discover_screenplay_files(input_dir)),
        "files_parsed": len(episodes),
        "limit_episodes": limit_episodes,
        "block_totals": totals,
        "expected_reference_totals": {
            "note": "Sanity-check expectations from assignment pack, not hard asserts",
            "total_labelled_blocks_approx": 3727,
            "camera_approx": 1754,
            "action_approx": 372,
            "sound_fx_approx": 114,
            "dialogue_approx": 1487,
            "primary_events_approx": 1973,
        },
        "per_episode": [
            {
                "episode_number": ep.header.episode_number,
                "source_filename": ep.header.source_filename,
                "primary_events": len(ep.events),
                "block_counts": ep.block_counts,
                "unattached_cameras": len(ep.unattached_camera_blocks),
                "issue_count": len(ep.issues),
                "logline_present": bool(ep.header.episode_logline),
            }
            for ep in episodes
        ],
        "issues": all_issues,
    }
    return episodes, report
