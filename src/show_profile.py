"""Stage 2A — show profile and dynamic lexicon."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import AppSettings, get_settings
from src.llm_client import LLMClient
from src.parser import GENERIC_ROLE_EXCLUDES
from src.reliability import save_raw_response, validate_with_retry
from src.schemas import AliasMapping, ParsedEpisode, ShowProfile


def collect_profile_input(episodes: list[ParsedEpisode]) -> str:
    speakers: set[str] = set()
    rows: list[dict] = []
    # ONLY SEND THE FIRST 10 EPISODES TO SAVE TOKENS
    for ep in episodes[:10]:
        for ev in ep.events:
            if ev.speaker_raw:
                speakers.add(ev.speaker_raw)
        rows.append(
            {
                "episode_number": ep.header.episode_number,
                "logline": ep.header.episode_logline,
                "continuity_notes": ep.header.continuity_notes,
            }
        )
    payload = {
        "episodes": rows,
        "unique_speaker_labels": sorted(speakers),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def sanitize_alias_map(profile: ShowProfile) -> ShowProfile:
    cleaned: dict[str, AliasMapping] = {}
    for raw, mapping in profile.alias_map.items():
        upper = raw.strip().upper()
        if upper in GENERIC_ROLE_EXCLUDES and mapping.confidence < 0.99:
            continue
        if mapping.confidence < 0.7:
            continue
        cleaned[raw] = mapping
    profile.alias_map = cleaned
    # Keep lexicon size reasonable
    lex = [t.strip() for t in profile.dynamic_lexicon if t.strip()]
    profile.dynamic_lexicon = lex[:20]
    return profile


async def build_show_profile(
    client: LLMClient,
    episodes: list[ParsedEpisode],
    *,
    run_dir: Path | None = None,
    settings: AppSettings | None = None,
) -> ShowProfile:
    settings = settings or get_settings()
    payload = collect_profile_input(episodes)
    prompt = client.render_prompt("show_profile.txt", payload)

    async def _call(prompt: str) -> str:
        # Route to 120b to bypass the 21-minute lockout on 20b
        return await client.complete(prompt, stage="show_profile")

    parsed, raw, err = await validate_with_retry(
        call_fn=_call,
        call_kwargs={"prompt": prompt},
        model_cls=ShowProfile,
        retry_attempts=settings.llm.retry_attempts,
        include_validation_error=settings.llm.retry_with_validation_error,
    )
    if run_dir is not None:
        save_raw_response(
            run_dir,
            "show_profile",
            raw_text=raw,
            metadata={
                "model": client.model,
                "error": err,
                "validated": parsed.model_dump() if parsed else None,
            },
        )
    if parsed is None:
        # Safe minimal fallback — do not invent rich aliases
        return ShowProfile(
            show_brief="Show profile unavailable due to LLM failure; contextual scoring limited.",
            alias_map={},
            dynamic_lexicon=[],
        )
    return sanitize_alias_map(parsed)
