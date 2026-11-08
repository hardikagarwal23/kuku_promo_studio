"""Pydantic models for parsed screenplays, candidates, scores, and outputs."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class BlockType(str, Enum):
    DIALOGUE = "DIALOGUE"
    ACTION = "ACTION"
    SOUND_FX = "SOUND_FX"
    CAMERA = "CAMERA"


class EpisodeHeader(BaseModel):
    episode_number: int
    episode_logline: str = ""
    continuity_notes: str = ""
    source_filename: str = ""


class CameraBlock(BaseModel):
    start_sec: float
    end_sec: float
    start_timestamp: str
    end_timestamp: str
    description: str
    source_order: int
    scene_number: int | None = None


class ParsedEvent(BaseModel):
    episode_number: int
    scene_number: int | None = None
    event_id: int
    block_type: Literal["DIALOGUE", "ACTION", "SOUND_FX"]
    speaker_raw: str | None = None
    speaker_canonical: str | None = None
    delivery_tags: list[str] = Field(default_factory=list)
    start_sec: float
    end_sec: float
    start_timestamp: str
    end_timestamp: str
    text: str
    camera_context: list[dict[str, Any]] = Field(default_factory=list)
    source_order: int
    source_text: str = ""


class ParseIssue(BaseModel):
    episode_number: int | None = None
    source_filename: str = ""
    line_number: int | None = None
    severity: Literal["warning", "error"] = "warning"
    message: str
    raw_snippet: str = ""


class ParsedEpisode(BaseModel):
    header: EpisodeHeader
    events: list[ParsedEvent] = Field(default_factory=list)
    unattached_camera_blocks: list[CameraBlock] = Field(default_factory=list)
    scenes: list[dict[str, Any]] = Field(default_factory=list)
    issues: list[ParseIssue] = Field(default_factory=list)
    block_counts: dict[str, int] = Field(default_factory=dict)


class CandidateWindow(BaseModel):
    window_id: str
    episode_number: int
    start_sec: float
    end_sec: float
    start_timestamp: str
    end_timestamp: str
    duration_sec: float
    start_event_id: int
    end_event_id: int
    event_ids: list[int]
    prefilter_score: float = 0.0
    prefilter_signals: dict[str, float] = Field(default_factory=dict)


class AliasMapping(BaseModel):
    canonical_name: str
    confidence: float = Field(ge=0.0, le=1.0)


class ShowProfile(BaseModel):
    show_brief: str
    alias_map: dict[str, AliasMapping] = Field(default_factory=dict)
    dynamic_lexicon: list[str] = Field(default_factory=list)


class StandaloneEvaluation(BaseModel):
    standalone_score: int = Field(ge=1, le=10)
    context_needed: bool
    missing_context: str = ""


class HookEvidence(BaseModel):
    score: int = Field(ge=1, le=10)
    event_id: int
    quote: str
    why: str = ""


class CliffhangerEvidence(BaseModel):
    score: int = Field(ge=1, le=10)
    event_id: int
    quote: str
    why: str = ""


class ContextualEvaluation(BaseModel):
    hook: HookEvidence
    cliffhanger: CliffhangerEvidence
    flags: dict[str, bool] = Field(default_factory=dict)
    archetypes: list[str] = Field(default_factory=list)
    why_it_works: str = ""


class ScoredCandidate(BaseModel):
    candidate: CandidateWindow
    standalone: StandaloneEvaluation | None = None
    contextual: ContextualEvaluation | None = None
    composite_score: float | None = None
    extended: bool = False
    extension_applied: bool = False
    llm_failed: bool = False
    llm_error: str | None = None
    validation_passed: bool | None = None
    validation_failures: list[str] = Field(default_factory=list)
    events: list[ParsedEvent] = Field(default_factory=list)


class ValidatedCandidate(BaseModel):
    scored: ScoredCandidate
    composite_score: float
    hook_score: int
    cliffhanger_score: int
    standalone_score: int


class PromoSegment(BaseModel):
    episode_number: int
    start_timestamp: str
    end_timestamp: str
    start_sec: float
    end_sec: float
    duration_sec: float
    start_event_id: int
    end_event_id: int
    event_ids: list[int]


class FinalPromo(BaseModel):
    rank: int
    promo_id: str
    episode_number: int
    segments: list[PromoSegment]
    duration_sec: float
    hook: dict[str, Any]
    cliffhanger: dict[str, Any]
    why_it_works: str
    scores: dict[str, float]
    composite_score: float
    human_review_status: Literal["PENDING", "APPROVED", "REJECTED"] = "PENDING"
    human_review_notes: str = ""
    window_id: str = ""
    archetypes: list[str] = Field(default_factory=list)
    flags: dict[str, bool] = Field(default_factory=dict)


class RunMetadata(BaseModel):
    run_id: str
    started_at: str
    finished_at: str | None = None
    input_dir: str
    mock_llm: bool = False
    limit_episodes: int | None = None
    resumed: bool = False
    model: str | None = None
    status: str = "running"
    notes: list[str] = Field(default_factory=list)


def normalize_quote(text: str) -> str:
    """Normalize case and whitespace only for quote comparison."""
    return " ".join((text or "").lower().split())
