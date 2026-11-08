"""Configuration loading for the promo finder."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"


class AppConfig(BaseModel):
    name: str = "Kuku Narrative-Aware Promo Finder"
    top_n_promos: int = 15
    max_candidates_per_episode: int = 2
    max_promos_per_episode: int = 2


class WindowsConfig(BaseModel):
    min_duration_sec: float = 20.0
    max_duration_sec: float = 60.0
    hook_zone_sec: float = 3.0
    cliffhanger_final_zone_sec: float = 5.0
    max_cliffhanger_events: int = 2


class PrefilterWeights(BaseModel):
    hook_dialogue_or_action: float = 2.0
    hook_universal_term: float = 2.0
    hook_dynamic_lexicon: float = 1.5
    hook_intensity_tag: float = 1.5
    hook_punchy_dialogue: float = 1.5
    hook_interruption: float = 1.0
    cliff_abrupt_marker: float = 2.5
    cliff_unanswered_question: float = 2.0
    cliff_dynamic_lexicon: float = 1.5
    cliff_resolution_penalty: float = -3.0
    ground_relationship_nouns: float = 2.0
    ground_named_speakers: float = 1.0
    ground_ungrounded_pronoun_penalty: float = -1.5


class PrefilterConfig(BaseModel):
    enable_dynamic_lexicon: bool = True
    nms_metric: str = "intersection_over_shorter"
    nms_threshold: float = 0.35
    universal_conflict_terms: list[str] = Field(default_factory=list)
    weights: PrefilterWeights = Field(default_factory=PrefilterWeights)


class ScoringConfig(BaseModel):
    hook_weight: float = 0.40
    cliffhanger_weight: float = 0.35
    standalone_weight: float = 0.25
    strong_score_threshold: int = 7


class ExtensionConfig(BaseModel):
    enabled: bool = True
    max_preceding_events: int = 2
    max_attempts: int = 1
    require_standalone_improvement: bool = True


class LLMSettings(BaseModel):
    temperature: float = 0.0
    max_concurrency: int = 1
    timeout_seconds: int = 90
    retry_attempts: int = 1
    retry_with_validation_error: bool = True


class ReliabilityConfig(BaseModel):
    checkpoint_file: str = "checkpoint_scored.jsonl"
    resume_enabled_by_default: bool = False
    save_raw_responses: bool = True


class OutputConfig(BaseModel):
    save_json: bool = True
    save_csv: bool = True
    save_failures: bool = True
    save_quality_report: bool = True
    save_run_log: bool = True


class AppSettings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    windows: WindowsConfig = Field(default_factory=WindowsConfig)
    prefilter: PrefilterConfig = Field(default_factory=PrefilterConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    extension: ExtensionConfig = Field(default_factory=ExtensionConfig)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


class LLMEnvConfig(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    model: str = "openai/gpt-oss-20b"
    temperature: float = 0.0
    max_concurrency: int = 2
    timeout_seconds: int = 90


def load_yaml_config(path: Path | None = None) -> AppSettings:
    config_path = path or DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
            if isinstance(loaded, dict):
                data = loaded
    return AppSettings.model_validate(data)


def load_llm_env(settings: AppSettings | None = None) -> LLMEnvConfig:
    load_dotenv(ROOT_DIR / ".env", override=True)
    settings = settings or load_yaml_config()
    return LLMEnvConfig(
        api_key=os.getenv("LLM_API_KEY") or None,
        base_url=os.getenv("LLM_BASE_URL") or None,
        model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
        temperature=float(os.getenv("LLM_TEMPERATURE", str(settings.llm.temperature))),
        max_concurrency=int(
            os.getenv("LLM_MAX_CONCURRENCY", str(settings.llm.max_concurrency))
        ),
        timeout_seconds=int(
            os.getenv("LLM_TIMEOUT_SECONDS", str(settings.llm.timeout_seconds))
        ),
    )


def get_settings(config_path: Path | None = None) -> AppSettings:
    return load_yaml_config(config_path)
