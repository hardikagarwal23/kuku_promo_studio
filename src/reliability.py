"""Retries, checkpoints, run IDs, and raw response capture."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

SECRET_PATTERNS = [
    re.compile(r"(api[_-]?key\s*[:=]\s*)(\S+)", re.IGNORECASE),
    re.compile(r"(authorization\s*:\s*bearer\s+)(\S+)", re.IGNORECASE),
    re.compile(r"(LLM_API_KEY\s*=\s*)(\S+)", re.IGNORECASE),
]


def new_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc).astimezone()
    return f"run_{now.strftime('%Y-%m-%d_%H%M%S')}"


RAW_RESPONSE_STAGES = (
    "show_profile",
    "standalone_blind",
    "hook_cliffhanger",
    "extension_rescore",
)


def ensure_run_dirs(output_root: Path, run_id: str) -> Path:
    run_dir = output_root / run_id
    raw_dir = run_dir / "raw_llm_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Ensure required raw JSONL files exist even if a stage never fires.
    for stage in RAW_RESPONSE_STAGES:
        path = raw_dir / f"{stage}.jsonl"
        if not path.exists():
            path.touch()
    return run_dir


def redact_secrets(text: str) -> str:
    redacted = text
    for pat in SECRET_PATTERNS:
        redacted = pat.sub(r"\1***REDACTED***", redacted)
    return redacted


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = json.loads(redact_secrets(json.dumps(record, ensure_ascii=False, default=str)))
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(safe, ensure_ascii=False) + "\n")


def save_raw_response(
    run_dir: Path,
    stage: str,
    *,
    raw_text: str,
    metadata: dict[str, Any],
) -> None:
    path = run_dir / "raw_llm_responses" / f"{stage}.jsonl"
    append_jsonl(
        path,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "raw_text": raw_text,
            **metadata,
        },
    )


def load_checkpoint_ids(checkpoint_path: Path) -> set[str]:
    done: set[str] = set()
    if not checkpoint_path.exists():
        return done
    with checkpoint_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                wid = obj.get("window_id")
                if wid:
                    done.add(wid)
            except json.JSONDecodeError:
                continue
    return done


def parse_json_from_text(text: str) -> Any:
    """Extract JSON object/array from model text that may include fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try first {...} or [...]
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
        if not m:
            raise
        return json.loads(m.group(1))


async def validate_with_retry(
    *,
    call_fn: Callable[..., Any],
    call_kwargs: dict[str, Any],
    model_cls: type[T],
    retry_attempts: int = 1,
    include_validation_error: bool = True,
) -> tuple[T | None, str, str | None]:
    """
    Call an LLM function returning raw text, validate with Pydantic.
    Returns (parsed_or_none, raw_text, error_message).
    """
    raw_text = ""
    last_error: str | None = None
    attempts = 1 + max(0, retry_attempts)
    kwargs = dict(call_kwargs)

    for attempt in range(attempts):
        try:
            raw_text = await call_fn(**kwargs)
            data = parse_json_from_text(raw_text)
            parsed = model_cls.model_validate(data)
            return parsed, raw_text, None
        except (ValidationError, json.JSONDecodeError, ValueError, TypeError) as exc:
            last_error = str(exc)
            logger.warning(
                "Schema/parse failure attempt %s/%s: %s",
                attempt + 1,
                attempts,
                redact_secrets(last_error),
            )
            if attempt + 1 >= attempts:
                break
            if include_validation_error:
                # Append validation error guidance for retry
                base_prompt = kwargs.get("prompt", "")
                kwargs["prompt"] = (
                    f"{base_prompt}\n\nYour previous response failed validation:\n"
                    f"{last_error}\nReturn corrected STRICT JSON only."
                )
        except Exception as exc:  # noqa: BLE001
            last_error = f"API/transport error: {exc}"
            logger.warning(redact_secrets(last_error))
            if attempt + 1 >= attempts:
                break
            await _sleep_backoff(attempt)

    return None, raw_text, last_error


async def _sleep_backoff(attempt: int) -> None:
    delay = min(8.0, 0.5 * (2**attempt))
    await asyncio.sleep(delay)
