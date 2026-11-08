"""Editorial studio helpers. Does not score or rank; it calls the existing pipeline."""

from __future__ import annotations

import json
import os
import re
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.config import ROOT_DIR, get_settings, load_llm_env
from src.parser import extract_episode_number_from_filename, parse_episode_text
from src.reliability import ensure_run_dirs, new_run_id
from src.run import parse_args, run_pipeline

UPLOAD_ROOT = ROOT_DIR / "data" / "uploads"
OUTPUT_ROOT = ROOT_DIR / "outputs"

STAGES: list[tuple[str, str]] = [
    ("parse", "Screenplay Parsing"),
    ("profile", "Show Profiling"),
    ("candidates", "Candidate Generation"),
    ("prefilter", "Prefiltering and Non-Maximum Suppression"),
    ("evaluate", "Blind and Contextual LLM Evaluation"),
    ("extension", "Conditional Clip Extension"),
    ("validate", "Hard Validation and Ranking"),
    ("export", "Final Export"),
]

JOBS: dict[str, threading.Thread] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_relative_name(name: str) -> str:
    """Keep the uploaded relative name and reject path traversal."""
    raw = (name or "").replace("\\", "/").lstrip("/")
    parts_raw = [part for part in raw.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts_raw):
        raise ValueError("Upload path cannot contain ..")
    parts = parts_raw
    if not parts:
        raise ValueError("Empty upload name")
    return "/".join(parts)


def save_uploads(project_dir: Path, files: list[Any]) -> list[Path]:
    """Write uploaded bytes unchanged under the project folder."""
    project_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for uploaded in files:
        rel = safe_relative_name(getattr(uploaded, "name", "untitled.txt"))
        suffix = Path(rel).suffix.lower()
        if suffix not in {".txt", ".md", ".markdown"}:
            continue
        dest = project_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = uploaded.getvalue()
        dest.write_bytes(data)
        saved.append(dest)
    return saved


def screenplay_files(project_dir: Path) -> list[Path]:
    if not project_dir.exists():
        return []
    found = [
        path
        for path in project_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".markdown"}
    ]
    return sorted(found, key=lambda path: path.name.lower())


def inspect_project(project_dir: Path) -> dict[str, Any]:
    """Validate saved screenplays without calling the LLM."""
    rows: list[dict[str, Any]] = []
    for path in screenplay_files(project_dir):
        row: dict[str, Any] = {
            "path": str(path),
            "filename": path.name,
            "relative": str(path.relative_to(project_dir)),
            "episode_number": None,
            "event_count": 0,
            "issue_count": 0,
            "status": "valid",
            "detail": "",
        }
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            row["status"] = "error"
            row["detail"] = f"Unreadable: {exc}"
            rows.append(row)
            continue
        if not text.strip():
            row["status"] = "error"
            row["detail"] = "File is empty"
            rows.append(row)
            continue
        try:
            parsed = parse_episode_text(
                text,
                source_filename=path.name,
                fallback_episode_number=extract_episode_number_from_filename(path.name),
            )
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["detail"] = f"Parser failed: {exc}"
            rows.append(row)
            continue
        row["episode_number"] = parsed.header.episode_number
        row["event_count"] = len(parsed.events)
        row["issue_count"] = len(parsed.issues)
        hard_errors = [issue.message for issue in parsed.issues if issue.severity == "error"]
        if parsed.header.episode_number <= 0 and not parsed.events:
            row["status"] = "error"
            row["detail"] = "No episode number and no primary events"
        elif not parsed.events and "SCRIPT START" not in text.upper():
            row["status"] = "error"
            row["detail"] = "No script start marker and no primary events"
        elif hard_errors and not parsed.events:
            row["status"] = "error"
            row["detail"] = "; ".join(hard_errors[:3])
        else:
            row["detail"] = f"{len(parsed.events)} primary events"
            if hard_errors:
                row["detail"] += f"; {len(hard_errors)} parser warnings logged as errors"
        rows.append(row)

    numbers = [row["episode_number"] for row in rows if row["status"] == "valid" and row["episode_number"]]
    counts: dict[int, int] = {}
    for number in numbers:
        counts[number] = counts.get(number, 0) + 1
    duplicates = sorted(number for number, count in counts.items() if count > 1)
    for row in rows:
        if row["episode_number"] in duplicates and row["status"] == "valid":
            row["status"] = "duplicate"
            row["detail"] = f"Episode {row['episode_number']} appears more than once"
    missing: list[int] = []
    if numbers:
        present = set(numbers)
        missing = [number for number in range(min(present), max(present) + 1) if number not in present]
    rows.sort(key=lambda row: (row["episode_number"] is None, row["episode_number"] or 10**9, row["filename"]))
    valid = sum(1 for row in rows if row["status"] == "valid")
    errors = sum(1 for row in rows if row["status"] == "error")
    return {
        "files": rows,
        "total_files": len(rows),
        "valid_episodes": valid,
        "duplicate_numbers": duplicates,
        "missing_numbers": missing,
        "errors": errors,
    }


def remove_project_file(project_dir: Path, relative: str) -> None:
    target = (project_dir / relative).resolve()
    root = project_dir.resolve()
    if root not in target.parents and target != root:
        raise ValueError("Refusing to delete outside the project folder")
    if target.exists() and target.is_file():
        target.unlink()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def audit(path: Path, run_id: str, event_type: str, **metadata: Any) -> None:
    append_jsonl(
        path,
        {
            "run_id": run_id,
            "event_type": event_type,
            "timestamp": utc_now(),
            **metadata,
        },
    )


def reviews_path(run_dir: Path) -> Path:
    return run_dir / "editorial_reviews.json"


def load_reviews(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = reviews_path(run_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_review(
    run_dir: Path,
    *,
    promo_id: str,
    window_id: str,
    status: str,
    comment: str,
) -> dict[str, Any]:
    if status not in {"APPROVED", "REJECTED", "NEEDS_MANUAL_REVIEW"}:
        raise ValueError(f"Unsupported review status: {status}")
    reviews = load_reviews(run_dir)
    record = {
        "promo_id": promo_id,
        "window_id": window_id,
        "status": status,
        "comment": comment,
        "updated_at": utc_now(),
    }
    reviews[promo_id or window_id] = record
    reviews_path(run_dir).write_text(
        json.dumps(reviews, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    audit(
        run_dir / "audit_log.jsonl",
        run_dir.name,
        "promo_review",
        promo_id=promo_id,
        window_id=window_id,
        status=status,
        comment=comment,
    )
    return record


def run_card(run_dir: Path) -> dict[str, Any]:
    """Lightweight summary for Home and History. Does not load checkpoints."""
    quality_path = run_dir / "quality_report.json"
    promos_path = run_dir / "promos.json"
    quality: dict[str, Any] = {}
    meta: dict[str, Any] = {}
    promo_count = 0
    promo_ids: list[str] = []
    if quality_path.exists():
        try:
            loaded = json.loads(quality_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                quality = loaded
        except (OSError, json.JSONDecodeError):
            quality = {}
    if promos_path.exists():
        try:
            loaded = json.loads(promos_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = {}
        if isinstance(loaded, dict):
            meta = loaded.get("run") if isinstance(loaded.get("run"), dict) else {}
            promos = loaded.get("promos") if isinstance(loaded.get("promos"), list) else []
            promo_count = len(promos)
            promo_ids = [str(promo.get("promo_id")) for promo in promos if promo.get("promo_id")]
    return {
        "run_id": meta.get("run_id") or run_dir.name,
        "mode": "REAL RUN",
        "status": meta.get("status") or "unknown",
        "episodes": quality.get("episodes_parsed"),
        "promos": quality.get("final_promos", promo_count),
        "promo_ids": promo_ids,
    }


def review_counts(run_dir: Path, promo_ids: list[str]) -> dict[str, int]:
    reviews = load_reviews(run_dir)
    counts = {"APPROVED": 0, "REJECTED": 0, "NEEDS_MANUAL_REVIEW": 0, "PENDING": 0}
    for promo_id in promo_ids:
        status = (reviews.get(promo_id) or {}).get("status") or "PENDING"
        counts[status] = counts.get(status, 0) + 1
    return counts


def slug_run_id(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip()).strip("_")
    if not slug:
        return new_run_id()
    if not slug.startswith("run_"):
        slug = f"run_{slug}"
    return slug[:80]


def unique_run_id(name: str) -> str:
    base = slug_run_id(name)
    candidate = base
    suffix = 2
    while (OUTPUT_ROOT / candidate).exists():
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def write_run_config(
    run_dir: Path,
    *,
    top_n: int,
    limit_episodes: int | None,
) -> Path:
    settings = get_settings()
    data = settings.model_dump()
    data["app"]["top_n_promos"] = int(top_n)
    path = run_dir / "studio_config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    note = {
        "top_n_promos": int(top_n),
        "limit_episodes": limit_episodes,
    }
    (run_dir / "studio_settings.json").write_text(
        json.dumps(note, indent=2),
        encoding="utf-8",
    )
    return path


def blank_execution_state(run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "running",
        "error": None,
        "stages": {
            key: {
                "label": label,
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "info": {},
            }
            for key, label in STAGES
        },
    }


def execution_state_path(run_dir: Path) -> Path:
    return run_dir / "execution_state.json"


def load_execution_state(run_dir: Path) -> dict[str, Any] | None:
    path = execution_state_path(run_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_state(run_dir: Path, state: dict[str, Any]) -> None:
    execution_state_path(run_dir).write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def make_progress(run_dir: Path, run_id: str):
    state = blank_execution_state(run_id)
    _write_state(run_dir, state)
    lock = threading.Lock()

    def callback(stage: str, status: str, info: dict | None = None) -> None:
        info = info or {}
        now = utc_now()
        with lock:
            stage_state = state["stages"].setdefault(
                stage,
                {"label": stage, "status": "pending", "started_at": None, "finished_at": None, "info": {}},
            )
            stage_state["status"] = status
            stage_state["info"] = info
            if status == "running" and not stage_state.get("started_at"):
                stage_state["started_at"] = now
            if status in {"completed", "failed"}:
                stage_state["finished_at"] = now
            if status == "failed":
                state["status"] = "failed"
                state["error"] = info.get("error")
            _write_state(run_dir, state)
        append_jsonl(
            run_dir / "execution_log.jsonl",
            {
                "run_id": run_id,
                "event_type": f"stage_{status}",
                "stage": stage,
                "timestamp": now,
                "info": info,
            },
        )

    def fail(message: str) -> None:
        callback("pipeline", "failed", {"error": message})
        with lock:
            state["status"] = "failed"
            state["error"] = message
            for stage_state in state["stages"].values():
                if stage_state["status"] == "running":
                    stage_state["status"] = "failed"
                    stage_state["finished_at"] = utc_now()
                    stage_state["info"] = {**stage_state.get("info", {}), "error": message}
            _write_state(run_dir, state)

    def finish() -> None:
        with lock:
            if state["status"] != "failed":
                state["status"] = "completed"
            _write_state(run_dir, state)

    return callback, fail, finish


def api_key_configured() -> bool:
    return bool(load_llm_env().api_key)


def public_model_settings() -> dict[str, str]:
    env = load_llm_env()
    return {
        "model": env.model or "",
        "base_url": env.base_url or "",
        "key_configured": "yes" if env.api_key else "no",
    }


def start_discovery(
    *,
    project_dir: Path,
    run_name: str,
    top_n: int,
    limit_episodes: int | None,
    model_name: str | None,
) -> Path:
    """Start the existing pipeline once, in a background thread."""
    if not api_key_configured():
        raise RuntimeError("LLM_API_KEY is not set in .env. The key is never shown in the app.")
    report = inspect_project(project_dir)
    if report["valid_episodes"] < 1:
        raise RuntimeError("No valid episodes to process.")
    if report["duplicate_numbers"]:
        raise RuntimeError(
            "Duplicate episode numbers: " + ", ".join(str(n) for n in report["duplicate_numbers"])
        )

    run_id = unique_run_id(run_name)
    run_dir = ensure_run_dirs(OUTPUT_ROOT, run_id)
    config_path = write_run_config(
        run_dir,
        top_n=top_n,
        limit_episodes=limit_episodes,
    )
    audit(
        run_dir / "audit_log.jsonl",
        run_id,
        "evaluation_started",
        top_n=top_n,
        limit_episodes=limit_episodes,
        input_dir=str(project_dir),
    )
    callback, fail, finish = make_progress(run_dir, run_id)
    argv = [
        "--input-dir",
        str(project_dir),
        "--output-dir",
        str(OUTPUT_ROOT),
        "--config",
        str(config_path),
    ]
    if limit_episodes:
        argv.extend(["--limit-episodes", str(int(limit_episodes))])
    args = parse_args(argv)
    args.run_id = run_id

    def worker() -> None:
        if model_name:
            os.environ["LLM_MODEL"] = model_name
        try:
            import asyncio

            asyncio.run(run_pipeline(args, progress=callback))
            finish()
        except Exception as exc:  # noqa: BLE001
            fail(f"{exc}\n{traceback.format_exc()}")

    thread = threading.Thread(target=worker, name=f"kuku-{run_id}", daemon=True)
    JOBS[run_id] = thread
    thread.start()
    return run_dir
