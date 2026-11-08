"""Kuku Promo Studio — editorial application on top of the existing pipeline."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import streamlit as st

from dashboard_data import find_scored, list_runs, load_run, promo_segment, ranking_breakdown
from src.config import get_settings
from studio_ops import (
    JOBS,
    STAGES,
    UPLOAD_ROOT,
    api_key_configured,
    audit,
    inspect_project,
    load_execution_state,
    load_reviews,
    public_model_settings,
    remove_project_file,
    review_counts,
    run_card,
    save_review,
    save_uploads,
    screenplay_files,
    start_discovery,
    utc_now,
)

st.set_page_config(
    page_title="Kuku Promo Studio",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #080808; color: #FFFFFF; }
    section[data-testid="stSidebar"] { background: #0E0E0E; border-right: 1px solid #2A2A2A; }
    section[data-testid="stSidebar"] * { color: #FFFFFF; }
    h1, h2, h3 { color: #FFFFFF !important; letter-spacing: -0.02em; }
    .studio-kicker { color: #E53935; font-weight: 700; letter-spacing: 0.14em; font-size: 0.75rem; }
    .studio-muted { color: #B3B3B3; }
    .log-error, .log-error li { color: #E53935 !important; }
    .studio-card {
        background: #1F1F1F; border: 1px solid #2A2A2A; border-radius: 12px;
        padding: 1rem 1.1rem; margin-bottom: 0.8rem;
    }
    .rank-num { color: #FFC107; font-size: 2.4rem; font-weight: 750; line-height: 1; }
    .pill { display: inline-block; border-radius: 999px; padding: 0.12rem 0.55rem;
            font-size: 0.75rem; font-weight: 700; margin-right: 0.35rem; }
    .pill-mock { background: #3a2a10; color: #FFC107; }
    .pill-real { background: #12361f; color: #1DB954; }
    .pill-ok { background: #12361f; color: #1DB954; }
    .pill-warn { background: #3a2a10; color: #FFC107; }
    .pill-bad { background: #3a1212; color: #E53935; }
    .pill-neutral { background: #2A2A2A; color: #B3B3B3; }
    div[data-testid="stButton"] button[kind="primary"] {
        background: #E53935; border: 1px solid #E53935; color: white; border-radius: 8px;
    }
    div[data-testid="stMetric"] {
        background: #1F1F1F; border: 1px solid #2A2A2A; border-radius: 12px; padding: 0.6rem;
    }
    .tl-wrap { position: relative; height: 36px; background: #141414; border-radius: 8px; margin: 0.4rem 0 0.6rem; }
    .tl-hookzone { position: absolute; top: 0; bottom: 0; background: rgba(229,57,53,0.35); border-radius: 8px 0 0 8px; }
    .tl-event { position: absolute; top: 7px; height: 22px; border-radius: 4px; }
    .tl-dlg { background: #FFFFFF; }
    .tl-act { background: #B3B3B3; }
    .tl-sfx { background: #FFC107; }
    .tl-hook { outline: 2px solid #E53935; }
    .tl-cliff { outline: 2px solid #FFC107; }
    </style>
    """,
    unsafe_allow_html=True,
)

NAV = [
    "Home",
    "New Project",
    "Execution",
    "Run History",
    "Final Promos",
    "Candidate Explorer",
    "Editorial Review",
    "Execution Logs",
]


def _init() -> None:
    st.session_state.setdefault("screen", "Home")
    st.session_state.setdefault("project_id", None)
    st.session_state.setdefault("active_run", None)
    st.session_state.setdefault("selected_promo", None)


def _go(screen: str) -> None:
    st.session_state.screen = screen
    st.rerun()


def _project_dir() -> Path | None:
    project_id = st.session_state.get("project_id")
    if not project_id:
        return None
    return UPLOAD_ROOT / project_id


def _active_bundle():
    run_name = st.session_state.get("active_run")
    if not run_name:
        runs = list_runs()
        if not runs:
            return None
        run_name = runs[0].name
        st.session_state.active_run = run_name
    path = Path("outputs") / run_name
    if not (path / "promos.json").exists():
        return None
    return load_run(path)


def _mode_html(mode: str) -> str:
    return '<span class="pill pill-real">REAL RUN</span>'


def _review_status(bundle, promo_id: str) -> str:
    reviews = load_reviews(Path(bundle["run_dir"]))
    return (reviews.get(promo_id) or {}).get("status") or "PENDING"


def _sidebar() -> None:
    st.sidebar.markdown('<div class="studio-kicker">KUKU</div>', unsafe_allow_html=True)
    st.sidebar.markdown("## Promo Studio")
    st.sidebar.caption("Find the moments. An editor decides.")
    current = st.session_state.screen
    for name in NAV:
        label = f"● {name}" if name == current else name
        if st.sidebar.button(label, key=f"nav-{name}", width="stretch"):
            _go(name)
    run_name = st.session_state.get("active_run")
    if run_name:
        st.sidebar.caption(f"Open run · {run_name}")


def _timeline(events: list[dict], start: float, end: float, hook_id, cliff_id) -> str:
    span = max(end - start, 0.001)
    hook_pct = min(100.0, 3.0 / span * 100.0)
    bars = [f'<div class="tl-hookzone" style="width:{hook_pct:.2f}%"></div>']
    for event in events:
        ev_start = float(event.get("start_sec") or start)
        ev_end = float(event.get("end_sec") or ev_start)
        left = max(0.0, (ev_start - start) / span * 100.0)
        width = max(0.8, (ev_end - ev_start) / span * 100.0)
        kind = {"DIALOGUE": "tl-dlg", "ACTION": "tl-act", "SOUND_FX": "tl-sfx"}.get(
            event.get("block_type"), "tl-act"
        )
        extra = ""
        if hook_id is not None and event.get("event_id") == hook_id:
            extra += " tl-hook"
        if cliff_id is not None and event.get("event_id") == cliff_id:
            extra += " tl-cliff"
        bars.append(
            f'<div class="tl-event {kind}{extra}" style="left:{left:.2f}%;width:{width:.2f}%"></div>'
        )
    return f'<div class="tl-wrap">{"".join(bars)}</div>'


def screen_home() -> None:
    st.markdown('<div class="studio-kicker">NARRATIVE-AWARE PROMO FINDER</div>', unsafe_allow_html=True)
    st.title("Kuku Promo Studio")
    st.markdown(
        '<p class="studio-muted">Upload a screenplay folder, run the existing nine-stage finder, '
        "and approve the clips an editor would actually post. The studio recommends time ranges. </p>",
        unsafe_allow_html=True,
    )
    if st.button("New Project", type="primary"):
        _go("New Project")

    st.subheader("Previous runs")
    runs = list_runs()
    if not runs:
        st.info("No runs yet. Start a project and discovery begins only after you press Start.")
        return
    for path in runs[:8]:
        try:
            card = run_card(path)
        except Exception as exc:  # noqa: BLE001
            st.error(f"{path.name} could not be read: {exc}")
            continue
        reviews = review_counts(path, card["promo_ids"])
        left, right = st.columns([5, 1])
        with left:
            st.markdown(
                f'<div class="studio-card">{_mode_html(card["mode"])}'
                f"<h3>{card['run_id']}</h3>"
                f'<p class="studio-muted">{card["status"]} · '
                f'{card["episodes"] if card["episodes"] is not None else "—"} episodes · '
                f'{card["promos"]} promos · '
                f'{reviews["APPROVED"]} approved</p></div>',
                unsafe_allow_html=True,
            )
        with right:
            if st.button("Open", key=f"open-run-{path.name}"):
                st.session_state.active_run = path.name
                _go("Final Promos")


def screen_upload() -> None:
    st.markdown('<div class="studio-kicker">PROJECT</div>', unsafe_allow_html=True)
    st.title("Upload screenplays")
    st.caption("Files stay on this machine. Nothing is sent until you later start discovery.")
    if st.button("Start a different project"):
        st.session_state.project_id = None
        st.rerun()
    if not st.session_state.project_id:
        st.session_state.project_id = "project_" + utc_now().replace(":", "").replace("-", "")[:15]
        audit(
            (_project_dir() or UPLOAD_ROOT) / "audit_log.jsonl",
            st.session_state.project_id,
            "project_created",
        )
    project_dir = _project_dir()
    assert project_dir is not None
    project_dir.mkdir(parents=True, exist_ok=True)

    folder = st.file_uploader(
        "Screenplay folder",
        type=["txt", "md", "markdown"],
        accept_multiple_files="directory",
        key="folder_upload",
        help="Select a folder of episode files. Original bytes are stored unchanged.",
    )
    many = st.file_uploader(
        "Or choose individual files",
        type=["txt", "md", "markdown"],
        accept_multiple_files=True,
        key="file_upload",
    )
    incoming = list(folder or []) + list(many or [])
    if incoming and st.button("Add files to this project", type="primary"):
        saved = save_uploads(project_dir, incoming)
        audit(
            project_dir / "audit_log.jsonl",
            st.session_state.project_id,
            "files_added",
            count=len(saved),
            filenames=[path.name for path in saved],
        )
        st.success(f"Stored {len(saved)} file(s) without editing their contents.")

    report = inspect_project(project_dir)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total files", report["total_files"])
    c2.metric("Valid episodes", report["valid_episodes"])
    c3.metric("Missing episodes", len(report["missing_numbers"]))
    c4.metric("Errors", report["errors"] + sum(1 for row in report["files"] if row["status"] == "duplicate"))
    if report["missing_numbers"]:
        st.warning("Missing episode numbers: " + ", ".join(str(n) for n in report["missing_numbers"]))
    if report["duplicate_numbers"]:
        st.error("Duplicate episode numbers: " + ", ".join(str(n) for n in report["duplicate_numbers"]))
    if not report["files"]:
        st.info("No screenplay files in this project yet.")
        return

    st.subheader("Detected episodes")
    for row in report["files"]:
        cols = st.columns([4, 1, 2, 1])
        cols[0].write(row["relative"])
        cols[1].write(row["episode_number"] if row["episode_number"] is not None else "—")
        cols[2].write(f"{row['status']} — {row['detail']}")
        if cols[3].button("Remove", key=f"rm-{row['relative']}"):
            remove_project_file(project_dir, row["relative"])
            audit(
                project_dir / "audit_log.jsonl",
                st.session_state.project_id,
                "file_removed",
                filename=row["relative"],
            )
            st.rerun()

    ready = report["valid_episodes"] > 0 and not report["duplicate_numbers"]
    if st.button("Continue to configuration", type="primary", disabled=not ready):
        audit(project_dir / "audit_log.jsonl", st.session_state.project_id, "configuration_opened")
        _go("Configure")


def screen_configure() -> None:
    st.markdown('<div class="studio-kicker">CONFIGURATION</div>', unsafe_allow_html=True)
    st.title("Project configuration")
    project_dir = _project_dir()
    if project_dir is None or not screenplay_files(project_dir):
        st.warning("Upload screenplays before configuring a run.")
        if st.button("Back to upload"):
            _go("New Project")
        return
    report = inspect_project(project_dir)
    st.write(
        f"{report['valid_episodes']} valid episodes ready. "
        "Discovery does not start on this page."
    )
    settings = get_settings()
    model = public_model_settings()
    st.markdown(
        f"**Provider endpoint:** `{model['base_url'] or 'default OpenAI endpoint'}`  \n"
        f"**API key configured:** {model['key_configured']}  \n"
        "The key itself is never displayed."
    )
    model_name = st.text_input("Model name", value=model["model"])
    if not api_key_configured():
        st.error("Discovery is unavailable until LLM_API_KEY is set in .env.")
    top_n = st.number_input(
        "Final promos to recommend",
        min_value=1,
        max_value=50,
        value=int(settings.app.top_n_promos),
    )
    limit = st.number_input(
        "Episode limit (0 = every valid file)",
        min_value=0,
        value=0,
        step=1,
    )
    run_name = st.text_input("Run name", value="")
    st.info(
        "Estimated API usage: unavailable. "
        "Call volume depends on the candidate count, which is only known after parsing and window generation."
    )
    if st.button("Start Promo Discovery", type="primary", disabled=not api_key_configured()):
        try:
            run_dir = start_discovery(
                project_dir=project_dir,
                run_name=run_name,
                top_n=int(top_n),
                limit_episodes=int(limit) or None,
                model_name=model_name.strip() or None,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return
        audit(
            project_dir / "audit_log.jsonl",
            run_dir.name,
            "configuration_changed",
            project_id=st.session_state.project_id,
            top_n=int(top_n),
            limit_episodes=int(limit) or None,
            model_name=model_name.strip(),
        )
        st.session_state.active_run = run_dir.name
        st.session_state.screen = "Execution"
        st.rerun()


def _render_stages(state: dict) -> None:
    stages = state.get("stages") or {}
    for key, label in STAGES:
        stage = stages.get(key) or {"status": "pending", "info": {}}
        status = stage.get("status") or "pending"
        pill = {
            "completed": "pill-ok",
            "running": "pill-warn",
            "failed": "pill-bad",
        }.get(status, "pill-neutral")
        info = stage.get("info") or {}
        counter = " · ".join(f"{name} {value}" for name, value in info.items() if name != "error")
        st.markdown(
            f'<div class="studio-card"><span class="pill {pill}">{status.upper()}</span> '
            f"<strong>{label}</strong>"
            f'<p class="studio-muted">Started {stage.get("started_at") or "—"} · '
            f'Finished {stage.get("finished_at") or "—"}'
            f"{('<br>' + counter) if counter else ''}</p></div>",
            unsafe_allow_html=True,
        )
        if status == "failed" and info.get("error"):
            st.error(str(info["error"]))
        if key == "evaluate" and info.get("total"):
            done = int(info.get("evaluated") or 0)
            total = int(info["total"])
            st.progress(done / total if total else 0.0, text=f"{done} / {total} candidates evaluated")


@st.fragment(run_every=1.0)
def _live_panel(run_dir_str: str) -> None:
    state = load_execution_state(Path(run_dir_str))
    if not state:
        st.info("Waiting for the pipeline to record its first stage.")
        return
    info_by_stage = {key: (state.get("stages") or {}).get(key, {}).get("info") or {} for key, _ in STAGES}
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Episodes", info_by_stage["parse"].get("episodes", "—"))
    c2.metric("Candidates", info_by_stage["candidates"].get("raw_candidates", "—"))
    c3.metric("Prefilter rejected", info_by_stage["prefilter"].get("rejected_by_prefilter", "—"))
    c4.metric("Evaluated", info_by_stage["evaluate"].get("evaluated", "—"))
    c5.metric("Final promos", info_by_stage["export"].get("final_promos", "—"))
    if state.get("error"):
        st.error(state["error"])
    _render_stages(state)
    st.subheader("Execution log")
    _render_execution_summary(Path(run_dir_str), state)
    if state.get("status") == "completed":
        st.markdown("Discovery finished. Open Final Promos to review the saved shortlist.")


def screen_execution() -> None:
    st.markdown('<div class="studio-kicker">LIVE RUN</div>', unsafe_allow_html=True)
    st.title("Pipeline execution")
    run_name = st.session_state.get("active_run")
    if not run_name:
        st.info("No discovery has been started in this session.")
        return
    run_dir = Path("outputs") / run_name
    thread = JOBS.get(run_name)
    if thread and thread.is_alive():
        st.caption(f"{run_name} is running. This view polls saved stage events once a second.")
    elif load_execution_state(run_dir):
        st.caption(f"{run_name} — background worker is not currently running.")
    else:
        st.warning("This run has no studio execution record. Older CLI runs can still be opened from history.")
        return
    _live_panel(str(run_dir))
    if st.button("Open results when ready"):
        _go("Final Promos")


def _require_bundle():
    bundle = _active_bundle()
    if bundle is None:
        st.info("Open a completed run from Home or Run History.")
        return None
    return bundle


def screen_history() -> None:
    st.title("Run history")
    runs = list_runs()
    if not runs:
        st.info("No completed runs in outputs/.")
        return
    for path in runs:
        try:
            card = run_card(path)
        except Exception as exc:  # noqa: BLE001
            st.error(f"{path.name}: {exc}")
            continue
        cols = st.columns([5, 1], vertical_alignment="center")
        cols[0].markdown(
            f'<div class="studio-card">{_mode_html(card["mode"])}'
            f"<h3>{card['run_id']}</h3>"
            f'<p class="studio-muted">{card["status"]} · '
            f'{card["episodes"] if card["episodes"] is not None else "—"} episodes · '
            f"{card['promos']} promos</p></div>",
            unsafe_allow_html=True,
        )
        if cols[1].button("Open", key=f"hist-{path.name}"):
            st.session_state.active_run = path.name
            _go("Final Promos")


def screen_results_and_promos(*, details: bool = False) -> None:
    bundle = _require_bundle()
    if bundle is None:
        return
    if not details:
        st.markdown(f"# Results {_mode_html(bundle['mode'])}", unsafe_allow_html=True)
        quality = bundle["quality"]
        cols = st.columns(4)
        cols[0].metric("Episodes", quality.get("episodes_parsed", "—"))
        cols[1].metric("Candidate windows", quality.get("raw_candidates", "—"))
        cols[2].metric("After prefilter", quality.get("candidates_after_nms", "—"))
        cols[3].metric("Final promos", quality.get("final_promos", len(bundle["promos"])))
        st.subheader("Top ranked")
    promos = list(bundle["promos"])
    if not promos:
        st.info("This run saved no final promos.")
        return
    episodes = sorted({p.get("episode_number") for p in promos if p.get("episode_number") is not None})
    statuses = ["PENDING", "APPROVED", "REJECTED", "NEEDS_MANUAL_REVIEW"]
    f1, f2, f3 = st.columns(3)
    ep_pick = f1.multiselect("Episode", episodes, default=episodes)
    status_pick = f2.multiselect("Review", statuses, default=statuses)
    sort_by = f3.selectbox("Sort", ["Rank", "Episode", "Composite score", "Review status"])
    shown = []
    for promo in promos:
        review = _review_status(bundle, promo.get("promo_id", ""))
        if ep_pick and promo.get("episode_number") not in ep_pick:
            continue
        if status_pick and review not in status_pick:
            continue
        shown.append((promo, review))
    def sort_key(item):
        promo, review = item
        if sort_by == "Episode":
            return (promo.get("episode_number") or 0, promo.get("rank") or 0)
        if sort_by == "Composite score":
            return (-(promo.get("composite_score") or 0), promo.get("rank") or 0)
        if sort_by == "Review status":
            return (review, promo.get("rank") or 0)
        return (promo.get("rank") or 0, 0)
    shown.sort(key=sort_key)
    if not shown:
        st.info("No promos match these filters.")
        return
    for promo, review in shown:
        _promo_card(bundle, promo, review)


def _promo_card(bundle, promo, review: str) -> None:
    seg = promo_segment(promo)
    scores = promo.get("scores") or {}
    hook = promo.get("hook") or {}
    cliff = promo.get("cliffhanger") or {}
    record = find_scored(bundle, str(promo.get("window_id") or ""))
    events = []
    if record and isinstance(record.get("events"), list):
        events = sorted(record["events"], key=lambda event: (event.get("start_sec") or 0, event.get("event_id") or 0))
    with st.container(border=True):
        left, mid, right = st.columns([1, 4, 1])
        left.markdown(f'<div class="rank-num">#{promo.get("rank", "—")}</div>', unsafe_allow_html=True)
        mid.markdown(
            f"### Episode {promo.get('episode_number', '—')} · "
            f"{seg.get('start_timestamp', '—')} – {seg.get('end_timestamp', '—')}"
        )
        mid.caption(
            f"{promo.get('duration_sec', '—')}s · composite {promo.get('composite_score', '—')} · "
            f"hook {scores.get('hook', '—')} · cliffhanger {scores.get('cliffhanger', '—')} · "
            f"standalone {scores.get('standalone', '—')}"
        )
        mid.write(f"Hook: {hook.get('quote') or 'No hook quote stored.'}")
        mid.write(f"Cliffhanger: {cliff.get('quote') or 'No cliffhanger quote stored.'}")
        mid.caption(f"Review {review}")
        if right.button("Inspect", key=f"inspect-{promo.get('promo_id')}"):
            st.session_state.selected_promo = promo.get("promo_id")
            audit(
                Path(bundle["run_dir"]) / "audit_log.jsonl",
                bundle["run_id"],
                "promo_opened",
                promo_id=promo.get("promo_id"),
                window_id=promo.get("window_id"),
            )
            st.session_state.screen = "Promo Details"
            st.rerun()
        if events and seg:
            st.markdown(
                _timeline(
                    events,
                    float(seg.get("start_sec") or 0),
                    float(seg.get("end_sec") or 0),
                    hook.get("event_id"),
                    cliff.get("event_id"),
                ),
                unsafe_allow_html=True,
            )
            st.caption("White dialogue · gray action · amber sound. Red band is the first 3 seconds. Red outline is the stored hook. Amber outline is the stored cliffhanger.")
        with st.expander("Dialogue, actions, and stored evidence"):
            st.write(promo.get("why_it_works") or "No explanation stored.")
            if not events:
                st.info("No event list was stored for this window.")
            for event in events:
                marker = []
                if event.get("event_id") == hook.get("event_id"):
                    marker.append("HOOK")
                if event.get("event_id") == cliff.get("event_id"):
                    marker.append("CLIFFHANGER")
                st.markdown(
                    f"**{event.get('start_timestamp', '—')}–{event.get('end_timestamp', '—')}** "
                    f"{event.get('speaker_raw') or event.get('block_type')}"
                    f"{(' · ' + ', '.join(marker)) if marker else ''}"
                )
                st.write(event.get("text") or "(empty text)")


def screen_details() -> None:
    bundle = _require_bundle()
    if bundle is None:
        return
    st.title("Promo details")
    promos = bundle["promos"]
    if not promos:
        st.info("This run has no promos.")
        return
    ids = [promo.get("promo_id") for promo in promos]
    selected = st.session_state.get("selected_promo")
    index = ids.index(selected) if selected in ids else 0
    promo_id = st.selectbox("Promo", ids, index=index)
    promo = next(item for item in promos if item.get("promo_id") == promo_id)
    record = find_scored(bundle, str(promo.get("window_id") or ""))
    if record is None:
        st.error("No scored checkpoint record for this promo. Evidence that was not saved is not shown.")
        return
    seg = promo_segment(promo)
    hook = (record.get("contextual") or {}).get("hook") or promo.get("hook") or {}
    cliff = (record.get("contextual") or {}).get("cliffhanger") or promo.get("cliffhanger") or {}
    events = record.get("events") or []
    st.markdown(
        _timeline(
            events,
            float(seg.get("start_sec") or 0),
            float(seg.get("end_sec") or 0),
            hook.get("event_id"),
            cliff.get("event_id"),
        ),
        unsafe_allow_html=True,
    )
    st.write(
        {
            "Episode": promo.get("episode_number"),
            "Start": seg.get("start_timestamp"),
            "End": seg.get("end_timestamp"),
            "Duration": promo.get("duration_sec"),
        }
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Blind evaluator")
        st.json(record.get("standalone") or {"stored": False})
        st.subheader("Hook evidence")
        st.json(hook or {"stored": False})
    with right:
        st.subheader("Contextual evaluator")
        contextual = record.get("contextual") or {}
        st.json(
            {
                "why_it_works": contextual.get("why_it_works"),
                "flags": contextual.get("flags"),
                "archetypes": contextual.get("archetypes"),
            }
        )
        st.subheader("Cliffhanger evidence")
        st.json(cliff or {"stored": False})
    st.subheader("Extension")
    st.write({"extended": record.get("extended"), "extension_applied": record.get("extension_applied")})
    st.subheader("Hard validation")
    st.write({"passed": record.get("validation_passed"), "failures": record.get("validation_failures") or []})
    st.subheader("Score breakdown")
    breakdown = ranking_breakdown(record, bundle["weights"])
    if breakdown is None:
        st.info("Component scores were not all stored, so the formula is not recomputed.")
        st.write("Stored composite:", record.get("composite_score"))
    else:
        weights = breakdown["weights"]
        st.write(
            f"{weights['hook']} × {breakdown['hook']} + "
            f"{weights['cliffhanger']} × {breakdown['cliffhanger']} + "
            f"{weights['standalone']} × {breakdown['standalone']} = {breakdown['recomputed']:.2f}"
        )
        st.write("Stored composite:", breakdown["stored"])
    st.subheader("Selected events")
    if not events:
        st.info("No events stored.")
    for event in events:
        st.markdown(
            f"**{event.get('start_timestamp')}–{event.get('end_timestamp')}** "
            f"{event.get('speaker_raw') or event.get('block_type')}"
        )
        st.write(event.get("text") or "(empty text)")


def screen_review() -> None:
    bundle = _require_bundle()
    if bundle is None:
        return
    st.title("Editorial review")
    st.caption("Decisions are stored beside the run. Original model scores are not changed.")
    promo_ids = [promo.get("promo_id", "") for promo in bundle["promos"]]
    counts = review_counts(Path(bundle["run_dir"]), promo_ids)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Approved", counts.get("APPROVED", 0))
    c2.metric("Rejected", counts.get("REJECTED", 0))
    c3.metric("Needs manual review", counts.get("NEEDS_MANUAL_REVIEW", 0))
    c4.metric("Pending", counts.get("PENDING", 0))
    reviews = load_reviews(Path(bundle["run_dir"]))
    for promo in bundle["promos"]:
        promo_id = promo.get("promo_id", "")
        existing = reviews.get(promo_id) or {}
        seg = promo_segment(promo)
        st.markdown(
            f"**#{promo.get('rank')} · Episode {promo.get('episode_number')} · "
            f"{seg.get('start_timestamp')}–{seg.get('end_timestamp')}**"
        )
        choice = st.radio(
            "Decision",
            ["PENDING", "APPROVED", "REJECTED", "NEEDS_MANUAL_REVIEW"],
            index=["PENDING", "APPROVED", "REJECTED", "NEEDS_MANUAL_REVIEW"].index(
                existing.get("status", "PENDING")
            ),
            key=f"decision-{promo_id}",
            horizontal=True,
        )
        comment = st.text_area(
            "Comment",
            value=existing.get("comment", ""),
            key=f"comment-{promo_id}",
        )
        if st.button("Save review", key=f"save-{promo_id}"):
            if choice == "PENDING":
                st.warning("Choose Approve, Reject, or Needs Manual Review before saving.")
            else:
                save_review(
                    Path(bundle["run_dir"]),
                    promo_id=promo_id,
                    window_id=promo.get("window_id", ""),
                    status=choice,
                    comment=comment,
                )
                st.success("Saved.")
                st.rerun()


def screen_explorer() -> None:
    bundle = _require_bundle()
    if bundle is None:
        return
    st.title("Candidate explorer")
    st.caption(
        "These rows were saved after prefilter/NMS. "
        "Windows dropped before scoring are counted in the run log and are not individually stored."
    )
    per = bundle.get("prefilter_per_episode") or []
    with st.expander("Prefilter counts"):
        if per:
            st.dataframe(per, width="stretch", hide_index=True)
        else:
            st.info("No per-episode prefilter counts in this run.")
    scored = bundle["scored"]
    if not scored:
        st.info("No scored candidates.")
        return
    episodes = sorted({(rec.get("candidate") or {}).get("episode_number") for rec in scored})
    statuses = sorted({rec.get("_status") for rec in scored})
    ep_pick = st.multiselect("Episode", [ep for ep in episodes if ep is not None], default=[ep for ep in episodes if ep is not None])
    status_pick = st.multiselect("Stage / status", statuses, default=statuses)
    reason_query = st.text_input("Rejection reason contains")
    rows = []
    for rec in scored:
        cand = rec.get("candidate") or {}
        if ep_pick and cand.get("episode_number") not in ep_pick:
            continue
        if status_pick and rec.get("_status") not in status_pick:
            continue
        reason = "; ".join(rec.get("validation_failures") or [])
        if rec.get("llm_error"):
            reason = rec["llm_error"]
        if reason_query and reason_query.lower() not in (reason or "").lower():
            continue
        rows.append(
            {
                "candidate_id": rec.get("_window_id"),
                "episode": cand.get("episode_number"),
                "start": cand.get("start_timestamp"),
                "end": cand.get("end_timestamp"),
                "composite": rec.get("composite_score"),
                "prefilter": cand.get("prefilter_score"),
                "status": rec.get("_status"),
                "reason": reason,
            }
        )
    st.write(f"{len(rows)} candidates")
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("Nothing matches.")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError:
            rows.append({"event_type": "unreadable", "error": text[:500]})
    return rows


def _bold(value: object) -> str:
    if value is None or value == "":
        return "<strong>—</strong>"
    if isinstance(value, bool):
        return f"<strong>{'yes' if value else 'no'}</strong>"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int):
        return f"<strong>{value:,}</strong>"
    return f"<strong>{html.escape(str(value))}</strong>"


def _latest_stage_events(events: list[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for event in events:
        stage = event.get("stage")
        if not stage:
            continue
        current = latest.get(stage)
        if current and current.get("event_type") == "stage_completed" and event.get("event_type") != "stage_completed":
            continue
        latest[stage] = event
    return latest


def _stage_sentence(stage: str, event: dict) -> str | None:
    info = event.get("info") or {}
    running = event.get("event_type") != "stage_completed"
    if stage == "parse":
        if running and "episodes" not in info:
            return "Reading the screenplay files."
        return (
            f"Read {_bold(info.get('files_discovered'))} screenplay files, "
            f"covering {_bold(info.get('episodes'))} episodes and "
            f"{_bold(info.get('primary_events'))} story events."
        )
    if stage == "profile":
        if running and not info:
            return "Building the show profile."
        return f"Show profile finished. Lexicon size: {_bold(info.get('lexicon_terms'))} terms."
    if stage == "candidates":
        if running and "raw_candidates" not in info:
            return "Building possible promo windows."
        return f"Built {_bold(info.get('raw_candidates'))} possible promo windows."
    if stage == "prefilter":
        if running and "after_nms" not in info:
            return f"Filtering {_bold(info.get('raw_candidates'))} windows for overlap."
        return (
            f"Kept {_bold(info.get('after_nms'))} windows after overlap filtering "
            f"and set aside {_bold(info.get('rejected_by_prefilter'))}."
        )
    if stage == "evaluate":
        if running and "evaluated" not in info:
            return "Scoring candidate clips with the model."
        return f"Scored {_bold(info.get('evaluated'))} of {_bold(info.get('total'))} candidate clips."
    if stage == "extension":
        if running and "checked" not in info:
            return "Checking whether any clip needs a longer opening."
        return (
            f"Checked {_bold(info.get('checked'))} clips for a longer opening "
            f"and extended {_bold(info.get('extended'))}."
        )
    if stage == "validate":
        if running and "validation_rejects" not in info:
            return f"Checking {_bold(info.get('candidates'))} scored clips against the hard rules."
        return (
            f"Checked {_bold(info.get('scored'))} scored clips. "
            f"Hard validation rejected {_bold(info.get('validation_rejects'))}. "
            f"Shortlist: {_bold(info.get('provisional_pool'))}."
        )
    if stage == "export":
        if running and "final_promos" not in info:
            return "Writing the final promo files."
        return f"Saved {_bold(info.get('final_promos'))} final promos."
    return None


def _deduped_error(text: str) -> str:
    parts = [part.strip() for part in text.split(";") if part.strip()]
    unique = list(dict.fromkeys(parts))
    return "; ".join(unique)


def _llm_error_reasons(run_dir: Path) -> list[str]:
    path = run_dir / "checkpoint_scored.jsonl"
    if not path.exists():
        return []
    reasons: list[str] = []
    pattern = re.compile(r'"llm_error"\s*:\s*"((?:\\.|[^"\\])*)"')
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if not match:
                continue
            try:
                raw = json.loads(f'"{match.group(1)}"')
            except json.JSONDecodeError:
                raw = match.group(1)
            cleaned = _deduped_error(str(raw))
            if cleaned and cleaned not in reasons:
                reasons.append(cleaned)
            if len(reasons) >= 8:
                break
    return reasons


def _bullet_html(items: list[str], error: bool = False) -> str:
    if not items:
        return ""
    css = ' class="log-error"' if error else ""
    rows = "".join(f"<li{css}>{item}</li>" for item in items)
    return f"<ul{css}>{rows}</ul>"


def _render_execution_summary(run_dir: Path, state: dict | None = None) -> None:
    events = _load_jsonl(run_dir / "execution_log.jsonl")
    run_log_path = run_dir / "run_log.json"
    run_log = json.loads(run_log_path.read_text(encoding="utf-8")) if run_log_path.exists() else {}
    meta = run_log.get("metadata") or {}
    quality_path = run_dir / "quality_report.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
    if not events and not run_log:
        st.info("No execution events recorded yet.")
        return

    latest = _latest_stage_events(events)
    steps: list[str] = []
    for key, _label in STAGES:
        event = latest.get(key)
        if not event:
            continue
        sentence = _stage_sentence(key, event)
        if sentence:
            if event.get("event_type") != "stage_completed":
                sentence = "In progress. " + sentence
            steps.append(sentence)

    model = meta.get("model") or (state or {}).get("model")
    header = f"Run {_bold(run_dir.name)}"
    if model:
        header += f" · model {_bold(model)}"
    calls = run_log.get("llm_call_count")
    fails = run_log.get("llm_fail_count")
    if calls is not None:
        header += f" · model calls {_bold(calls)}"
    st.markdown(header, unsafe_allow_html=True)
    if steps:
        st.markdown(_bullet_html(steps), unsafe_allow_html=True)
    elif run_log:
        st.markdown(
            _bullet_html(
                [
                    f"This run has a saved report with {_bold(run_log.get('final_promo_count'))} final promos "
                    f"and {_bold(run_log.get('llm_fail_count'))} model failures. "
                    "It has no stage-by-stage studio log."
                ]
            ),
            unsafe_allow_html=True,
        )

    errors: list[str] = []
    state_error = (state or {}).get("error")
    if state_error:
        errors.append(html.escape(str(state_error)).replace("\n", "<br>"))
    for event in events:
        info = event.get("info") or {}
        if event.get("event_type") == "unreadable":
            errors.append("Could not read one log line: " + html.escape(str(event.get("error") or "")))
        if info.get("error"):
            errors.append(html.escape(str(info["error"])))
        failed = int(info.get("llm_failed") or 0) if event.get("event_type") == "stage_completed" else 0
        if event.get("stage") == "evaluate" and event.get("event_type") == "stage_completed" and failed:
            reasons = _llm_error_reasons(run_dir)
            if reasons:
                call_note = f" across {_bold(fails)} model calls" if fails else ""
                for reason in reasons:
                    errors.append(
                        f"{_bold(failed)} candidate clips failed{call_note}. Reason: {html.escape(reason)}"
                    )
            else:
                errors.append(
                    f"{_bold(failed)} candidate clips failed. The provider did not leave a reason in the checkpoint."
                )
    for reason, count in (quality.get("hard_validation_reject_reasons") or {}).items():
        message = f"Validation rejected {_bold(count)} clips. Reason: {html.escape(str(reason))}."
        if message not in errors and "LLM scoring incomplete" not in str(reason):
            errors.append(message)
        elif "LLM scoring incomplete" in str(reason) and not any("model calls failed" in item for item in errors):
            errors.append(message)
    notes = meta.get("notes") or []
    for note in notes:
        text = str(note)
        if any(word in text.lower() for word in ("fail", "error", "skip")):
            if "skip" in text.lower() and "fail" not in text.lower() and "error" not in text.lower():
                continue
            errors.append(html.escape(text))

    unique_errors = list(dict.fromkeys(errors))
    if unique_errors:
            st.markdown(
            '<p class="log-error"><strong>Errors</strong></p>' + _bullet_html(unique_errors, error=True),
            unsafe_allow_html=True,
        )


def _render_raw_jsonl(paths: list[Path], empty_message: str) -> None:
    found = False
    for path in paths:
        if not path.exists():
            continue
        found = True
        st.caption(str(path))
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            st.code("(empty)")
            continue
        for line in text.splitlines():
            try:
                st.json(json.loads(line))
            except json.JSONDecodeError:
                st.code(line)
    if not found:
        st.info(empty_message)


def screen_logs() -> None:
    st.title("Logs")
    run_name = st.session_state.get("active_run")
    project_id = st.session_state.get("project_id")
    st.subheader("What happened")
    if not run_name:
        st.info("No active run.")
    else:
        run_dir = Path("outputs") / run_name
        state = load_execution_state(run_dir)
        _render_execution_summary(run_dir, state)
        with st.expander("Raw execution log"):
            exec_path = run_dir / "execution_log.jsonl"
            if exec_path.exists():
                st.code(exec_path.read_text(encoding="utf-8")[-4000:] or "(empty)")
            else:
                st.caption("This run has no studio execution log.")
    st.subheader("Editor actions")
    paths = []
    if run_name:
        paths.append(Path("outputs") / run_name / "audit_log.jsonl")
    if project_id:
        paths.append(UPLOAD_ROOT / project_id / "audit_log.jsonl")
    _render_raw_jsonl(paths, "No audit log yet.")


def main() -> None:
    _init()
    if st.session_state.screen == "Configure":
        pass
    _sidebar()
    screen = st.session_state.screen
    if screen == "Home":
        screen_home()
    elif screen == "New Project":
        screen_upload()
    elif screen == "Configure":
        screen_configure()
    elif screen == "Execution":
        screen_execution()
    elif screen == "Run History":
        screen_history()
    elif screen == "Final Promos":
        screen_results_and_promos()
    elif screen == "Promo Details":
        screen_details()
    elif screen == "Editorial Review":
        screen_review()
    elif screen == "Candidate Explorer":
        screen_explorer()
    elif screen == "Execution Logs":
        screen_logs()
    else:
        screen_home()


main()
