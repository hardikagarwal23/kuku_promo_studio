# Kuku PS1 — Narrative-Aware Promo Finder

Finds ranked 20–60 second promotional clip recommendations from microdrama screenplay text. It does **not** edit video, stitch cross-episode segments, or judge real acting/cinematography beyond what the screenplay describes.

## Setup

```bash
cd Promo
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env with LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
```

Place confidential screenplay `.txt` files in `data/input/` (gitignored). Do **not** commit Kuku’s pack.

## Run

```bash
# Full pack with real LLM
python -m src.run --input-dir data/input

# Resume same run after interruption
python -m src.run --input-dir data/input --resume run_YYYY-MM-DD_HHMMSS

# UI
streamlit run app.py
```

## Pipeline stages

1. Parse screenplays → structured events (+ camera metadata)
2. Show profile (LLM)
3. Generate continuous 20–60s event-boundary candidates
4. Recall-first prefilter + temporal NMS (IoS ≥ 0.35)
5. Blind standalone + contextual hook/cliffhanger scorers (concurrent)
6. Conditional contiguous start extension (at most once)
7. Hard validation + deterministic weighted ranking, capped at 2 promos per episode
8. Export JSON/CSV/reports + optional Streamlit review UI

Composite score: `0.40*hook + 0.35*cliffhanger + 0.25*standalone`. Max 2 promos per episode. Final shortlist up to 15.

## Outputs

Each run writes `outputs/<run_id>/` including `promos.json`, `promos.csv`, `all_scored_candidates.csv`, `failures.md`, `quality_report.json`, `parser_report.json`, `show_profile.json`, `checkpoint_scored.jsonl`, and `raw_llm_responses/*.jsonl`.

## Limits

- Text-only analysis; human editors must check footage.
- Event-boundary windows may miss ideal in-event cut points.
- Heuristic prefilter and LLM scores are subjective.
- MVP: one continuous segment per promo; no multi-segment stitching.
