# Hawk-I — AI drone infrastructure inspection

Hackathon build (Equinox '26, ZeroDefect track). A drone streams video to a laptop/Jetson,
YOLOv11 finds defects, SAM 2 segments them, an LLM writes the inspection report, and a
Streamlit dashboard shows detections on a live GPS map.

Repo: `github.com/Arvoxis/hawk-i` (branch `master`)

## Pipeline

```
Jetson / video source ─► video_stream.py ─► processing_worker.py
                                              ├─ YOLOv11  (models/hawki_yolo11n.pt)  detect
                                              ├─ sam2_segmenter.py                   mask
                                              └─ dinov2_embedder.py                  embed
                                                      │
                          detections.jsonl / Postgres ─┤
                                                      ▼
                            llm_worker.py ─► llm_reporter.py ─► pdf_generator.py
                                                      ▼
                                       dashboard/app.py (Streamlit + folium map)
```

## Layout

| Path | What lives there |
|---|---|
| `run.py` | Top-level entry point |
| `backend/main.py` | FastAPI app, WebSocket endpoints |
| `backend/processing_worker.py` | The detect→segment→embed loop. Hot path. |
| `backend/sam2_segmenter.py`, `sam3_worker.py` | SAM 2 segmentation — see the name warning below |
| `backend/dinov2_embedder.py` | Embeddings, similarity, cross-inspection defect growth |
| `backend/llm_worker.py`, `llm_reporter.py` | Ollama report generation |
| `backend/database.py`, `schemas.py` | asyncpg + Postgres |
| `backend/config.py` | Backend config — change settings here, not inline |
| `edge/` | Jetson side: `jetson_client.py`, `multi_query_yoloworld.py`, its own `config.py` |
| `dashboard/app.py` | Streamlit UI, folium map, plotly charts |
| `scripts/` | `fake_jetson.py`, `jetson_test_sender.py`, `preflight_check.py`, `run_batch.py` |
| `tests/` | `run_all.py` plus geometry, growth, LLM-worker, fake-drone, multi-query tests |
| `models/` | Weights: `hawki_yolo11n.pt`, `sam2.1_hiera_small.pt`, `best.onnx` |
| `notebooks/` | `hawki_YOLOv11n_Training.ipynb` — how the custom weights were trained |
| `docs/` | PRD, presentation, `MULTI_QUERY.md`, and `legacy/` (pre-build planning pages) |

## There is no SAM 3

`backend/sam3_worker.py` is named after the original plan in `docs/legacy/softdev.html`
("install segment-anything-3"), but it imports `SAM2Segmenter` from `sam2_segmenter.py` and
runs **SAM 2** — as do the "SAM3" strings in `database.py` and `dashboard/utils.py`.
Renaming it would touch `main.py`, `processing_worker.py`, `database.py` and
`video_stream.py`, so the name stays. Don't let the filename mislead you.

Likewise the LLM is **Gemma-3 4B**, not 12B — the model is `config.LLM_MODEL`, default
`gemma3:4b`.

`docs/legacy/` holds the pre-build planning pages. They describe SAM 3, Gemma-3 12B,
WeasyPrint and LangChain — none of which shipped. Each carries a banner saying so. Treat
them as history, not as instructions.

## Rules

- **Never load a model inside a request handler or a frame loop.** SAM 2 and DINOv2 are
  initialised once at startup. On 6 GB VRAM, a second copy will OOM.
- Frames are dropped, not queued, when the worker is behind. Live-first — don't add
  unbounded buffering to "fix" lag.
- `detections.jsonl` is append-only. It's the crash-recovery record; don't rewrite it.
- `.env` holds the GCS and Postgres credentials and is gitignored. Keep `.env.example` current.
- `captures/`, `data/`, `reports/`, `models/*.pt` are artifacts — not commit material.
- **Env: conda `ml`** (torch+cu121, ultralytics, opencv, streamlit). The leftover
  `hawki_env/` directory is stale — ignore it, don't install into it.

## Running

```bash
python run.py                          # top-level entry
uvicorn backend.main:app --reload      # API + WebSocket
streamlit run dashboard/app.py         # dashboard
python scripts/jetson_test_sender.py   # fake a drone feed without hardware
python tests/run_all.py                # test suite
```
