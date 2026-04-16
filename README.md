<div align="center">

<img src="https://img.shields.io/badge/YOLOv11n-TensorRT%20INT8-00C4B4?style=for-the-badge&logo=nvidia&logoColor=white"/>
<img src="https://img.shields.io/badge/Grounding%20DINO-Zero--Shot-4A90D9?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/SAM3-Segmentation-FF6B35?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/DINOv2-Anomaly%20Detection-7B61FF?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Gemma--3%2012B-LangChain-F5A623?style=for-the-badge"/>
<img src="https://img.shields.io/badge/FastAPI-WebSocket-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/Jetson%20Orin%20Nano-Edge%20AI-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>

<br/><br/>

# Hawk-I
### AI-Powered Drone Infrastructure Inspection System

*Zero-shot defect detection · Pixel-accurate segmentation · Temporal anomaly scoring · LLM-generated reports*

<br/>

> Built for **Equinox '26** — Smart Infrastructure Track

</div>

---

## What Is Hawk-I?

Infrastructure inspection in India is still largely manual — engineers physically climbing bridges, flyovers, and buildings to look for cracks with their eyes. It's slow, inconsistent, and dangerous. Hawk-I is built to replace that.

It's an end-to-end drone inspection system where a quadcopter equipped with an **NVIDIA Jetson Orin Nano** flies over a structure, running two AI detectors simultaneously at the edge: a custom **YOLOv11n** model (TensorRT INT8, 60+ FPS) for known defect classes, and **Grounding DINO-T** (FP16) for zero-shot text-prompted detection — meaning an engineer can type any natural-language defect description and the drone finds it in real time without any retraining.

Detection payloads stream over WebSocket to a **FastAPI** backend on the ground, where **SAM 3** produces pixel-accurate segmentation masks, real-world defect area is calculated in cm² from camera intrinsics and altitude, and severity is auto-classified (L1/L2/L3). **DINOv2-B/14** compares the current inspection against stored visual embeddings of the structure's healthy baseline, flagging degradation that neither YOLO nor DINO would catch. **Gemma-3 12B via Ollama**, orchestrated through **LangChain**, generates structured inspection report entries every 30 seconds — severity level, recommended remediation, urgency timeline, and cost estimate in INR. Everything is stored in a **PostGIS** database and visualised on a live GPS-mapped **Streamlit** dashboard with one-click PDF report export.

The entire pipeline runs offline. No internet required during field inspections.

---

## Key Features

**Dual-Model Edge Detection** — Two inference threads run in parallel on the Jetson. YOLOv11n handles six trained defect classes (crack, spalling, corrosion, exposed rebar, efflorescence, vegetation encroachment) at 60+ FPS using a TensorRT INT8 engine. Grounding DINO-T runs on every 5th frame at 8–12 FPS, accepting arbitrary text queries pushed from the dashboard at runtime. Detections from both models are fused using IoU > 0.50 deduplication before transmission.

**Zero-Shot Detection via Grounding DINO** — Unlike standard object detectors, Grounding DINO uses a dual-encoder architecture (Swin Transformer image encoder + BERT-based text encoder) fused through a cross-attention Feature Enhancer. This lets an engineer type a phrase like `"rust stain . exposed rebar . white salt deposits"` into the dashboard and have the drone detect it immediately, without retraining or fine-tuning anything.

**SAM 3 Pixel-Level Segmentation** — Every bounding box from the Jetson is passed as a prompt to SAM 3 Small on the GCS backend, producing a 1920×1080 binary mask of the exact defect boundary. Pixel counts are converted to real-world area (cm²) using the Ground Sampling Distance formula derived from drone altitude and camera intrinsics (IMX477: 6.287mm sensor width, 4.74mm focal length).

**Severity Auto-Classification** — Defects are automatically tiered based on measured area against IRC (Indian Roads Congress) calibrated thresholds: L1 Minor (≤100 cm², monitor), L2 Moderate (101–500 cm², repair within 30–90 days), L3 Critical (>500 cm², immediate action). Map pins are colour-coded green/orange/red accordingly.

**DINOv2 Temporal Anomaly Detection** — DINOv2-B/14 generates 768-dimensional visual embeddings for every 10th frame. On repeat inspections, these are compared against stored baseline embeddings (from the structure's first inspection) using cosine similarity. Frames where similarity drops below 0.82 are flagged as anomalous and shown as a heat overlay on the dashboard map — catching degradation that looks visually different but doesn't match any specific defect class.

**LLM-Powered Report Generation** — A FastAPI BackgroundTask batches high-confidence (>0.60) detections every 30 seconds and sends them to Gemma-3 12B via a LangChain pipeline with a Pydantic output schema. The prompt grounds the model in IS 456:2000 and IRC 22-2015 standards and enforces structured JSON output (severity, recommended action, urgency in days, cost estimate in INR, plain-English description). Invalid responses are retried once with a correction prompt before falling back to a default template.

**WebSocket Resilience** — If the GCS connection drops mid-flight, the Jetson buffers up to 100 detections in a deque and flushes them on reconnect with exponential backoff (1s → 2s → 4s → max 30s). If the Jetson GPU hits 85°C, Grounding DINO is paused automatically while YOLO continues.

**PDF Report Export** — On mission end, all PostGIS detections are queried sorted by severity, rendered into an HTML template (defect cards, annotated images, Folium map screenshot, severity stats), and converted to a multi-page PDF with WeasyPrint.

**Offline Demo Mode** — A sidebar toggle replaces the live backend with pre-recorded detection JSON, ensuring the full software pipeline can be demonstrated even if the drone can't fly.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DRONE (Edge — Jetson Orin Nano)              │
│                                                                     │
│  IMX477 Camera → GStreamer (nvarguscamerasrc) → OpenCV frame queue  │
│                                                                     │
│  Thread 1: YOLOv11n TensorRT INT8  →  60+ FPS  (all frames)        │
│  Thread 2: Grounding DINO-T FP16   →  8-12 FPS (every 5th frame)   │
│                                                                     │
│  IoU Fusion & Deduplication → GPS attach (MAVLink/Pixhawk)         │
│              ↓ WebSocket (JSON + base64 JPEG frame)                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                  GCS BACKEND (FastAPI + Uvicorn)                    │
│                                                                     │
│  /ws/drone  ←── Jetson detections                                   │
│  /api/live/frame  ←── HTTP polling (Streamlit)                      │
│  /query  ←── text query from dashboard → forwarded to Jetson        │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌──────────────────┐  │
│  │  SAM 3   │  │ DINOv2   │  │  Gemma-3   │  │    WeasyPrint    │  │
│  │ Segment  │  │ Anomaly  │  │  LangChain │  │   PDF Export     │  │
│  │ + Area   │  │ Scoring  │  │  Reports   │  │                  │  │
│  └──────────┘  └──────────┘  └────────────┘  └──────────────────┘  │
│                          ↓                                          │
│            PostgreSQL 16 + PostGIS 3.4 (Docker)                    │
│            GIST spatial index on location (EPSG:4326)              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│               DASHBOARD (Streamlit + Folium)                        │
│  Live MJPEG feed · GPS severity map · LLM report panel · PDF DL    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## AI Model Stack

| Model | Location | Task | Input | Speed | Precision |
|---|---|---|---|---|---|
| YOLOv11n | Jetson | Fixed-class defect detection | 640×640 | 60+ FPS | TensorRT INT8 |
| Grounding DINO-T | Jetson | Zero-shot text-prompted detection | 800×800 + text | 8–12 FPS | TensorRT FP16 |
| SAM 3 Small | GCS | Pixel-level segmentation + area measurement | 1080p + bbox | ~50–80ms | FP32 |
| DINOv2-B/14 | GCS | Temporal anomaly detection via embedding similarity | 224×224 crop | ~15ms | FP32 |
| Gemma-3 12B | GCS | Structured inspection report generation | Text metadata | 2–4s/batch | via Ollama |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Edge Device | NVIDIA Jetson Orin Nano 8GB (40 TOPS) |
| Camera | Raspberry Pi IMX477 / IMX708, MIPI CSI-2, via GStreamer |
| GPS | u-blox NEO-M8N, MAVLink v2 via pymavlink, 10 Hz |
| Flight Controller | Pixhawk 6C, ArduPilot |
| Backend | FastAPI 0.115 · Uvicorn · asyncpg · WebSockets |
| Segmentation | SAM 3 Small (~400MB) |
| Anomaly Detection | DINOv2-B/14 via timm (768-dim embeddings, cosine similarity) |
| LLM Pipeline | Ollama · Gemma-3 12B · LangChain 0.3 · Pydantic |
| Database | PostgreSQL 16 + PostGIS 3.4 · Docker |
| Dashboard | Streamlit 1.38 · Folium 0.17 · Plotly 5.24 |
| PDF Generation | WeasyPrint 62 |
| Model Training | Ultralytics · Google Colab |

---

## Defect Classes (YOLOv11n)

The custom model was trained on 1,680 annotated images of Indian infrastructure defects across six classes:

| Class | Description |
|---|---|
| Crack | Surface fractures in concrete or masonry |
| Spalling | Concrete surface degradation exposing aggregate |
| Corrosion / Rust | Metal surface oxidation |
| Exposed Rebar | Visible reinforcing steel through concrete cover |
| Efflorescence | White salt deposits indicating water seepage |
| Vegetation Encroachment | Plant growth on structural surfaces |

**Training details:** mAP@0.5 of 0.45 · NMS confidence threshold 0.45 · IoU threshold 0.50 · TensorRT INT8 conversion with 500-frame calibration dataset · ~8ms inference latency on Jetson

---

## Severity Classification

Defect severity is auto-classified based on SAM 3 measured area against IRC-calibrated thresholds:

| Level | Area | Label | Pin Colour | Urgency |
|---|---|---|---|---|
| L1 | ≤ 100 cm² | Minor | Green | Monitor at next scheduled inspection |
| L2 | 101–500 cm² | Moderate | Orange | Repair within 30–90 days |
| L3 | > 500 cm² | Critical | Red | Immediate action required (<7 days) |

Area is calculated from pixel count using: `area_cm² = pixel_count × GSD²` where `GSD (cm/px) = (altitude_m × sensor_width_mm) / (focal_length_mm × image_width_px) × 100`

---

## Project Structure

```
hawk-i/
├── backend/
│   ├── main.py               # FastAPI app, WebSocket endpoints, MJPEG stream
│   ├── sam3_worker.py        # SAM 3 inference, mask extraction, area calculation
│   ├── dinov2_worker.py      # DINOv2 embeddings, baseline comparison, anomaly scoring
│   ├── llm_worker.py         # Gemma-3 LangChain pipeline, report generation
│   ├── database.py           # asyncpg connection pool, PostGIS CRUD
│   ├── report_gen.py         # WeasyPrint PDF generation
│   ├── video_stream.py       # MJPEG feed endpoint
│   └── models.py             # Pydantic schemas
├── dashboard/
│   ├── app.py                # Streamlit dashboard
│   ├── map_utils.py          # Folium map builder and pin rendering
│   └── report_template.html  # HTML/CSS template for PDF reports
├── data/
│   ├── frames/               # Saved annotated images from missions
│   ├── reports/              # Generated PDF reports
│   └── demo_detections.json  # Pre-recorded data for offline demo mode
├── scripts/                  # Utility scripts
├── tests/                    # Test files and mock drone senders
│
├── hawki_YOLOv11n_Training.ipynb   # Model training notebook
├── hawki_yolo11n.pt                # Trained YOLO weights
├── jetson_client.py                # Jetson-side WebSocket streaming client
├── gcs_client.py                   # GCS-side client utilities
├── config.py                       # Central configuration
├── docker-compose.yml              # PostgreSQL + PostGIS setup
├── requirements.txt
└── .env.example
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- NVIDIA GPU on the GCS machine (for SAM 3 and DINOv2)
- Ollama with Gemma-3 pulled: `ollama pull gemma3:12b`
- (For edge deployment) NVIDIA Jetson Orin Nano with JetPack 6.0

### Setup

```bash
git clone https://github.com/Arvoxis/hawk-i.git
cd hawk-i
pip install -r requirements.txt

cp .env.example .env
# Fill in DB credentials, ports, and thresholds

docker-compose up -d            # Start PostGIS database
python run.py                   # Start FastAPI backend
streamlit run dashboard/app.py  # Launch dashboard
```

To simulate a drone feed without hardware:

```bash
python test_fake_drone.py
```

---

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/ws/drone` | WebSocket | Receives detection payloads from Jetson |
| `/ws/dashboard` | WebSocket | Pushes live pin data and LLM entries to dashboard |
| `/query` | POST | Receives text query from dashboard, forwards to drone |
| `/detections` | GET | Returns all PostGIS detections as GeoJSON |
| `/detections/latest` | GET | Returns detections from last 5 seconds (for polling) |
| `/video_feed` | GET | MJPEG stream of latest annotated frame |
| `/report/pdf` | GET | Generates and returns PDF inspection report |
| `/health` | GET | Health check with uptime |

---

## Documentation

- [Product Requirements Document](./Hawk-I_PRD_v1.0.docx)
- [Multi-Query YOLO Setup](./MULTI_QUERY.md)
- [Model Training Notebook](./hawki_YOLOv11n_Training.ipynb)

---

<div align="center">

*Built at Equinox '26 · Smart Infrastructure Track*

</div>
