<div align="center">

<img src="https://img.shields.io/badge/YOLOv11-Custom%20Trained-00C4B4?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/SAM2-Segmentation-FF6B35?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-WebSocket-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/Jetson%20Orin%20Nano-Edge%20Inference-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
<img src="https://img.shields.io/badge/Ollama-Gemma--3%204B-F5A623?style=for-the-badge"/>

<br/><br/>

# Hawk-I
### AI-Powered Drone Infrastructure Inspection System

*Real-time defect detection · SAM2 segmentation · LLM-generated reports · Live GPS dashboard*

<br/>

> Built for the **Equinox '26 Hackathon** — Smart Infrastructure Track  
> Edge AI · Full-Stack ML Pipeline

</div>

---

## What Is Hawk-I?

Infrastructure inspection is slow, expensive, and often dangerous — sending people up bridges, towers, and buildings to look for cracks with their eyes. Hawk-I is an attempt to fix that.

It's a full drone-based inspection system that runs a custom-trained **YOLOv11n** model on a **Jetson Orin Nano** at the edge, detecting structural defects like cracks, spalling, and corrosion in real time as the drone flies. Detected frames are streamed via WebSocket to a **FastAPI** backend where **SAM 2** generates pixel-level segmentation masks on each defect, and **Gemma-3 4B (via Ollama)** synthesizes everything into a structured inspection report. All of this gets logged to a **PostGIS-enabled PostgreSQL** database and visualized on a live GPS-mapped **Streamlit** dashboard — defect locations, confidence scores, segmentation overlays, and exportable PDF reports.

The whole pipeline runs in real time. No post-flight processing, no manual review step.

---

## Key Features

**Edge Inference on Jetson Orin Nano** — The drone runs YOLOv11n locally, so detection happens on-device before anything is sent over the network. This keeps latency low and makes the system viable even with limited bandwidth.

**Custom-Trained Defect Detection Model** — YOLOv11n trained from scratch on 1,680 annotated concrete defect images across four classes: cracks, spalling, corrosion, and delamination. Achieves mAP@0.5 of 0.45 on the validation set. Weights are included in the repo (`hawki_yolo11n.pt`).

**SAM 2 Segmentation** — Every YOLO bounding box gets passed to SAM 2 for precise pixel-level segmentation. This gives far more useful output than a bounding box alone — you can actually see the shape and extent of each defect.

**LLM-Generated Inspection Reports** — The backend aggregates detections per session and prompts Gemma-3 4B to generate a structured inspection report: defect types, severity assessment, location context, and recommendations. Reports are exportable as PDF.

**Live GPS Dashboard** — The Streamlit dashboard shows the drone's real-time position on a map with defect markers, a live detection feed, and per-session history. Built on HTTP polling (`/api/live/frame`) rather than WebSocket to keep the Streamlit state model stable.

**Session-Isolated Database** — Each inspection flight creates its own session-specific tables in PostGIS, keeping data clean and queryable by location using geospatial queries.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DRONE (Edge)                             │
│  Camera → Jetson Orin Nano → YOLOv11n Inference                 │
│              ↓ WebSocket Stream                                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                    BACKEND (FastAPI)                             │
│                                                                 │
│  /api/live/frame  ←── HTTP Polling (Streamlit)                  │
│  WebSocket Endpoint ←── Jetson Client                           │
│                                                                 │
│  ┌────────────┐   ┌──────────┐   ┌──────────────────────────┐  │
│  │  YOLOv11n  │ → │   SAM 2  │ → │  Ollama (Gemma-3 4B)     │  │
│  │  Detection │   │  Segment │   │  Report Generation       │  │
│  └────────────┘   └──────────┘   └──────────────────────────┘  │
│                       ↓                                         │
│            PostgreSQL + PostGIS (Docker)                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                  DASHBOARD (Streamlit)                           │
│  Live GPS Map · Defect Feed · Session Reports · PDF Export      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Edge Device | NVIDIA Jetson Orin Nano |
| Object Detection | Custom YOLOv11n · Ultralytics |
| Segmentation | SAM 2 (Segment Anything Model 2) |
| LLM Reports | Ollama · Gemma-3 4B |
| Backend | FastAPI · WebSockets · HTTP Polling |
| Database | PostgreSQL + PostGIS · Docker |
| Dashboard | Streamlit |
| Model Training | Ultralytics · Google Colab |

---

## Project Structure

```
hawk-i/
├── backend/                  # FastAPI app, WebSocket endpoints, inference pipeline
├── dashboard/                # Streamlit dashboard
├── data/                     # Dataset & annotations
├── scripts/                  # Utility scripts
├── tests/                    # Test files & mock drone senders
│
├── hawki_YOLOv11n_Training.ipynb   # Model training notebook
├── hawki_yolo11n.pt                # Trained YOLO weights
├── jetson_client.py                # Jetson-side streaming client
├── receiver.py                     # Backend frame receiver
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
- Ollama with Gemma-3 4B pulled (`ollama pull gemma3:4b`)
- NVIDIA GPU (for SAM 2 and YOLO inference)

### Setup

```bash
git clone https://github.com/Arvoxis/hawk-i.git
cd hawk-i
pip install -r requirements.txt

cp .env.example .env
# Fill in your DB credentials and config

docker-compose up -d            # Start PostGIS database
python run.py                   # Start FastAPI backend
streamlit run dashboard/app.py  # Launch dashboard
```

To simulate a drone feed without hardware:

```bash
python test_fake_drone.py
```

---

## Model Details

| Attribute | Value |
|---|---|
| Architecture | YOLOv11n (nano) |
| Dataset | 1,680 annotated concrete defect images |
| Classes | Cracks · Spalling · Corrosion · Delamination |
| mAP@0.5 | 0.45 |
| Training Platform | Google Colab · Ultralytics |
| Weights | `hawki_yolo11n.pt` |

Full training process in [`hawki_YOLOv11n_Training.ipynb`](./hawki_YOLOv11n_Training.ipynb)

---

## Documentation

- [Product Requirements Document](./Hawk-I_PRD_v1.0.docx)
- [Hackathon Presentation](./HAWKI_Presentation.pptx)
- [Multi-Query YOLO Setup](./MULTI_QUERY.md)

---

<div align="center">

*Built at Equinox '26 · Smart Infrastructure Track*

</div>
