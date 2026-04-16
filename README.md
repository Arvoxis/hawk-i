<div align="center">

<img src="https://img.shields.io/badge/YOLOv11-Custom%20Trained-00C4B4?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/SAM2-Segmentation-FF6B35?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-WebSocket-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/Jetson%20Orin%20Nano-Edge%20Inference-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
<img src="https://img.shields.io/badge/Ollama-Gemma--3%204B-F5A623?style=for-the-badge"/>

<br/><br/>

# 🦅 Hawk-I
### AI-Powered Drone Infrastructure Inspection System

*Real-time defect detection · SAM2 segmentation · LLM-generated reports · Live GPS dashboard*

<br/>

> Built for the **Equinox '26 Hackathon** — ZeroDefect Track  
> Smart Infrastructure · Edge AI · Full-Stack ML Pipeline

</div>

---

## 🧠 What Is Hawk-I?

Hawk-I is an end-to-end drone-based infrastructure inspection system that detects, segments, and reports structural defects in real time. A drone equipped with a **Jetson Orin Nano** streams live video to a backend that runs a custom-trained **YOLOv11n** model for defect detection, **SAM 2** for precise segmentation masks, and an **LLM (Gemma-3 4B via Ollama)** to auto-generate structured inspection reports — all visualized on a live GPS-mapped Streamlit dashboard.

No manual inspection. No post-processing delays. Just fly, detect, and report.

---

## ⚡ Key Features

- 🔍 **Real-Time Defect Detection** — Custom YOLOv11n model trained on 1,680 concrete defect images (mAP 0.45), detecting cracks, spalling, corrosion, and more
- 🎯 **SAM 2 Segmentation** — Precise pixel-level masks overlaid on detected defect regions
- 🤖 **LLM Inspection Reports** — Gemma-3 4B via Ollama auto-generates structured reports per inspection session
- 🗺️ **Live GPS Dashboard** — Real-time map showing drone position and annotated defect locations
- 📡 **Edge-to-Cloud Streaming** — Jetson Orin Nano handles on-device inference, streams frames via WebSocket to a FastAPI backend
- 📄 **PDF Report Export** — One-click downloadable inspection reports per session
- 🐳 **Dockerized PostGIS DB** — Geospatial PostgreSQL for storing session data, defect coordinates, and inspection history

---

## 🏗️ System Architecture

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

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Edge Inference | NVIDIA Jetson Orin Nano, YOLOv11n |
| Object Detection | Custom YOLOv11n (trained on concrete defect dataset) |
| Segmentation | SAM 2 (Segment Anything Model 2) |
| LLM Reports | Ollama · Gemma-3 4B |
| Backend | FastAPI · WebSockets · HTTP Polling |
| Database | PostgreSQL + PostGIS · Docker |
| Dashboard | Streamlit |
| Geospatial | PostGIS · GPS coordinate mapping |
| Model Training | Ultralytics · Google Colab |

---

## 📁 Project Structure

```
hawk-i/
├── backend/                  # FastAPI backend, WebSocket endpoints
├── dashboard/                # Streamlit dashboard app
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

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Ollama installed with Gemma-3 4B pulled (`ollama pull gemma3:4b`)
- NVIDIA GPU (for SAM 2 and YOLO inference)

### 1. Clone & Install

```bash
git clone https://github.com/Arvoxis/hawk-i.git
cd hawk-i
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your DB credentials and config
```

### 3. Start the Database

```bash
docker-compose up -d
```

### 4. Run the Backend

```bash
python run.py
```

### 5. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

### 6. Simulate Drone Feed (Testing)

```bash
python test_fake_drone.py
```

---

## 🧪 Model Details

| Attribute | Value |
|---|---|
| Architecture | YOLOv11n (nano) |
| Dataset | 1,680 concrete defect images |
| Classes | Cracks, Spalling, Corrosion, Delamination |
| mAP@0.5 | 0.45 |
| Training | Google Colab · Ultralytics |
| Weights | `hawki_yolo11n.pt` |

Training notebook: [`hawki_YOLOv11n_Training.ipynb`](./hawki_YOLOv11n_Training.ipynb)

---

## 📄 Documentation

- [Product Requirements Document](./Hawk-I_PRD_v1.0.docx)
- [Hackathon Presentation](./HAWKI_Presentation.pptx)
- [Multi-Query YOLO Setup](./MULTI_QUERY.md)

---

## 🙋 Author

**Rakshit Sinha**  
B.Tech Computer Science · VIT Vellore ('28)  
Vice Chairperson, AI & ML Club (TAM-VIT)

[![GitHub](https://img.shields.io/badge/GitHub-Arvoxis-181717?style=flat-square&logo=github)](https://github.com/Arvoxis)

---

<div align="center">

*Built in 24 hours at Equinox '26 · ZeroDefect Track*

</div>