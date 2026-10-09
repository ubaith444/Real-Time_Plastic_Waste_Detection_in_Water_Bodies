# Real-Time Plastic Waste Detection in Water Bodies

A highly-scalable, AI-driven environmental intelligence platform designed to detect, track, quantify, and map floating and submerged plastic waste across diverse aquatic ecosystems. The platform ingests real-time video streams from multi-modal endpoints (drones, boats, underwater ROVs, and static webcams) to provide actionable insights for environmental cleanup and monitoring.

---

## 1. WHAT: System Definition, Objectives, and Capabilities

**System Definition:**  
An end-to-end computer vision and stream-event processing pipeline combining real-time Object Detection, Multi-Object Tracking (MOT), and Geospatial Telemetry. 

**Objectives:**
- Automate the manual, labor-intensive process of monitoring aquatic environments for plastic pollution.
- Standardize marine waste classification into a unified ontology.
- Provide continuous 24/7 observation capabilities regardless of the deployment vector (air, surface, or underwater).

**Capabilities:**
- **Real-Time Inference:** Processes RTSP, USB, and synthetic video streams at 30+ FPS using optimized YOLOv8 weights (PyTorch/ONNX).
- **Persistent Tracking:** Employs a Centroid/IoU tracking algorithm to assign persistent IDs to floating waste, mitigating duplicate counts caused by wave occlusion.
- **Dynamic Stream Multiplexing:** Seamlessly switches between live camera feeds (Drone, Boat, Subsurface, Shoreline) without interrupting inference loops.
- **Evidence Archival:** Automatically captures, watermarks, and stores high-resolution snapshots of detected waste.
- **Dynamic Model Hot-Swapping:** Allows administrators to register and swap AI models in real-time without backend restarts.

---

## 2. WHY: Linguistic, Architectural, and Engineering Rationale

**Why this Architecture?**
- **Decoupled Processing (Pub/Sub):** The inference engine operates asynchronously from the API and frontend. Telemetry data (bounding boxes, FPS, drift trajectories) is dispatched via an Event Engine, preventing slow consumers (like database writes) from bottlenecking the 30 FPS video ingestion loop.
- **Edge-First Design:** The system is built to operate in resource-constrained environments (e.g., aboard a research vessel or remote shore station). Using lightweight models like YOLOv8 Nano (`yolov8n`), SQLite, and a Vite-bundled React frontend ensures low memory footprints.
- **Resilient Tracking over Simple Detection:** Marine environments are chaotic. Waves obscure objects, and sun glare disrupts frame continuity. Tracking (MOT) ensures that a plastic bottle temporarily submerged by a wave isn't counted twice when it resurfaces.

---

## 3. HOW: End-to-End System Architecture and Workflow

The system relies on a modular, event-driven architecture designed to process high-throughput video streams.

1. **Ingestion Layer (`StreamManager`):**  
   Reads frames from active camera endpoints (RTSP streams, simulated synthetic environments, or local webcams) into a continuous ring buffer.
2. **Vision Engine (`PlasticDetector`):**  
   Pre-processes frames and passes them to the Ultralytics YOLOv8 inference engine. Bounding boxes, confidence scores, and class labels are extracted.
3. **Tracking Engine (`MarineTracker`):**  
   Maps bounding boxes to persistent spatial IDs. Evaluates object dwell time and filters out false positives.
4. **Event & Telemetry Bus (`EventEngine`):**  
   Deduplicates alerts to prevent log flooding. Validated detections trigger a background process to save high-res JPEG snapshots. Telemetry is emitted via WebSockets to the frontend.
5. **Presentation Layer (React Dashboard):**  
   Subscribes to the WebSocket feed to paint bounding boxes over the live MJPEG stream, update analytical charts, and map detections on a geospatial UI.

---

## 4. DETAILED TECH STACK AND MODEL SELECTION

### Backend Stack
- **Framework:** FastAPI (Python 3.11+) for high-performance asynchronous REST routes and WebSockets.
- **Database:** SQLite with SQLAlchemy ORM (Portable, zero-configuration setup for telemetry and alerts).
- **Computer Vision:** OpenCV (Video processing, MJPEG encoding) and Ultralytics (YOLO object detection).
- **Architecture Patterns:** Pub/Sub Event Bus, Background Workers.

### Frontend Stack
- **Framework:** React 19 (TypeScript), bundled with Vite.
- **Styling:** Custom CSS Design System optimized for a clean, professional, light-theme interface.
- **Data Visualization:** Chart.js for time-series analysis and telemetry graphs.
- **Icons:** Lucide-React.

### AI Model Strategy (YOLOv8)
The core model is an Ultralytics YOLOv8 edge-optimized model. 
- **Ontology (5 Classes):** 
  - `plastic_bottle`: Beverage bottles and caps.
  - `plastic_bag`: Wrappers, films, single-use bags.
  - `plastic_packaging`: Styrofoam containers, tetra packs.
  - `plastic_fragment`: Rigid shards and macroplastics.
  - `other_floating_waste`: Nets, ropes, unclassified debris.
- **Optimization:** Supports dynamic model swapping between standard PyTorch (`.pt`) and highly optimized ONNX (`.onnx`) formats.

---

## 5. WHERE: Codebase Directory and File Mapping

```text
/
├── backend/app/                 # Python FastAPI Backend
│   ├── main.py                  # FastAPI application entrypoint & middleware configuration
│   ├── config.py                # Environment variables, directory paths, global settings
│   ├── models.py                # SQLAlchemy Database Models (Camera, Alert, Detections)
│   ├── schemas.py               # Pydantic validation schemas for API requests/responses
│   ├── database.py              # SQLite Engine and SessionMaker initialization
│   ├── cv/                      # Core Computer Vision Engine
│   │   ├── camera_stream.py     # StreamManager for reading RTSP/Webcam buffers
│   │   ├── detector.py          # YOLOv8 inference wrapper and tensor processing
│   │   ├── tracker.py           # Multi-Object Tracker (MOT) and dwell time logic
│   │   ├── event_engine.py      # Pub/Sub alert deduplication and snapshot generation
│   │   └── aquatic_simulators.py# Synthetic wave and turbidity simulators for testing
│   ├── routers/                 # API Endpoints
│   │   ├── cameras.py           # Camera lifecycle management
│   │   ├── models.py            # AI Model registry and hot-swapping
│   │   ├── stream.py            # MJPEG video feeds and snapshot downloads
│   │   └── analytics.py         # Time-series reporting and statistical aggregation
│   └── services/                # Background Tasks
│       ├── background_worker.py # Thread pool for non-blocking I/O tasks
│       └── storage.py           # File system managers for snapshots and weights
│
├── frontend/                    # React Vite Frontend
│   ├── src/
│   │   ├── App.tsx              # Main routing and global state container
│   │   ├── index.css            # Global CSS design system and tokens
│   │   ├── components/          # Dashboard UI Components
│   │   │   ├── LiveMonitor.tsx      # Real-time video canvas and bounding boxes
│   │   │   ├── AnalyticsPanel.tsx   # Chart.js statistical views
│   │   │   ├── AlertsEvidence.tsx   # Snapshot gallery and incident logs
│   │   │   └── ModelManagement.tsx  # Dynamic AI model uploader and active model selector
│   │   ├── services/api.ts      # Axios wrappers for communicating with the backend
│   │   └── types.ts             # Global TypeScript interfaces matching Pydantic schemas
│   └── vite.config.ts           # Vite bundler configuration
│
└── data/                        # Persistent Storage
    ├── models/                  # Stored .pt and .onnx model weights
    ├── snapshots/               # Autogenerated JPEG evidence files
    └── marine_waste.db          # SQLite Database file
```

---

## 6. Global Skill

This repository serves as a foundational "Global Skill" template for high-throughput, real-time edge computer vision applications. It demonstrates:
- **Asynchronous Video Pipelines:** Processing streams concurrently without blocking HTTP/WebSocket layers.
- **Robust Abstractions:** Decoupling the AI detection loop from tracking and alerting, allowing for modular upgrades (e.g., swapping YOLO for another architecture without touching the tracker).
- **Deployment Agnosticism:** Running independently from the cloud, perfectly suited for remote edge deployments in aquatic environments where internet access is intermittent.
