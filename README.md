# Real-Time Plastic Waste Detection in Water Bodies

A highly-scalable, AI-driven environmental intelligence platform designed to detect, track, quantify, and map floating and submerged plastic waste across diverse aquatic ecosystems. The platform ingests real-time video streams from multi-modal endpoints (drones, boats, underwater ROVs, and static webcams) to provide actionable insights for environmental cleanup and monitoring.

---

## 1. WHAT: System Definition, Objectives, and Capabilities

**System Definition:**  
An end-to-end computer vision and stream-event processing pipeline designed for the edge. It combines real-time Object Detection, Multi-Object Tracking (MOT), and Geospatial Telemetry to transform unstructured video feeds into structured, actionable environmental data. The system operates autonomously, identifying pollutants in challenging aquatic environments characterized by waves, glare, and turbidity.

**Primary Objectives:**
- **Automate Environmental Monitoring:** Replace manual, labor-intensive observation with scalable, 24/7 computer vision that does not suffer from fatigue.
- **Ontological Standardization:** Categorize marine waste into a unified ontology to provide consistent data for environmental policymakers and cleanup crews.
- **Resource Optimization:** Direct autonomous surface vehicles (ASVs) or manual cleanup crews to precise coordinates where plastic accumulation hotspots are actively forming.
- **Continuous Operation:** Ensure fault-tolerant monitoring capabilities regardless of the deployment vector (air, surface, or underwater) and without reliance on constant high-bandwidth cloud connectivity.

**Core Capabilities:**
- **High-Frequency Real-Time Inference:** Processes RTSP network streams, USB cameras, and synthetic video feeds at 30+ frames per second using highly optimized YOLOv8 weights (PyTorch/ONNX).
- **Persistent Multi-Object Tracking:** Employs an advanced Centroid & IoU-based tracking algorithm to assign persistent IDs to floating waste. This mitigates duplicate counts caused by wave occlusion (e.g., when a bottle bobs underwater and resurfaces).
- **Dynamic Stream Multiplexing:** Seamlessly switches between live camera feeds (Drone, Boat, Subsurface, Shoreline) on the fly without interrupting the underlying background inference loops.
- **Automated Evidence Archival:** Automatically captures, watermarks, and stores high-resolution JPEG snapshots of detected waste. These serve as verifiable proof for environmental reporting.
- **Dynamic Model Hot-Swapping:** Allows administrators to register, upload, and swap AI models in real-time via the frontend UI without requiring backend restarts or service downtime.
- **Interactive Geospatial Dashboard:** A React-based interface providing live MJPEG feeds, real-time analytics, camera management, and alert logs.

---

## 2. WHY: Linguistic, Architectural, and Engineering Rationale

**Architectural Rationale:**
- **Edge-First Design:** The system is explicitly built to operate in resource-constrained environments (e.g., a Raspberry Pi or Jetson Nano aboard a research vessel or remote shore station). Utilizing lightweight models like YOLOv8 Nano (`yolov8n`), SQLite for local storage, and a Vite-bundled React frontend ensures ultra-low memory footprints and eliminates the need for expensive cloud compute.
- **Decoupled Processing (Pub/Sub):** The computer vision inference engine operates in a completely separate, asynchronous thread from the REST API and WebSocket servers. Telemetry data (bounding boxes, FPS, drift trajectories) is dispatched via an in-memory Event Engine. This prevents slow consumers (like database writes or slow network clients) from bottlenecking the 30 FPS video ingestion loop.
- **Resilient Tracking over Simple Detection:** Marine environments are visually chaotic. Waves obscure objects, sun glare disrupts frame continuity, and boats induce camera roll. Simple frame-by-frame detection would result in wildly inaccurate counts. Multi-Object Tracking (MOT) ensures that temporal context is maintained, yielding highly accurate statistics over time.

**Engineering Choices:**
- **FastAPI over Django/Flask:** Required for native asynchronous support (WebSockets) and high-throughput routing essential for real-time telemetry streaming.
- **SQLite over PostgreSQL:** Chosen for its zero-configuration portability, which is critical for remote edge deployments where setting up dedicated database servers is impractical.
- **Vanilla CSS over Tailwind/Bootstrap:** Ensures a lightweight frontend bundle and total control over the design system, tailored specifically for high-contrast outdoor visibility.

---

## 3. HOW: End-to-End System Architecture and Workflow

The system relies on a modular, event-driven architecture designed to process high-throughput video streams autonomously.

### The Pipeline Workflow

1. **Ingestion Layer (`StreamManager`):**  
   Continuously reads frames from active camera endpoints (RTSP streams, simulated synthetic environments, or local webcams) into a thread-safe ring buffer, dropping stale frames to prevent latency buildup.
2. **Vision Engine (`PlasticDetector`):**  
   Pre-processes frames (resizing, normalization) and passes them to the Ultralytics YOLOv8 inference engine. Bounding boxes, confidence scores, and class labels are extracted.
3. **Tracking Engine (`MarineTracker`):**  
   Maps incoming bounding boxes to persistent spatial IDs based on Euclidean distance and Intersection over Union (IoU). Evaluates object dwell time to filter out transient false positives.
4. **Event & Telemetry Bus (`EventEngine`):**  
   Applies deduplication logic to prevent log flooding (e.g., only one alert per track ID). Validated detections trigger a background process to save high-res JPEG snapshots to disk. Telemetry is simultaneously emitted via WebSockets to the frontend.
5. **Presentation Layer (React Dashboard):**  
   The client subscribes to the WebSocket feed to paint bounding boxes over the live MJPEG stream, updates analytical charts in real-time, and logs incidents in the Evidence gallery.

---

## 4. DETAILED TECH STACK AND MODEL SELECTION

### Backend Stack
- **Framework:** FastAPI (Python 3.11+) - Asynchronous REST APIs, MJPEG streaming responses, and WebSocket telemetry.
- **Database:** SQLite3 paired with SQLAlchemy ORM - Portable, relational schema for storing camera configurations, alerts, and analytics.
- **Computer Vision:** OpenCV (`cv2`) - Handles hardware video capture, color space conversion, and frame drawing.
- **AI Inference Engine:** Ultralytics YOLO (`ultralytics`) - PyTorch-based execution of `.pt` and `.onnx` models.
- **Architecture Patterns:** Threading, Pub/Sub Event Bus, non-blocking Background Workers.

### Frontend Stack
- **Framework:** React 19 (TypeScript) - Component-driven architecture with strict type safety.
- **Bundler:** Vite - Extremely fast HMR (Hot Module Replacement) and optimized production builds.
- **Data Visualization:** Chart.js (`react-chartjs-2`) - High-performance canvas rendering for time-series analysis and telemetry graphs.
- **Icons & UI:** Lucide-React for crisp, scalable vector icons.
- **API Communication:** Axios with standardized interceptors and typed response handling.

### AI Model Strategy (YOLOv8)
The core model is an Ultralytics YOLOv8 edge-optimized model, selected for its superior balance of speed (latency) and mean Average Precision (mAP). 
- **Ontology (5 Unified Classes):** 
  - `plastic_bottle`: Beverage bottles and caps.
  - `plastic_bag`: Wrappers, films, single-use polythene bags.
  - `plastic_packaging`: Styrofoam containers, tetra packs, cups.
  - `plastic_fragment`: Rigid shards, microplastics, and macroplastics.
  - `other_floating_waste`: Ghost nets, ropes, buckets, unclassified debris.
- **Optimization:** Supports dynamic model swapping between standard PyTorch (`.pt`) for rapid prototyping and highly optimized ONNX (`.onnx`) formats for production edge inference.

---

## 5. WHERE: Codebase Directory and File Mapping

The repository is structured to separate the inference engine from the web server, ensuring maintainability and scalability.

```text
/
├── backend/app/                 # Python FastAPI Backend
│   ├── main.py                  # FastAPI application entrypoint, middleware, & router registry
│   ├── config.py                # Environment variables, directory paths, global constants
│   ├── models.py                # SQLAlchemy Database Models (Camera, Alert, Detections)
│   ├── schemas.py               # Pydantic validation schemas for robust API request/response typing
│   ├── database.py              # SQLite Engine initialization and SQLAlchemy SessionMaker
│   ├── cv/                      # Core Computer Vision & AI Inference Engine
│   │   ├── camera_stream.py     # StreamManager for threading RTSP/Webcam buffers without blocking
│   │   ├── detector.py          # YOLOv8 inference wrapper, tensor processing, and NMS logic
│   │   ├── tracker.py           # Multi-Object Tracker (MOT) and dwell time calculation
│   │   ├── event_engine.py      # Pub/Sub alert deduplication and snapshot generation logic
│   │   └── aquatic_simulators.py# Synthetic wave and turbidity simulators for stress testing
│   ├── routers/                 # FastAPI Endpoints (Controllers)
│   │   ├── cameras.py           # Camera lifecycle management (CRUD operations)
│   │   ├── models.py            # AI Model registry and hot-swapping logic
│   │   ├── stream.py            # MJPEG video feed generator and snapshot download endpoints
│   │   └── analytics.py         # Time-series reporting and statistical data aggregation
│   └── services/                # Background Tasks & Utilities
│       ├── background_worker.py # Thread pool execution for non-blocking I/O tasks
│       └── storage.py           # File system managers for handling snapshots and weight files
│
├── frontend/                    # React Vite Frontend Application
│   ├── src/
│   │   ├── App.tsx              # Main routing, state container, and layout structure
│   │   ├── index.css            # Global CSS design system, typography, and utility classes
│   │   ├── components/          # Dashboard UI Components
│   │   │   ├── LiveMonitor.tsx      # Real-time video canvas, telemetry overlays, and controls
│   │   │   ├── AnalyticsPanel.tsx   # Chart.js statistical views and metrics aggregation
│   │   │   ├── AlertsEvidence.tsx   # Snapshot evidence gallery and detailed incident logs
│   │   │   └── ModelManagement.tsx  # Dynamic AI model uploader and active model selector
│   │   ├── services/api.ts      # Axios wrappers for seamless communication with the backend
│   │   └── types.ts             # Global TypeScript interfaces syncing with Pydantic schemas
│   └── vite.config.ts           # Vite bundler configuration and plugin setup
│
└── data/                        # Persistent Local Storage (Ignored by Git)
    ├── models/                  # Stored .pt and .onnx AI model weights
    ├── snapshots/               # Autogenerated, timestamped JPEG evidence files
    └── marine_waste.db          # SQLite Relational Database file
```
