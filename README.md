# Real-Time Plastic Waste Detection in Water Bodies

A highly-scalable, AI-driven environmental intelligence platform designed to detect, track, quantify, and map floating and submerged plastic waste across diverse aquatic ecosystems. The platform ingests real-time video streams from multi-modal endpoints (drones, boats, underwater ROVs, and static webcams) to provide actionable insights for environmental cleanup and monitoring.

---

## 1. WHAT: System Definition, Objectives, and Capabilities

### 1.1 Project Overview
The Real-Time Plastic Waste Monitoring System is an enterprise-grade, edge-optimized artificial intelligence platform engineered to combat marine pollution. Operating on standard commodity hardware without requiring dedicated cloud GPU clusters, the system captures live video streams at 30+ frames per second (FPS), detects 5 classes of marine plastic using an optimized YOLO architecture, tracks trajectory across wave occlusions via Multi-Object Tracking (MOT), and maps accumulation zones on an interactive geospatial dashboard. 

The system is designed with a strict distinction between pure object detection and resilient tracking, acknowledging that marine environments feature chaotic visual noise (glare, turbidity, waves) that require temporal tracking to prevent duplicate counting.

### 1.2 Core System Capabilities
- **Holistic Real-Time Vision Tracking:** Continuously extracts bounding boxes, confidence scores, and centroid coordinates for 5 distinct classes of marine plastic.
- **Persistent Multi-Object Tracking (MOT):** Employs an advanced Centroid & IoU-based tracking algorithm over a sliding temporal window to assign persistent IDs, mitigating duplicate counts caused by wave occlusion.
- **Dynamic Stream Multiplexing:** Seamlessly switches between live camera feeds (Drone, Boat, Subsurface, Shoreline) without interrupting the underlying background inference loops.
- **Continuous Kinetic Auto-Segmentation:** Utilizes spatial dwell-time filters to separate transient false positives (e.g., sun glint) from actual physical plastic debris.
- **Automated Evidence Archival:** Automatically captures, watermarks, and stores high-resolution JPEG snapshots of detected waste.
- **Dynamic Model Hot-Swapping:** Allows administrators to register, upload, and swap AI models in real-time (PyTorch `.pt` or `.onnx`) via the frontend UI without server restarts.
- **Interactive Geospatial Dashboard:** A strict light-mode React dashboard providing live MJPEG feeds, real-time analytics, camera management, and alert logs.

---

## 2. WHY: Linguistic, Architectural, and Engineering Rationale

### 2.1 The Environmental Tracking Problem: Detection vs. Tracking
A fundamental error in naive pollution monitoring systems is treating every detected bounding box as a unique piece of trash. Marine environments are chaotic:
- **Wave Occlusion:** A plastic bottle bobbing in a river will disappear and reappear multiple times per minute. Simple detection will count this single bottle 50 times.
- **Sun Glare & Reflections:** Dynamic lighting on water surfaces causes rapid flickering of false positives.
- **Tracking Resolution:** By employing Multi-Object Tracking (MOT) using Euclidean centroid distance matching, this system ensures that an object is assigned a persistent ID and only counted once, yielding highly accurate statistics.

### 2.2 Edge Computing and Zero Cloud Dependencies
Monitoring lakes, rivers, and oceans often involves deploying cameras to remote locations (e.g., Dal Lake or coastal buoys) where cellular bandwidth is poor. Streaming uncompressed video to cloud APIs introduces massive latency and high recurring costs. This system executes entirely on local edge hardware: video frames are processed locally, bounding box telemetry is highly compressed, and only lightweight JSON sockets are transmitted to the dashboard.

---

## 3. HOW: End-to-End System Architecture and Workflow

```text
+----------------------------------------------------------------------------------------------------+
|                                    END-TO-END PIPELINE ARCHITECTURE                                |
+----------------------------------------------------------------------------------------------------+

  [Video Endpoints (Drone, Boat, Subsurface ROV, Webcam)]
                            |
                            v
  [Ingestion Layer (StreamManager)]
   - Thread-safe ring buffer
   - Stale frame dropping (Latency mitigation)
                            |
                            v
  [Vision Engine (PlasticDetector - YOLOv8)]
   - Bounding Boxes, Classes, Confidence
   - NMS (Non-Maximum Suppression)
   - Execution Time: < 15 ms per frame (ONNX)
                            |
                            v
  [Tracking Engine (MarineTracker)]
   - Centroid Euclidean Distance Matching
   - Intersection over Union (IoU) overlap calculation
   - Dwell Time filtering & ID persistence
                            |
           +----------------+----------------+
           |                                 |
           v                                 v
  [Event & Telemetry Bus (Pub/Sub)]  [Evidence Archival Worker]
   - WebSocket JSON Emitter           - High-res JPEG Snapshots
   - Alert Deduplication              - Timestamp Watermarking
                            |
                            v
  [React 19 Presentation Layer (Dashboard)]
   - Live MJPEG Canvas HUD
   - Chart.js Statistical Aggregation
   - Geospatial Coordinate Mapping
```

### 3.1 Step-by-Step Data Flow
1. **Frame Capture:** `camera_stream.py` acquires frames via OpenCV at 30 FPS, providing resilient drop tolerance.
2. **AI Inference:** `detector.py` executes Ultralytics YOLOv8 inference, classifying 5 plastic categories.
3. **Tracking & Debouncing:** `tracker.py` maintains spatial persistence, filtering transient noise.
4. **Event Dispatch:** `event_engine.py` applies deduplication logic to prevent log flooding.
5. **Persistence:** Detections are committed to the SQLite database via SQLAlchemy (`database.py`).
6. **Presentation:** The React dashboard (`App.tsx`) receives WebSockets, painting bounding boxes over the live MJPEG stream.

---

## 4. DETAILED TECH STACK AND MODEL SELECTION

### 4.1 Technology Stack Breakdown

| Category | Technology | Version | Purpose (What) | Architectural Rationale (Why) |
| :--- | :--- | :--- | :--- | :--- |
| **Language & Runtime** | Python | 3.11.x | Core execution runtime for computer vision and APIs. | Optimal ecosystem compatibility for OpenCV, YOLO, and FastAPI. |
| **Video I/O & Frame Ops** | OpenCV (`cv2`) | 4.11+ | RTSP acquisition, MJPEG encoding, synthetic simulators. | Industry standard providing high-throughput frame conversion and network stream ingestion. |
| **Machine Learning** | Ultralytics YOLO | 8.x | Object detection and bounding box classification. | Delivers the best balance of mAP and inference latency for edge CPU/GPU execution. |
| **Web Server & APIs** | FastAPI | 0.100+ | Asynchronous REST endpoints, MJPEG streaming, WebSockets. | Native `asyncio` support prevents blocking I/O during heavy video streaming. |
| **Database** | SQLite & SQLAlchemy | 2.0+ | Relational storage for telemetry, alerts, and configurations. | Zero-configuration setup perfect for portable, edge-deployed Docker containers. |
| **Frontend Framework** | React (TS) + Vite | 19 / 5.x | Accessible dashboard providing live video HUD and charts. | Extremely fast HMR and strict type safety syncing with Pydantic backend models. |
| **Data Visualization** | Chart.js | 4.x | Statistical views and metrics aggregation. | High-performance HTML5 canvas rendering for massive time-series datasets. |

### 4.2 Comprehensive Model Selection Analysis
**Why YOLOv8 over Mask R-CNN or SSD?**
- **Mask R-CNN:** Highly accurate polygon segmentation, but requires heavy GPU compute and operates at < 10 FPS on edge CPUs.
- **SSD (Single Shot Detector):** Fast, but struggles with small object detection (e.g., distant plastic bottles on a lake).
- **YOLOv8:** Selected because it handles varied scales well via its anchor-free architecture, and the `.onnx` export runs at 30+ FPS on commodity hardware.

---

## 5. WHERE: Codebase Directory and File Mapping

```text
d:/Real-Time Plastic Waste Monitoring System/
├── backend/app/                 
│   ├── main.py                  <- FastAPI application entrypoint, middleware, & router registry
│   ├── config.py                <- Environment variables, directory paths, global constants
│   ├── models.py                <- SQLAlchemy Database Models (Camera, Alert, Detections)
│   ├── schemas.py               <- Pydantic validation schemas for API request/response typing
│   ├── database.py              <- SQLite Engine initialization and SQLAlchemy SessionMaker
│   ├── cv/                      
│   │   ├── camera_stream.py     <- StreamManager for threading RTSP/Webcam buffers without blocking
│   │   ├── detector.py          <- YOLOv8 inference wrapper, tensor processing, and NMS logic
│   │   ├── tracker.py           <- Multi-Object Tracker (MOT) and dwell time calculation
│   │   ├── event_engine.py      <- Pub/Sub alert deduplication and snapshot generation logic
│   │   └── aquatic_simulators.py<- Synthetic wave and turbidity simulators for stress testing
│   ├── routers/                 
│   │   ├── cameras.py           <- Camera lifecycle management (CRUD operations)
│   │   ├── models.py            <- AI Model registry and hot-swapping logic
│   │   ├── stream.py            <- MJPEG video feed generator and snapshot download endpoints
│   │   └── analytics.py         <- Time-series reporting and statistical data aggregation
│   └── services/                
│       ├── background_worker.py <- Thread pool execution for non-blocking I/O tasks
│       └── storage.py           <- File system managers for handling snapshots and weight files
├── frontend/                    
│   ├── src/
│   │   ├── App.tsx              <- Main routing, state container, and layout structure
│   │   ├── index.css            <- Strict light-mode CSS design system and tokens
│   │   ├── components/          
│   │   │   ├── LiveMonitor.tsx      <- Real-time video canvas, telemetry overlays, and HUD
│   │   │   ├── AnalyticsPanel.tsx   <- Chart.js statistical views and metrics aggregation
│   │   │   ├── AlertsEvidence.tsx   <- Snapshot evidence gallery and detailed incident logs
│   │   │   └── ModelManagement.tsx  <- Dynamic AI model uploader and active model selector
│   │   └── services/api.ts      <- Axios wrappers for seamless communication with the backend
│   └── vite.config.ts           <- Vite bundler configuration and plugin setup
└── data/                        
    ├── models/                  <- Stored .pt and .onnx AI model weights
    ├── snapshots/               <- Autogenerated, timestamped JPEG evidence files
    └── marine_waste.db          <- SQLite Relational Database file
```

---

## 6. Installation, Configuration, and Operation

### 6.1 Prerequisites
- Windows 10/11 or Linux
- Python 3.11.x (Available in PATH)
- Node.js v20+ and npm

### 6.2 Setup Environment
```powershell
# Navigate to repository:
cd "d:\Real-Time Plastic Waste Monitoring System"

# Create and activate virtual environment:
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install backend dependencies:
pip install -r requirements.txt
```

### 6.3 Launching Applications
**1. Run the FastAPI Backend Server:**
```powershell
$env:PYTHONPATH="."
.\venv\Scripts\python.exe run_system.py
```
*The server will start on http://localhost:8000. API Docs available at http://localhost:8000/docs.*

**2. Run the React Vite Frontend:**
```powershell
cd frontend
npm install
npm run dev
```
*Open your browser to: http://localhost:5173*

### 6.4 Executing Automated Test Suite
```powershell
$env:PYTHONPATH="."
.\venv\Scripts\python.exe -m unittest discover tests
# 12 tests ran - OK (0 failures, 0 errors)
```

