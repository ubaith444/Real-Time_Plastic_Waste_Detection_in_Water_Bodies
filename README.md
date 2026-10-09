# Real-Time Plastic Waste Monitoring System

## Why
Marine plastic pollution is a critical environmental crisis. This AI-powered platform provides environmental intelligence to detect, track, quantify, and map floating and submerged plastic waste across various water bodies. By transforming unstructured video feeds into actionable data, it enables automated monitoring and faster cleanup responses, replacing manual observation with scalable, 24/7 computer vision.

## Where
Designed for versatile deployment across diverse aquatic environments:
- **Coastal Harbors:** Using boat-mounted cameras to navigate waves and monitor marine surfaces.
- **Lakes & Rivers:** Utilizing high-altitude aerial drones (e.g., Dal Lake) and static shoreline webcams to monitor currents and identify accumulation hotspots.
- **Oceans & Reefs:** Deploying subsurface underwater ROVs (e.g., Great Barrier Reef) to detect submerged waste despite turbidity and light attenuation.

## Model
The core inference engine is powered by **Ultralytics YOLOv8**.
- **Architecture:** Optimized for real-time edge processing, capable of running inference at 30+ FPS. Supports PyTorch (`.pt`) and ONNX (`.onnx`) weights.
- **Classes:** Unified Marine Plastic Ontology detecting 5 categories: `plastic_bottle`, `plastic_bag`, `plastic_packaging`, `plastic_fragment`, and `other_floating_waste`.
- **Capabilities:**
  - Real-time bounding box detection with dynamic NMS and confidence thresholds.
  - Multi-Object Tracking (MOT) using Centroid/IoU logic to maintain track IDs across wave occlusions and calculate dwell times.
  - Hot-swappable model management via a built-in import registry, allowing real-time switching without server restarts.

## Tech Stacks
- **Frontend:** React 19, TypeScript, Vite, CSS (Custom Design System with a modern, clean UI), Chart.js for analytics, Lucide-React for iconography.
- **Backend:** Python 3.11, FastAPI (Asynchronous REST & WebSockets).
- **Computer Vision:** OpenCV (Video Ingestion & Synthetic Aquatic Simulators), Ultralytics (YOLO inference).
- **Database:** SQLite & SQLAlchemy ORM for lightweight, portable telemetry and alert storage.
- **Architecture Patterns:** Pub/Sub Event Engine, Background Workers for alert deduplication and high-res snapshot evidence capture.
