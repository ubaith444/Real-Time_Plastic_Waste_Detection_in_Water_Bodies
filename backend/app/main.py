"""
Main FastAPI Application Entrypoint
AI-Based Real-Time Plastic Waste Detection System
"""

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings, SNAPSHOTS_DIR, BASE_DIR
from backend.app.database import engine, Base, SessionLocal
from backend.app.models import Camera
from backend.app.routers import cameras, detections, alerts, analytics, stream, models
from backend.app.cv.camera_stream import stream_manager

def seed_default_cameras():
    """Initializes default environment camera stations."""
    db = SessionLocal()
    default_cameras = [
        Camera(
            id="drone-01",
            name="Aerial Drone Alpha (Dal Lake)",
            environment_type="drone",
            source_url="synthetic://drone",
            location_name="Dal Lake, Srinagar, Kashmir",
            latitude=34.1250,
            longitude=74.8720,
            is_active=True,
            fps=30.0,
            resolution="1920x1080"
        ),
        Camera(
            id="boat-01",
            name="Research Vessel Seeker (Coastal Harbor)",
            environment_type="boat",
            source_url="synthetic://boat",
            location_name="Marina Bay Coastal Channel",
            latitude=13.0827,
            longitude=80.2707,
            is_active=False,
            fps=30.0,
            resolution="1280x720"
        ),
        Camera(
            id="underwater-01",
            name="Subsurface ROV Explorer (Coral Reef)",
            environment_type="underwater",
            source_url="synthetic://underwater",
            location_name="Great Barrier Marine Sanctuary",
            latitude=-16.5004,
            longitude=145.7500,
            is_active=False,
            fps=30.0,
            resolution="1280x720"
        ),
        Camera(
            id="webcam-01",
            name="Surface Test Tank / USB Webcam",
            environment_type="webcam",
            source_url="0",
            location_name="Marine Robotics Laboratory",
            latitude=12.9716,
            longitude=77.5946,
            is_active=False,
            fps=30.0,
            resolution="1280x720"
        )
    ]

    for cam in default_cameras:
        existing = db.query(Camera).filter(Camera.id == cam.id).first()
        if not existing:
            db.add(cam)
    db.commit()
    db.close()

# Ensure DB schema and seed cameras exist
Base.metadata.create_all(bind=engine)
seed_default_cameras()

# Background stream processing worker
stream_thread_running = True

def stream_worker_loop():
    print("Background video ingestion and inference worker started.")
    while stream_thread_running:
        try:
            stream_manager.process_cycle()
            time.sleep(0.033)  # ~30 FPS loop rate
        except Exception as e:
            print(f"Error in stream processing loop: {e}")
            time.sleep(0.1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables and seed default cameras
    Base.metadata.create_all(bind=engine)
    seed_default_cameras()

    # Start background ingestion thread
    global stream_thread_running
    stream_thread_running = True
    worker_thread = threading.Thread(target=stream_worker_loop, daemon=True)
    worker_thread.start()

    yield

    # Shutdown
    stream_thread_running = False

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API and real-time inference server for marine plastic waste detection.",
    lifespan=lifespan
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(cameras.router)
app.include_router(detections.router)
app.include_router(alerts.router)
app.include_router(analytics.router)
app.include_router(stream.router)
app.include_router(models.router)

@app.get("/api/health")
def api_health():
    return {
        "status": "online",
        "service": settings.app_name,
        "version": settings.app_version,
        "model": "YOLOv8-Marine-Trained",
        "sahi_batched": True
    }

# Mount frontend production build if available
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
else:
    @app.get("/")
    def health_check():
        return {
            "status": "online",
            "service": settings.app_name,
            "version": settings.app_version,
            "active_camera": stream_manager.current_camera_id,
            "fps": stream_manager.fps
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=False)
