from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from backend.app.database import get_db
from backend.app.models import Camera
from backend.app.schemas import CameraCreate, CameraResponse
from backend.app.cv.camera_stream import stream_manager, scan_hardware_cameras, test_rtsp_connection

router = APIRouter(prefix="/api/cameras", tags=["cameras"])

class RTSPTestPayload(BaseModel):
    url: str

class SAHITogglePayload(BaseModel):
    enabled: bool

@router.get("", response_model=list[CameraResponse])
def get_cameras(db: Session = Depends(get_db)):
    cameras = db.query(Camera).all()
    return cameras

@router.get("/scan")
def scan_cameras():
    """
    Scans host machine for physically attached USB webcams and video capture devices.
    """
    devices = scan_hardware_cameras(max_indices=4)
    return {
        "status": "success",
        "devices_found": len(devices),
        "devices": devices
    }

@router.post("/test-stream")
def test_stream_url(payload: RTSPTestPayload):
    """
    Validates network connectivity and frame readability for an RTSP or IP camera URL.
    """
    result = test_rtsp_connection(payload.url)
    return result

@router.post("/toggle-sahi")
def toggle_sahi(payload: SAHITogglePayload):
    """
    Toggles Slicing Aided Hyper Inference (SAHI) for high-altitude drone small objects.
    """
    stream_manager.sahi_enabled = payload.enabled
    return {
        "status": "success",
        "sahi_enabled": stream_manager.sahi_enabled
    }

@router.post("", response_model=CameraResponse)
def create_camera(payload: CameraCreate, db: Session = Depends(get_db)):
    existing = db.query(Camera).filter(Camera.id == payload.id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Camera ID already exists.")

    cam = Camera(
        id=payload.id,
        name=payload.name,
        environment_type=payload.environment_type,
        source_url=payload.source_url,
        location_name=payload.location_name,
        latitude=payload.latitude,
        longitude=payload.longitude,
        is_active=payload.is_active,
        fps=payload.fps,
        resolution=payload.resolution
    )
    db.add(cam)
    db.commit()
    db.refresh(cam)
    return cam

@router.post("/{camera_id}/activate")
def activate_camera(camera_id: str, db: Session = Depends(get_db)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found.")

    # Deactivate all, activate this one
    db.query(Camera).update({"is_active": False})
    cam.is_active = True
    db.commit()

    # Inform stream manager to switch source
    stream_manager.switch_camera(
        camera_id=cam.id,
        env_type=cam.environment_type,
        name=cam.name,
        source_url=cam.source_url,
        gps=(cam.latitude, cam.longitude)
    )

    return {
        "status": "success",
        "active_camera_id": cam.id,
        "environment_type": cam.environment_type,
        "name": cam.name
    }

class CameraUpdatePayload(BaseModel):
    name: str | None = None
    source_url: str | None = None
    location_name: str | None = None
    fps: float | None = None
    resolution: str | None = None

@router.patch("/{camera_id}", response_model=CameraResponse)
def update_camera(camera_id: str, payload: CameraUpdatePayload, db: Session = Depends(get_db)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found.")

    if payload.name is not None:
        cam.name = payload.name
    if payload.source_url is not None:
        cam.source_url = payload.source_url
    if payload.location_name is not None:
        cam.location_name = payload.location_name
    if payload.fps is not None:
        cam.fps = payload.fps
    if payload.resolution is not None:
        cam.resolution = payload.resolution

    db.commit()
    db.refresh(cam)

    if cam.is_active:
        stream_manager.switch_camera(
            camera_id=cam.id,
            env_type=cam.environment_type,
            name=cam.name,
            source_url=cam.source_url,
            gps=(cam.latitude, cam.longitude)
        )

    return cam
