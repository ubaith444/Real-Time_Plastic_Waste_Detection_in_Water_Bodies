import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from backend.app.cv.camera_stream import stream_manager
from backend.app.services.storage import StorageService

router = APIRouter(tags=["stream"])

class StreamConfigUpdate(BaseModel):
    confidence_threshold: float | None = None
    iou_threshold: float | None = None
    suppression_window_sec: float | None = None

@router.post("/api/stream/config")
def update_stream_config(payload: StreamConfigUpdate):
    if payload.confidence_threshold is not None:
        stream_manager.detector.confidence_threshold = max(0.1, min(0.99, payload.confidence_threshold))
    if payload.iou_threshold is not None:
        stream_manager.detector.iou_threshold = max(0.1, min(0.99, payload.iou_threshold))
        stream_manager.tracker.iou_threshold = payload.iou_threshold
    if payload.suppression_window_sec is not None:
        stream_manager.event_engine.suppression_window_sec = max(1.0, payload.suppression_window_sec)

    return {
        "status": "success",
        "confidence_threshold": stream_manager.detector.confidence_threshold,
        "iou_threshold": stream_manager.detector.iou_threshold,
        "suppression_window_sec": stream_manager.event_engine.suppression_window_sec
    }

@router.get("/api/snapshots/{filename}")
def get_snapshot_image(filename: str):
    path = StorageService.get_snapshot_path(filename)
    if not path:
        raise HTTPException(status_code=404, detail="Snapshot not found.")
    return FileResponse(str(path), media_type="image/jpeg")

@router.get("/api/stream/snapshot")
def get_current_stream_snapshot():
    """
    Downloads an instantaneous annotated snapshot frame directly from the active camera feed.
    """
    if stream_manager.current_frame_jpeg:
        from fastapi.responses import Response
        return Response(
            content=stream_manager.current_frame_jpeg,
            media_type="image/jpeg",
            headers={"Content-Disposition": f"attachment; filename=marine_debris_snapshot_{int(time.time())}.jpg"}
        )
    raise HTTPException(status_code=503, detail="Active camera frame not ready.")

import time

def generate_mjpeg_stream():
    """Generator for standard MJPEG video streaming of the active camera."""
    while True:
        if stream_manager.current_frame_jpeg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + stream_manager.current_frame_jpeg + b"\r\n"
            )
        time_sleep = 1.0 / max(10.0, stream_manager.fps)
        time.sleep(time_sleep)

def generate_quad_mjpeg_stream():
    """Generator for 2x2 multi-grid quad MJPEG video stream."""
    while True:
        quad_jpeg = stream_manager.get_quad_frame_jpeg()
        if quad_jpeg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + quad_jpeg + b"\r\n"
            )
        time.sleep(0.08)

def generate_station_mjpeg_stream(env_type: str):
    """Generator for individual station MJPEG stream."""
    while True:
        station_jpeg = stream_manager.get_station_frame_jpeg(env_type)
        if station_jpeg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + station_jpeg + b"\r\n"
            )
        time.sleep(0.08)

@router.get("/api/stream/mjpeg")
def mjpeg_video_feed():
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.get("/api/stream/quad")
def quad_video_feed():
    return StreamingResponse(
        generate_quad_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.get("/api/stream/station/{env_type}")
def station_video_feed(env_type: str):
    return StreamingResponse(
        generate_station_mjpeg_stream(env_type),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

from backend.app.services.pubsub import telemetry_bus

class DroneAltitudeUpdate(BaseModel):
    altitude_m: float

@router.post("/api/stream/drone/altitude")
def update_drone_altitude(payload: DroneAltitudeUpdate):
    stream_manager.set_drone_altitude(payload.altitude_m)
    return {
        "status": "success",
        "drone_altitude_m": stream_manager.drone_altitude_m,
        "environment_type": stream_manager.environment_type
    }

@router.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    sub_queue = telemetry_bus.subscribe()
    try:
        while True:
            # Wait for next published telemetry frame from pub/sub bus
            try:
                telemetry = await asyncio.wait_for(sub_queue.get(), timeout=1.0)
                await websocket.send_text(json.dumps(telemetry))
            except asyncio.TimeoutError:
                # Keep-alive heartbeat fallback
                if stream_manager.current_telemetry:
                    await websocket.send_text(json.dumps(stream_manager.current_telemetry))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket telemetry error: {e}")
    finally:
        telemetry_bus.unsubscribe(sub_queue)

@router.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            frame_bytes = stream_manager.current_frame_jpeg
            if frame_bytes:
                await websocket.send_bytes(frame_bytes)
            await asyncio.sleep(0.04)  # ~25 FPS binary video frame rate
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket video error: {e}")
