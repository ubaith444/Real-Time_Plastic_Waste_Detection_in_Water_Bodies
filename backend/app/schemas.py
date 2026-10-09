import datetime
from pydantic import BaseModel, Field

class CameraBase(BaseModel):
    id: str
    name: str
    environment_type: str
    source_url: str
    location_name: str
    latitude: float
    longitude: float
    is_active: bool = True
    fps: float = 30.0
    resolution: str = "1280x720"

class CameraCreate(CameraBase):
    pass

class CameraResponse(CameraBase):
    created_at: datetime.datetime

    class Config:
        from_attributes = True

class DetectionEventResponse(BaseModel):
    id: int
    camera_id: str
    track_id: int
    class_name: str
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    snapshot_path: str | None = None
    environmental_setting: str | None = None
    dwell_time_sec: float = 0.0
    timestamp: datetime.datetime

    class Config:
        from_attributes = True

class AlertResponse(BaseModel):
    id: int
    camera_id: str
    track_id: int
    severity: str
    class_name: str
    message: str
    snapshot_path: str | None = None
    acknowledged: bool
    timestamp: datetime.datetime

    class Config:
        from_attributes = True

class AlertAcknowledgeRequest(BaseModel):
    acknowledged: bool = True

class AnalyticsSummaryResponse(BaseModel):
    total_detections: int
    unique_objects_tracked: int
    total_alerts: int
    active_cameras: int
    class_breakdown: dict[str, int]
    severity_breakdown: dict[str, int]
    camera_breakdown: dict[str, int]
    recent_detections_timeline: list[dict]

class StreamTelemetry(BaseModel):
    fps: float
    latency_ms: float
    camera_id: str
    active_tracks: int
    current_frame_detections: list[dict]
