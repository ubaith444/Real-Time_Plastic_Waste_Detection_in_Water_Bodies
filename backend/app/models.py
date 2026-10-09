import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from backend.app.database import Base

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    environment_type = Column(String, nullable=False)  # webcam, drone, boat, underwater
    source_url = Column(String, nullable=False)  # 0, rtsp://..., synthetic://...
    location_name = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    is_active = Column(Boolean, default=True)
    fps = Column(Float, default=30.0)
    resolution = Column(String, default="1280x720")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    detections = relationship("DetectionEvent", back_populates="camera", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="camera", cascade="all, delete-orphan")

class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    camera_id = Column(String, ForeignKey("cameras.id"), nullable=False, index=True)
    track_id = Column(Integer, nullable=False, index=True)
    class_name = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    bbox_x = Column(Float, nullable=False)  # normalized 0-1
    bbox_y = Column(Float, nullable=False)
    bbox_w = Column(Float, nullable=False)
    bbox_h = Column(Float, nullable=False)
    snapshot_path = Column(String, nullable=True)
    environmental_setting = Column(String, nullable=True)
    dwell_time_sec = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    camera = relationship("Camera", back_populates="detections")

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    camera_id = Column(String, ForeignKey("cameras.id"), nullable=False, index=True)
    track_id = Column(Integer, nullable=False)
    severity = Column(String, nullable=False, index=True)  # Critical, High, Medium
    class_name = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    snapshot_path = Column(String, nullable=True)
    acknowledged = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    camera = relationship("Camera", back_populates="alerts")

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_role = Column(String, default="Operator")
    action_type = Column(String, nullable=False)
    target_entity = Column(String, nullable=False)
    details = Column(Text, nullable=False)
    ip_address = Column(String, default="127.0.0.1")
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)

