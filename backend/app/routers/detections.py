from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from backend.app.database import get_db
from backend.app.models import DetectionEvent
from backend.app.schemas import DetectionEventResponse

router = APIRouter(prefix="/api/detections", tags=["detections"])

@router.get("", response_model=list[DetectionEventResponse])
def get_detections(
    camera_id: str | None = Query(None),
    class_name: str | None = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    db: Session = Depends(get_db)
):
    query = db.query(DetectionEvent)
    if camera_id:
        query = query.filter(DetectionEvent.camera_id == camera_id)
    if class_name:
        query = query.filter(DetectionEvent.class_name == class_name)

    records = query.order_by(desc(DetectionEvent.timestamp)).offset(offset).limit(limit).all()
    return records
