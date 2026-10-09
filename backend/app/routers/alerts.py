from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from backend.app.database import get_db
from backend.app.models import Alert
from backend.app.schemas import AlertResponse, AlertAcknowledgeRequest

router = APIRouter(prefix="/api/alerts", tags=["alerts"])

@router.get("", response_model=list[AlertResponse])
def get_alerts(
    severity: str | None = Query(None),
    unacknowledged_only: bool = Query(False),
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db)
):
    query = db.query(Alert)
    if severity:
        query = query.filter(Alert.severity == severity)
    if unacknowledged_only:
        query = query.filter(Alert.acknowledged == False)

    records = query.order_by(desc(Alert.timestamp)).limit(limit).all()
    return records

@router.post("/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: int, payload: AlertAcknowledgeRequest, db: Session = Depends(get_db)):
    al = db.query(Alert).filter(Alert.id == alert_id).first()
    if not al:
        raise HTTPException(status_code=404, detail="Alert not found.")
    al.acknowledged = payload.acknowledged
    db.commit()
    return {"status": "success", "alert_id": alert_id, "acknowledged": al.acknowledged}

@router.post("/acknowledge-all")
def acknowledge_all_alerts(db: Session = Depends(get_db)):
    db.query(Alert).filter(Alert.acknowledged == False).update({"acknowledged": True})
    db.commit()
    return {"status": "success", "message": "All alerts acknowledged."}

import json
import shutil
import time
from pathlib import Path
from backend.app.config import BASE_DIR

ACTIVE_LEARNING_DIR = BASE_DIR / "data" / "active_learning"
FP_DIR = ACTIVE_LEARNING_DIR / "false_positives"
REVIEW_QUEUE_FILE = ACTIVE_LEARNING_DIR / "review_queue.json"

ACTIVE_LEARNING_DIR.mkdir(parents=True, exist_ok=True)
FP_DIR.mkdir(parents=True, exist_ok=True)

if not REVIEW_QUEUE_FILE.exists():
    with open(REVIEW_QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

@router.post("/{alert_id}/flag-fp")
def flag_false_positive(alert_id: int, db: Session = Depends(get_db)):
    """
    Active Learning Loop:
    Flags an alert as a False Positive detection.
    Copies the evidence snapshot into the active-learning review directory
    and registers it in the retraining candidate queue.
    """
    al = db.query(Alert).filter(Alert.id == alert_id).first()
    if not al:
        raise HTTPException(status_code=404, detail="Alert not found.")

    al.acknowledged = True

    saved_snapshot_name = None
    if al.snapshot_path:
        src_path = Path(al.snapshot_path)
        if not src_path.is_absolute():
            src_path = BASE_DIR / al.snapshot_path
        if src_path.exists():
            dest_name = f"fp_alert_{al.id}_{src_path.name}"
            dest_path = FP_DIR / dest_name
            shutil.copy2(str(src_path), str(dest_path))
            saved_snapshot_name = str(dest_path.relative_to(BASE_DIR))

    # Record in active learning review queue
    queue_data = []
    if REVIEW_QUEUE_FILE.exists():
        try:
            with open(REVIEW_QUEUE_FILE, "r", encoding="utf-8") as f:
                queue_data = json.load(f)
        except Exception:
            queue_data = []

    fp_entry = {
        "alert_id": al.id,
        "track_id": al.track_id,
        "camera_id": al.camera_id,
        "predicted_class": al.class_name,
        "severity": al.severity,
        "message": al.message,
        "archived_snapshot": saved_snapshot_name,
        "flagged_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "status": "pending_retraining_curation"
    }
    queue_data.append(fp_entry)

    with open(REVIEW_QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(queue_data, f, indent=2)

    db.commit()
    return {
        "status": "success",
        "message": f"Alert #{al.id} flagged as False Positive for active-learning retraining.",
        "entry": fp_entry
    }

@router.get("/active-learning/queue")
def get_active_learning_queue():
    """
    Returns the list of false-positive samples flagged by operators for scheduled model retraining.
    """
    if REVIEW_QUEUE_FILE.exists():
        try:
            with open(REVIEW_QUEUE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

