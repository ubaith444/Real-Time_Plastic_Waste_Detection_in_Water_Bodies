import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
MODELS_DIR = DATA_DIR / "models"
DATASETS_DIR = DATA_DIR / "datasets"

# Ensure directories exist
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

class Settings(BaseModel):
    app_name: str = "Real-Time Plastic Waste Detection System"
    app_version: str = "1.0.0"
    database_url: str = f"sqlite:///{BASE_DIR / 'marine_waste.db'}"
    default_confidence_threshold: float = 0.40
    default_iou_threshold: float = 0.45
    tracker_max_age_frames: int = 30
    tracker_min_hits: int = 3
    duplicate_suppression_window_sec: float = 15.0
    snapshot_quality_jpeg: int = 90
    max_recent_alerts: int = 100
    
    # Unified marine plastic classes
    classes: list[str] = [
        "plastic_bottle",
        "plastic_bag",
        "plastic_container",
        "fishing_net_rope",
        "micro_macro_fragment"
    ]

settings = Settings()
