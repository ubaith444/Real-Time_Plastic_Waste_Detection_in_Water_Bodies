"""
Storage Service
Manages captured snapshot evidence files, bounding box exports, and disk quota.
"""

from pathlib import Path
from backend.app.config import SNAPSHOTS_DIR

class StorageService:
    @staticmethod
    def get_snapshot_path(filename: str) -> Path | None:
        filepath = SNAPSHOTS_DIR / filename
        if filepath.exists() and filepath.is_file():
            return filepath
        return None

    @staticmethod
    def list_snapshots() -> list[str]:
        if not SNAPSHOTS_DIR.exists():
            return []
        return [f.name for f in SNAPSHOTS_DIR.glob("*.jpg")]

    @staticmethod
    def get_storage_stats() -> dict:
        total_size = sum(f.stat().st_size for f in SNAPSHOTS_DIR.glob("*.jpg"))
        count = len(list(SNAPSHOTS_DIR.glob("*.jpg")))
        return {
            "snapshot_count": count,
            "total_bytes": total_size,
            "total_mb": round(total_size / (1024 * 1024), 2)
        }
