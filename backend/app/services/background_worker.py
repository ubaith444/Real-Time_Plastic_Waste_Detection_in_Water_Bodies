"""
Decoupled Asynchronous Background Task Queue
Offloads database writes, telemetry logging, and snapshot disk I/O
off the high-speed computer vision frame processing thread.
"""

import queue
import threading
import time
from backend.app.database import SessionLocal
from backend.app.models import DetectionEvent, Alert

class BackgroundWorkerQueue:
    def __init__(self, max_queue_size: int = 5000):
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._is_running = True
        self._total_processed = 0
        self._total_dropped = 0
        self._worker_thread = threading.Thread(target=self._process_loop, daemon=True, name="AquaWaste-DB-Worker")
        self._worker_thread.start()

    def enqueue_events_and_alerts(self, events: list[dict], alerts: list[dict]):
        """
        Non-blocking enqueue of detection events and alerts for database persistence.
        Drops oldest if queue is saturated to prevent blocking the video ingestion pipeline.
        """
        task = {
            "type": "persist_events",
            "events": events,
            "alerts": alerts,
            "timestamp": time.time()
        }
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            self._total_dropped += 1
            # Evict one item and insert
            try:
                _ = self._queue.get_nowait()
                self._queue.put_nowait(task)
            except Exception:
                pass

    def _process_loop(self):
        while self._is_running:
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                if task["type"] == "persist_events":
                    self._persist_to_database(task.get("events", []), task.get("alerts", []))
                    self._total_processed += 1
            except Exception as e:
                print(f"[BackgroundWorker] Database write exception: {e}")
            finally:
                self._queue.task_done()

    def _persist_to_database(self, events: list[dict], alerts: list[dict]):
        if not events and not alerts:
            return

        db = SessionLocal()
        try:
            for ev in events:
                db_event = DetectionEvent(
                    camera_id=ev["camera_id"],
                    track_id=ev["track_id"],
                    class_name=ev["class_name"],
                    confidence=ev["confidence"],
                    bbox_x=ev["bbox_x"],
                    bbox_y=ev["bbox_y"],
                    bbox_w=ev["bbox_w"],
                    bbox_h=ev["bbox_h"],
                    snapshot_path=ev.get("snapshot_path"),
                    environmental_setting=ev.get("environmental_setting", "marine"),
                    dwell_time_sec=ev.get("dwell_time_sec", 0.0)
                )
                db.add(db_event)

            for al in alerts:
                db_alert = Alert(
                    camera_id=al["camera_id"],
                    track_id=al["track_id"],
                    severity=al["severity"],
                    class_name=al["class_name"],
                    message=al["message"],
                    snapshot_path=al.get("snapshot_path"),
                    acknowledged=False
                )
                db.add(db_alert)

            db.commit()
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    def get_stats(self) -> dict:
        return {
            "queue_depth": self._queue.qsize(),
            "total_processed": self._total_processed,
            "total_dropped": self._total_dropped,
            "is_alive": self._worker_thread.is_alive()
        }

    def stop(self):
        self._is_running = False

background_worker = BackgroundWorkerQueue()
