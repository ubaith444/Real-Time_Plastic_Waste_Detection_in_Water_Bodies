"""
Event and Alert Qualification Engine
Handles duplicate suppression, snapshot evidence storage, and severity rules.
"""

import time
import os
from pathlib import Path
import numpy as np
import cv2
from backend.app.config import SNAPSHOTS_DIR, settings

class EventEngine:
    def __init__(self, suppression_window_sec: float = 15.0):
        self.suppression_window_sec = suppression_window_sec
        # track_id -> timestamp of last event recorded
        self.recorded_events: dict[int, float] = {}
        # track_id -> snapshot path
        self.recorded_snapshots: dict[int, str] = {}

    def process_tracks(
        self,
        camera_id: str,
        tracks: list,
        raw_frame: np.ndarray,
        environment_setting: str = "drone"
    ) -> tuple[list[dict], list[dict]]:
        """
        Evaluates active tracks for qualifying detection events and alerts.
        Returns:
            qualifying_events: list of events to log to DB.
            new_alerts: list of alerts to fire.
        """
        now = time.time()
        qualifying_events = []
        new_alerts = []
        total_items_in_view = len(tracks)

        for track in tracks:
            # Require at least 2 hits to prevent single-frame noise glitches
            if track.hits < 2:
                continue

            last_time = self.recorded_events.get(track.track_id, 0.0)
            is_new = track.track_id not in self.recorded_events
            cooldown_passed = (now - last_time) > self.suppression_window_sec

            if is_new or cooldown_passed:
                # Capture snapshot
                snapshot_filename = self._save_snapshot(camera_id, track, raw_frame)
                self.recorded_events[track.track_id] = now
                self.recorded_snapshots[track.track_id] = snapshot_filename

                event_data = {
                    "camera_id": camera_id,
                    "track_id": track.track_id,
                    "class_name": track.class_name,
                    "confidence": track.confidence,
                    "bbox_x": track.normalized_box[0],
                    "bbox_y": track.normalized_box[1],
                    "bbox_w": track.normalized_box[2],
                    "bbox_h": track.normalized_box[3],
                    "snapshot_path": snapshot_filename,
                    "environmental_setting": environment_setting,
                    "dwell_time_sec": track.dwell_time_sec,
                    "timestamp": now
                }
                qualifying_events.append(event_data)

                # Evaluate alert rules
                alert = self._evaluate_alert_rules(camera_id, track, snapshot_filename, total_items_in_view)
                if alert:
                    new_alerts.append(alert)

        return qualifying_events, new_alerts

    def _evaluate_alert_rules(self, camera_id: str, track, snapshot_path: str, total_items_in_view: int) -> dict | None:
        """
        Deterministic alert rule evaluation.
        """
        severity = None
        message = ""

        # Critical: Ghost fishing gear or dense cluster
        if track.class_name == "fishing_net_rope":
            severity = "Critical"
            message = f"Marine hazard detected: Submerged/floating fishing net or rope [Track #{track.track_id}] (Entanglement Risk)"
        elif total_items_in_view >= 5:
            severity = "Critical"
            message = f"High-density plastic accumulation detected: {total_items_in_view} active items in water body"
        # High: Persistent accumulation or larger container
        elif track.dwell_time_sec >= 12.0:
            severity = "High"
            message = f"Persistent plastic accumulation: {track.class_name} floating in place for {track.dwell_time_sec}s [Track #{track.track_id}]"
        elif track.class_name == "plastic_container":
            severity = "High"
            message = f"Large plastic container detected: {track.class_name} with {int(track.confidence * 100)}% confidence"
        # Medium: Floating single-use items
        elif track.confidence >= 0.60:
            severity = "Medium"
            message = f"Floating marine litter identified: {track.class_name} [Track #{track.track_id}]"

        if severity:
            return {
                "camera_id": camera_id,
                "track_id": track.track_id,
                "severity": severity,
                "class_name": track.class_name,
                "message": message,
                "snapshot_path": snapshot_path,
                "timestamp": time.time()
            }
        return None

    def _save_snapshot(self, camera_id: str, track, frame: np.ndarray) -> str:
        """
        Saves snapshot evidence image with timestamp metadata burned in.
        """
        ts_str = int(time.time() * 1000)
        filename = f"{camera_id}_trk{track.track_id}_{ts_str}.jpg"
        filepath = SNAPSHOTS_DIR / filename

        annotated_crop = frame.copy()
        x1, y1, x2, y2 = track.box
        h, w = frame.shape[:2]

        # Draw highlight on snapshot
        cv2.rectangle(annotated_crop, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated_crop,
            f"{track.class_name} #{track.track_id} ({int(track.confidence * 100)}%)",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

        # Header metadata watermark
        header_text = f"CAM: {camera_id.upper()} | TRACK: #{track.track_id} | TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        cv2.putText(annotated_crop, header_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imwrite(str(filepath), annotated_crop, [cv2.IMWRITE_JPEG_QUALITY, settings.snapshot_quality_jpeg])
        return filename
