"""
Multi-Source Video Ingestion and Stream Processing Pipeline
Supports USB webcams, RTSP streams, recorded video files, and synthetic marine camera feeds.
Integrates SAHI for drone small objects and Area Estimator for plastic surface coverage.
"""

import asyncio
import time
import cv2
import numpy as np
from backend.app.cv.aquatic_simulators import MarineEnvironmentSimulator
from backend.app.cv.detector import PlasticDetector
from backend.app.cv.tracker import MarineTracker
from backend.app.cv.event_engine import EventEngine
from backend.app.cv.annotator import annotate_frame
from backend.app.cv.sahi_engine import SAHIEngine
from backend.app.cv.segmentation_engine import PlasticAreaEstimator
from backend.app.database import SessionLocal
from backend.app.models import DetectionEvent, Alert, Camera
from backend.app.services.background_worker import background_worker
from backend.app.services.pubsub import telemetry_bus

import threading
import queue

class AsyncRTSPReader:
    """
    Decoupled threaded video stream reader with a single-element ring-buffer queue.
    Prevents OpenCV read() from blocking during network jitter and automatically drops stale frames.
    """
    def __init__(self, source_url: str | int, max_queue_size: int = 1):
        self.source_url = source_url
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.thread: threading.Thread | None = None
        self.cap: cv2.VideoCapture | None = None
        self.last_frame: np.ndarray | None = None
        self.is_connected: bool = False
        self.start()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()

    def _capture_worker(self):
        try:
            url = int(self.source_url) if str(self.source_url).isdigit() else str(self.source_url)
            self.cap = cv2.VideoCapture(url)
            self.is_connected = self.cap.isOpened()
        except Exception:
            self.is_connected = False

        while self.running:
            if not self.is_connected or not self.cap or not self.cap.isOpened():
                time.sleep(1.0)
                try:
                    url = int(self.source_url) if str(self.source_url).isdigit() else str(self.source_url)
                    self.cap = cv2.VideoCapture(url)
                    self.is_connected = self.cap.isOpened()
                except Exception:
                    self.is_connected = False
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.04)
                continue

            self.last_frame = frame
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.queue.put_nowait(frame)
            except queue.Full:
                pass

    def read_latest(self) -> tuple[bool, np.ndarray | None]:
        """
        Retrieves the freshest real-time frame from the ring buffer.
        """
        try:
            frame = self.queue.get_nowait()
            return True, frame
        except queue.Empty:
            if self.last_frame is not None:
                return True, self.last_frame
            return False, None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

def apply_aquatic_enhancement(
    frame: np.ndarray,
    apply_clahe: bool = True,
    suppress_glint: bool = True
) -> np.ndarray:
    """
    Preprocessing filter to neutralize dynamic sun glint, wave reflections, and aquatic haze.
    Uses Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L-channel in LAB color space.
    """
    if not apply_clahe and not suppress_glint:
        return frame

    processed = frame.copy()

    # 1. Specular Sun Glint Suppression
    if suppress_glint:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        _, glint_mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(glint_mask) > 40:
            glint_dilated = cv2.dilate(glint_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            mean_color = cv2.mean(processed, mask=cv2.bitwise_not(glint_dilated))[:3]
            processed[glint_dilated > 0] = [int(c * 0.85) for c in mean_color]

    # 2. CLAHE for water depth / turbidity
    if apply_clahe:
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        processed = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return processed

def scan_hardware_cameras(max_indices: int = 4) -> list[dict]:
    """
    Scans physical USB / integrated cameras connected to the system.
    Returns list of verified devices with resolution and FPS.
    """
    found_devices = []
    for idx in range(max_indices):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                found_devices.append({
                    "device_index": idx,
                    "name": f"USB Video Capture Device #{idx}",
                    "resolution": f"{w}x{h}",
                    "fps": round(fps, 1),
                    "is_available": True
                })
            cap.release()
    return found_devices

def test_rtsp_connection(url: str, timeout_sec: float = 3.0) -> dict:
    """
    Tests an external RTSP / HTTP video stream.
    """
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return {
            "reachable": False,
            "error": "Could not connect to stream URL or connection timed out.",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
        }

    ret, frame = cap.read()
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    if ret and frame is not None:
        h, w = frame.shape[:2]
        cap.release()
        return {
            "reachable": True,
            "resolution": f"{w}x{h}",
            "latency_ms": latency_ms
        }
    cap.release()
    return {
        "reachable": False,
        "error": "Connection opened but unable to read video frame.",
        "latency_ms": latency_ms
    }

class StreamManager:
    def __init__(self):
        self.detector = PlasticDetector()
        self.tracker = MarineTracker()
        self.event_engine = EventEngine()
        self.sahi_engine = SAHIEngine()
        self.area_estimator = PlasticAreaEstimator(default_gsd_cm_per_pixel=1.2)

        self.current_camera_id: str = "drone-01"
        self.environment_type: str = "drone"
        self.camera_name: str = "Aerial Drone (Dal Lake)"
        self.gps_coords: tuple[float, float] = (34.1250, 74.8720)
        self.sahi_enabled: bool = True
        self.enhancement_enabled: bool = True

        # Drone flight optics for dynamic GSD calculation
        self.drone_altitude_m: float = 28.5
        self.drone_sensor_width_mm: float = 13.2  # 1-inch CMOS sensor
        self.drone_focal_length_mm: float = 8.8   # 24mm equivalent focal length

        # Simulator instances for instant testing of all 4 operating environments
        self.simulators = {
            "drone": MarineEnvironmentSimulator("drone", 1280, 720),
            "boat": MarineEnvironmentSimulator("boat", 1280, 720),
            "underwater": MarineEnvironmentSimulator("underwater", 1280, 720),
            "webcam": MarineEnvironmentSimulator("webcam", 1280, 720)
        }

        self.async_reader: AsyncRTSPReader | None = None
        self.use_physical_stream: bool = False

        self.current_frame_jpeg: bytes = b""
        self.current_telemetry: dict = {}
        self.is_running: bool = False
        self.fps: float = 30.0
        self.last_frame_time = time.time()

    def set_drone_altitude(self, altitude_m: float):
        """Updates drone altitude dynamically (simulating MAVLink/EXIF telemetry updates)."""
        self.drone_altitude_m = max(5.0, min(250.0, float(altitude_m)))

    def switch_camera(self, camera_id: str, env_type: str, name: str, source_url: str, gps: tuple[float, float]):
        self.current_camera_id = camera_id
        self.environment_type = env_type
        self.camera_name = name
        self.gps_coords = gps

        # Release existing asynchronous reader if running
        if self.async_reader:
            self.async_reader.stop()
            self.async_reader = None

        if source_url == "0" or source_url.isdigit() or source_url.startswith("rtsp://") or source_url.startswith("http://"):
            try:
                self.async_reader = AsyncRTSPReader(source_url)
                self.use_physical_stream = True
            except Exception as e:
                print(f"Async stream reader init error: {e}")
                self.use_physical_stream = False
        else:
            self.use_physical_stream = False

        # Reset tracker for the new scene
        self.tracker = MarineTracker()

    def get_raw_frame(self) -> tuple[np.ndarray, list[dict] | None]:
        if self.use_physical_stream and self.async_reader:
            ret, frame = self.async_reader.read_latest()
            if ret and frame is not None:
                if self.enhancement_enabled:
                    frame = apply_aquatic_enhancement(frame)
                return frame, None

        # Fallback / simulated feed with optional CLAHE enhancement
        sim = self.simulators.get(self.environment_type, self.simulators["drone"])
        raw_frame, synth_dets = sim.generate_frame()
        if self.enhancement_enabled:
            raw_frame = apply_aquatic_enhancement(raw_frame, apply_clahe=False, suppress_glint=True)
        return raw_frame, synth_dets

    def process_cycle(self):
        """
        Processes a single frame:
        Ingestion -> Sliced SAHI / Standard Inference -> Tracking -> Surface Coverage & Convex Hull -> Event Engine -> Annotation.
        """
        now = time.time()
        dt = max(now - self.last_frame_time, 0.001)
        self.fps = round(1.0 / dt, 1)
        self.last_frame_time = now

        raw_frame, synthetic_detections = self.get_raw_frame()

        # Step 1: AI Inference (Standard or SAHI for high-altitude drone small objects)
        if self.environment_type == "drone" and self.sahi_enabled:
            detections, latency_ms = self.sahi_engine.apply_sahi_inference(
                raw_frame,
                self.detector,
                synthetic_detections=synthetic_detections
            )
        else:
            detections, latency_ms = self.detector.detect(raw_frame, synthetic_detections)

        # Step 2: Multi-Object Tracking with Kalman Motion & Appearance Re-ID
        active_tracks = self.tracker.update(detections, frame=raw_frame)

        # Step 3: Dynamic GSD and Surface Area Coverage with Convex Hull Slick Metrics
        if self.environment_type == "drone":
            # Dynamic GSD: GSD = (sensor_width * altitude * 100) / (focal_length * image_width)
            gsd = self.area_estimator.compute_dynamic_gsd(
                self.drone_sensor_width_mm,
                self.drone_altitude_m,
                self.drone_focal_length_mm,
                raw_frame.shape[1]
            )
        else:
            gsd = 1.20

        coverage_stats = self.area_estimator.compute_coverage(raw_frame.shape, active_tracks, custom_gsd=gsd)

        # Step 4: Event Engine (deduplication, snapshot capture, alert qualification)
        qualifying_events, new_alerts = self.event_engine.process_tracks(
            self.current_camera_id,
            active_tracks,
            raw_frame,
            self.environment_type
        )

        # Step 5: Offload database persistence and snapshot writes to asynchronous background queue
        if qualifying_events or new_alerts:
            background_worker.enqueue_events_and_alerts(qualifying_events, new_alerts)

        # Step 6: Annotation
        annotated_frame = annotate_frame(
            raw_frame,
            active_tracks,
            self.fps,
            latency_ms,
            self.camera_name,
            self.environment_type,
            self.gps_coords
        )

        # Draw Debris Slick Convex Hull Perimeter for Cleanup Boom Teams
        hull_pts = coverage_stats.get("hull_points", [])
        if len(hull_pts) >= 3:
            pts_np = np.array(hull_pts, dtype=np.int32).reshape((-1, 1, 2))
            overlay = annotated_frame.copy()
            cv2.fillPoly(overlay, [pts_np], (210, 160, 40))
            cv2.addWeighted(overlay, 0.18, annotated_frame, 0.82, 0, annotated_frame)
            cv2.polylines(annotated_frame, [pts_np], isClosed=True, color=(240, 180, 50), thickness=2, lineType=cv2.LINE_AA)

            cx = int(np.mean([p[0] for p in hull_pts]))
            cy = int(np.mean([p[1] for p in hull_pts]))
            cv2.putText(
                annotated_frame,
                f"Slick Boom: {coverage_stats['containment_boom_meters']}m ({coverage_stats['slick_area_m2']}m2)",
                (max(10, cx - 110), max(25, cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 235, 120),
                1,
                cv2.LINE_AA
            )

        # Draw Surface Area Coverage HUD in bottom right corner
        cov_text = f"Coverage: {coverage_stats['coverage_percentage']}% ({coverage_stats['area_sq_meters']} m2) [{coverage_stats['density_status']}]"
        h, w = annotated_frame.shape[:2]
        cv2.putText(
            annotated_frame,
            cov_text,
            (w - 660, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 220, 100),
            1,
            cv2.LINE_AA
        )

        # Encode frame as JPEG for WebSocket/MJPEG streaming
        ret, jpeg = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret:
            self.current_frame_jpeg = jpeg.tobytes()

        # Telemetry packet
        self.current_telemetry = {
            "camera_id": self.current_camera_id,
            "camera_name": self.camera_name,
            "environment_type": self.environment_type,
            "gps": self.gps_coords,
            "fps": self.fps,
            "latency_ms": latency_ms,
            "debris_count": len(active_tracks),
            "sahi_enabled": self.sahi_enabled,
            "sahi_mode": "batched_tensor" if self.sahi_enabled else "disabled",
            "sahi_slices": self.sahi_engine.get_slice_count(raw_frame.shape) if (self.environment_type == "drone" and self.sahi_enabled) else 0,
            "tracker_mode": "kalman_2d_fluid",
            "water_current_vector": list(self.tracker.water_current),
            "coverage_percentage": coverage_stats["coverage_percentage"],
            "area_sq_meters": coverage_stats["area_sq_meters"],
            "density_status": coverage_stats["density_status"],
            "density_badge": coverage_stats["density_badge"],
            "drone_altitude_m": self.drone_altitude_m if self.environment_type == "drone" else None,
            "dynamic_gsd_cm_per_px": gsd,
            "slick_area_m2": coverage_stats.get("slick_area_m2", 0.0),
            "containment_boom_meters": coverage_stats.get("containment_boom_meters", 0.0),
            "hull_points": hull_pts,
            "tracks": [
                {
                    "track_id": t.track_id,
                    "class_name": t.class_name,
                    "confidence": t.confidence,
                    "dwell_time_sec": t.dwell_time_sec,
                    "is_occluded": getattr(t, "is_occluded", False),
                    "velocity": getattr(t, "velocity", (0.0, 0.0)),
                    "box": t.box,
                    "normalized_box": t.normalized_box
                }
                for t in active_tracks
            ],
            "recent_alert_count": len(new_alerts),
            "timestamp": time.time()
        }

        # Broadcast telemetry through pubsub broker
        telemetry_bus.publish_telemetry(self.current_telemetry)

    def get_station_frame_jpeg(self, env_type: str) -> bytes:
        """
        Returns an annotated JPEG frame for a specific environment type.
        """
        if env_type == self.environment_type and self.current_frame_jpeg:
            return self.current_frame_jpeg

        sim = self.simulators.get(env_type)
        if not sim:
            return self.current_frame_jpeg

        frame, dets = sim.generate_frame()
        h, w = frame.shape[:2]
        title = f"Station: {env_type.upper()} Ingestion"
        cv2.rectangle(frame, (0, 0), (w, 36), (15, 23, 42), -1)
        cv2.putText(frame, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if dets:
            for d in dets:
                x1, y1, x2, y2 = d["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (235, 99, 37), 2)
                cv2.putText(
                    frame,
                    d["class_name"].replace("_", " ").title(),
                    (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return jpeg.tobytes() if ret else b""

    def get_quad_frame_jpeg(self) -> bytes:
        """
        Generates a 2x2 multi-grid frame (1280x720) combining all 4 operating environments:
        Top-Left: Drone | Top-Right: Boat | Bottom-Left: Underwater | Bottom-Right: Webcam.
        """
        tiles = []
        labels = [
            ("drone", "STATION 1: AERIAL DRONE (DAL LAKE)"),
            ("boat", "STATION 2: PATROL BOAT (GANGES)"),
            ("underwater", "STATION 3: COASTAL ROV (MUMBAI)"),
            ("webcam", "STATION 4: TEST TANK (USB WEBCAM)")
        ]

        for env, label in labels:
            if env == self.environment_type and self.current_frame_jpeg:
                nparr = np.frombuffer(self.current_frame_jpeg, np.uint8)
                tile = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if tile is not None:
                    tile = cv2.resize(tile, (640, 360))
                else:
                    sim = self.simulators.get(env, self.simulators["drone"])
                    raw, _ = sim.generate_frame()
                    tile = cv2.resize(raw, (640, 360))
            else:
                sim = self.simulators.get(env, self.simulators["drone"])
                raw, dets = sim.generate_frame()
                tile = cv2.resize(raw, (640, 360))
                if dets:
                    for d in dets:
                        bx = [int(v * 0.5) for v in d["box"]]
                        cv2.rectangle(tile, (bx[0], bx[1]), (bx[2], bx[3]), (235, 99, 37), 1)

            # Draw banner header
            cv2.rectangle(tile, (0, 0), (640, 30), (15, 23, 42), -1)
            cv2.putText(tile, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            status_text = "FOCUSED" if env == self.environment_type else "INGESTING"
            status_color = (100, 255, 100) if env == self.environment_type else (200, 220, 240)
            cv2.putText(tile, status_text, (550, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_color, 1, cv2.LINE_AA)
            tiles.append(tile)

        top_row = np.hstack([tiles[0], tiles[1]])
        bot_row = np.hstack([tiles[2], tiles[3]])
        quad = np.vstack([top_row, bot_row])

        # Draw grid dividing lines
        cv2.line(quad, (640, 0), (640, 720), (50, 70, 90), 2)
        cv2.line(quad, (0, 360), (1280, 360), (50, 70, 90), 2)

        ret, jpeg = cv2.imencode(".jpg", quad, [cv2.IMWRITE_JPEG_QUALITY, 72])
        return jpeg.tobytes() if ret else b""

    def _persist_events_and_alerts(self, events: list[dict], alerts: list[dict]):
        try:
            db = SessionLocal()
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
                    snapshot_path=ev["snapshot_path"],
                    environmental_setting=ev["environmental_setting"],
                    dwell_time_sec=ev["dwell_time_sec"]
                )
                db.add(db_event)

            for al in alerts:
                db_alert = Alert(
                    camera_id=al["camera_id"],
                    track_id=al["track_id"],
                    severity=al["severity"],
                    class_name=al["class_name"],
                    message=al["message"],
                    snapshot_path=al["snapshot_path"],
                    acknowledged=False
                )
                db.add(db_alert)

            db.commit()
            db.close()
        except Exception as e:
            print(f"Error persisting detection events to database: {e}")

stream_manager = StreamManager()
