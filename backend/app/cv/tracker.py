"""
Multi-Object Tracking (MOT) Engine for Marine Plastic Debris
Integrated with 2D Kalman Filter and fluid dynamics modeling for water current drift.
Maintains persistent Track IDs across wave occlusions, smooths jittery trajectories,
and computes dwell time in aquatic bodies.
"""

import time
import math
import numpy as np

def calculate_iou(boxA: list[int], boxB: list[int]) -> float:
    """
    Computes Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return float(iou)

import cv2

def extract_appearance_embedding(frame: np.ndarray | None, box: list[int]) -> np.ndarray:
    """
    Extracts a lightweight normalized visual appearance Re-ID feature vector (32-dim HSV descriptor).
    Enables DeepSORT-style identity maintenance across complex trajectory crossings.
    """
    if frame is None:
        return np.zeros(32, dtype=np.float32)

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
    x2, y2 = max(x1 + 1, min(w, x2)), max(y1 + 1, min(h, y2))
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return np.zeros(32, dtype=np.float32)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).flatten()
    norm = np.linalg.norm(hist)
    if norm > 1e-6:
        hist = hist / norm
    return hist.astype(np.float32)

def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

class KalmanFilter2D:
    """
    2D Kalman Filter with fluid dynamics modeling for marine floating debris.
    State vector: [cx, cy, vx, vy, w, h]^T
    Measurement vector: [cx, cy, w, h]^T
    """
    def __init__(self, initial_box: list[int], current_velocity: tuple[float, float] = (0.0, 0.0)):
        x1, y1, x2, y2 = initial_box
        cx = float(x1 + x2) / 2.0
        cy = float(y1 + y2) / 2.0
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        vx, vy = float(current_velocity[0]), float(current_velocity[1])

        # State vector: [cx, cy, vx, vy, w, h]
        self.x = np.array([cx, cy, vx, vy, w, h], dtype=np.float64)

        # State transition matrix F (constant velocity motion model)
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 2] = 1.0  # cx += vx * dt
        self.F[1, 3] = 1.0  # cy += vy * dt

        # Water current bias control matrix B
        self.B = np.zeros((6, 2), dtype=np.float64)
        self.B[0, 0] = 0.5  # cx drift component
        self.B[1, 1] = 0.5  # cy drift component

        # Measurement matrix H
        self.H = np.zeros((4, 6), dtype=np.float64)
        self.H[0, 0] = 1.0  # measure cx
        self.H[1, 1] = 1.0  # measure cy
        self.H[2, 4] = 1.0  # measure w
        self.H[3, 5] = 1.0  # measure h

        # Error covariance matrix P
        self.P = np.diag([10.0, 10.0, 50.0, 50.0, 10.0, 10.0]).astype(np.float64)

        # Process noise Q (wave chop and turbulent water flow)
        self.Q = np.diag([1.0, 1.0, 4.0, 4.0, 1.0, 1.0]).astype(np.float64)

        # Measurement noise R (bounding-box detector jitter)
        self.R = np.diag([4.0, 4.0, 9.0, 9.0]).astype(np.float64)

    def predict(self, water_current: tuple[float, float] = (0.0, 0.0), dt: float = 1.0) -> list[int]:
        """
        Advances the filter state forward in time using constant velocity and water current bias.
        """
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        u = np.array(water_current, dtype=np.float64)

        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.get_box()

    def update(self, measurement_box: list[int]):
        """
        Corrects state estimate using matched YOLO bounding-box detection.
        """
        x1, y1, x2, y2 = measurement_box
        cx = float(x1 + x2) / 2.0
        cy = float(y1 + y2) / 2.0
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        z = np.array([cx, cy, w, h], dtype=np.float64)

        # Measurement innovation y and innovation covariance S
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state and error covariance
        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

    def get_centroid(self) -> tuple[int, int]:
        return int(round(self.x[0])), int(round(self.x[1]))

    def get_velocity(self) -> tuple[float, float]:
        return float(round(self.x[2], 2)), float(round(self.x[3], 2))

    def get_box(self) -> list[int]:
        cx, cy, _, _, w, h = self.x
        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2 = int(round(cx + w / 2.0))
        y2 = int(round(cy + h / 2.0))
        return [max(0, x1), max(0, y1), max(1, x2), max(1, y2)]

class Track:
    def __init__(self, track_id: int, detection: dict, initial_current: tuple[float, float] = (0.0, 0.0), frame: np.ndarray | None = None):
        self.track_id = track_id
        self.class_name = detection["class_name"]
        self.box = list(detection["box"])
        self.normalized_box = list(detection.get("normalized_box", [0.0, 0.0, 0.0, 0.0]))
        self.confidence = float(detection["confidence"])
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.hits = 1
        self.misses = 0
        self.is_occluded = False

        # Initialize 2D Kalman Filter
        self.kf = KalmanFilter2D(self.box, current_velocity=initial_current)
        self.trajectory: list[tuple[int, int]] = [self.get_centroid()]

        # Initialize visual appearance Re-ID embedding
        self.embedding: np.ndarray = extract_appearance_embedding(frame, self.box)

    def predict(self, water_current: tuple[float, float] = (0.0, 0.0)) -> list[int]:
        """
        Predicts future position along water current velocity vector.
        """
        predicted_box = self.kf.predict(water_current=water_current)
        if self.misses > 0:
            # Under wave occlusion: smoothly update box and trajectory with Kalman prediction
            self.box = predicted_box
            centroid = self.get_centroid()
            self.trajectory.append(centroid)
            if len(self.trajectory) > 25:
                self.trajectory.pop(0)
            self.is_occluded = True
        return predicted_box

    def update(self, detection: dict, frame: np.ndarray | None = None):
        """
        Corrects track state with incoming detector measurement and updates appearance embedding.
        """
        self.class_name = detection["class_name"]
        self.confidence = float(detection["confidence"])
        self.last_seen = time.time()
        self.hits += 1
        self.misses = 0
        self.is_occluded = False

        # Update Kalman Filter with actual detection
        self.kf.update(detection["box"])
        self.box = self.kf.get_box()
        self.normalized_box = list(detection.get("normalized_box", self.normalized_box))

        # Update appearance Re-ID embedding with EMA
        if frame is not None:
            new_emb = extract_appearance_embedding(frame, detection["box"])
            if np.any(new_emb):
                if np.any(self.embedding):
                    self.embedding = 0.85 * self.embedding + 0.15 * new_emb
                    norm = np.linalg.norm(self.embedding)
                    if norm > 1e-6:
                        self.embedding = self.embedding / norm
                else:
                    self.embedding = new_emb

        centroid = self.get_centroid()
        self.trajectory.append(centroid)
        if len(self.trajectory) > 25:
            self.trajectory.pop(0)

    def get_centroid(self) -> tuple[int, int]:
        return self.kf.get_centroid()

    @property
    def velocity(self) -> tuple[float, float]:
        return self.kf.get_velocity()

    @property
    def dwell_time_sec(self) -> float:
        return round(self.last_seen - self.first_seen, 1)

class MarineTracker:
    def __init__(
        self,
        max_misses: int = 15,
        iou_threshold: float = 0.20,
        dist_threshold: float = 90.0,
        enable_current_learning: bool = True
    ):
        self.max_misses = max_misses
        self.iou_threshold = iou_threshold
        self.dist_threshold = dist_threshold
        self.enable_current_learning = enable_current_learning
        self.next_id = 1
        self.tracks: list[Track] = []
        # Ambient water current drift vector (vx, vy)
        self.water_current: tuple[float, float] = (0.8, 0.3)

    def update(self, detections: list[dict], frame: np.ndarray | None = None) -> list[Track]:
        """
        Associates current frame detections with existing tracks using Kalman prediction,
        DeepSORT-style appearance Re-ID embeddings, IoU matching, and distance-based drift matching.
        """
        # Step 1: Predict new locations for all existing tracks using Kalman Filter
        predicted_boxes = {}
        for track in self.tracks:
            pred_box = track.predict(water_current=self.water_current)
            predicted_boxes[track.track_id] = pred_box

        # If no tracks exist yet, initialize all detections as new tracks
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(Track(self.next_id, det, initial_current=self.water_current, frame=frame))
                self.next_id += 1
            return self.tracks

        matched_tracks = set()
        matched_dets = set()

        # Step 2: Associate detections using IoU + Appearance Cosine Similarity against Kalman-predicted boxes
        for t_idx, track in enumerate(self.tracks):
            best_score = self.iou_threshold
            best_d_idx = -1
            pred_box = predicted_boxes[track.track_id]

            for d_idx, det in enumerate(detections):
                if d_idx in matched_dets:
                    continue
                iou = calculate_iou(pred_box, det["box"])
                if frame is not None and np.any(track.embedding):
                    det_emb = extract_appearance_embedding(frame, det["box"])
                    sim = compute_cosine_similarity(track.embedding, det_emb)
                    combined_score = 0.65 * iou + 0.35 * max(0.0, sim)
                else:
                    combined_score = iou

                if combined_score > best_score:
                    best_score = combined_score
                    best_d_idx = d_idx

            if best_d_idx != -1:
                track.update(detections[best_d_idx], frame=frame)
                matched_tracks.add(t_idx)
                matched_dets.add(best_d_idx)

        # Step 3: Distance-based fallback association for objects drifting quickly in waves
        for t_idx, track in enumerate(self.tracks):
            if t_idx in matched_tracks:
                continue
            tcx, tcy = track.get_centroid()
            min_dist = self.dist_threshold
            best_d_idx = -1

            for d_idx, det in enumerate(detections):
                if d_idx in matched_dets:
                    continue
                dcx = int((det["box"][0] + det["box"][2]) / 2)
                dcy = int((det["box"][1] + det["box"][3]) / 2)
                dist = math.hypot(tcx - dcx, tcy - dcy)
                if dist < min_dist:
                    min_dist = dist
                    best_d_idx = d_idx

            if best_d_idx != -1:
                track.update(detections[best_d_idx])
                matched_tracks.add(t_idx)
                matched_dets.add(best_d_idx)

        # Step 4: Manage unmatched tracks (increment misses and maintain across wave occlusions)
        active_surviving_tracks = []
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.misses += 1
                track.is_occluded = True
            if track.misses <= self.max_misses:
                active_surviving_tracks.append(track)

        self.tracks = active_surviving_tracks

        # Step 5: Initialize new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                self.tracks.append(Track(self.next_id, det, initial_current=self.water_current))
                self.next_id += 1

        # Step 6: Adaptive water current vector estimation
        if self.enable_current_learning:
            self._adapt_water_current()

        return self.tracks

    def _adapt_water_current(self):
        """
        Estimates the ambient water current drift vector from confirmed active tracks.
        """
        confirmed = [t for t in self.tracks if t.hits >= 3 and not t.is_occluded]
        if confirmed:
            avg_vx = float(np.mean([t.velocity[0] for t in confirmed]))
            avg_vy = float(np.mean([t.velocity[1] for t in confirmed]))
            # Exponential Moving Average update
            alpha = 0.90
            cur_vx = alpha * self.water_current[0] + (1.0 - alpha) * avg_vx
            cur_vy = alpha * self.water_current[1] + (1.0 - alpha) * avg_vy
            self.water_current = (round(cur_vx, 2), round(cur_vy, 2))
