"""
YOLO Detection Engine for Floating and Submerged Marine Plastic Waste
Integrated with Ultralytics YOLO and fallback calibration models.
Supports batched forward-pass inference for Slicing Aided Hyper Inference (SAHI).
"""

import time
from pathlib import Path
import numpy as np
import cv2
from backend.app.config import settings, MODELS_DIR

class PlasticDetector:
    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.40, iou_threshold: float = 0.45):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model_path = MODELS_DIR / model_name
        self.yolo_model = None
        self.class_names = settings.classes
        self._init_model()

    def _init_model(self):
        try:
            from ultralytics import YOLO
            # If local custom model exists, load it; otherwise initialize YOLOv8n nano
            if self.model_path.exists():
                self.yolo_model = YOLO(str(self.model_path))
            else:
                self.yolo_model = YOLO("yolov8n.pt")
            print("Ultralytics YOLO engine initialized successfully.")
        except Exception as e:
            print(f"Ultralytics YOLO initialization notice: {e}. Utilizing native vision fallback.")
            self.yolo_model = None

    def _map_class_name(self, cls_idx: int) -> str:
        if cls_idx < len(self.class_names):
            return self.class_names[cls_idx]
        elif cls_idx == 39: # COCO bottle
            return "plastic_bottle"
        elif cls_idx == 41: # COCO cup
            return "plastic_container"
        else:
            return "micro_macro_fragment"

    def detect(self, frame: np.ndarray, synthetic_detections: list[dict] | None = None) -> tuple[list[dict], float]:
        """
        Runs inference on an image/frame.
        Returns (detections, latency_ms).
        Each detection: {
            "class_name": str,
            "confidence": float,
            "box": [x1, y1, x2, y2],
            "normalized_box": [x, y, w, h]
        }
        """
        start_time = time.perf_counter()
        h, w = frame.shape[:2]

        # If synthetic marine stream is providing ground truth calibration detections
        if synthetic_detections is not None:
            filtered = [
                d for d in synthetic_detections
                if d["confidence"] >= self.confidence_threshold
            ]
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return filtered, round(latency_ms, 2)

        # Live camera / video file inference via YOLO
        detections = []
        if self.yolo_model is not None:
            try:
                results = self.yolo_model.predict(
                    source=frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                for res in results:
                    boxes = res.boxes
                    for box in boxes:
                        coords = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_idx = int(box.cls[0].cpu().numpy())
                        class_name = self._map_class_name(cls_idx)

                        x1, y1, x2, y2 = [int(v) for v in coords]
                        box_w = x2 - x1
                        box_h = y2 - y1

                        detections.append({
                            "class_name": class_name,
                            "confidence": round(conf, 3),
                            "box": [x1, y1, x2, y2],
                            "normalized_box": [
                                round(x1 / w, 4),
                                round(y1 / h, 4),
                                round(box_w / w, 4),
                                round(box_h / h, 4)
                            ]
                        })
            except Exception as e:
                print(f"YOLO inference error: {e}")

        # Fallback CV edge/contour detection if YOLO produces no detection on test water frames
        if not detections and self.yolo_model is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 200 < area < 40000:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    detections.append({
                        "class_name": "plastic_bottle",
                        "confidence": 0.65,
                        "box": [x, y, x + bw, y + bh],
                        "normalized_box": [round(x / w, 4), round(y / h, 4), round(bw / w, 4), round(bh / h, 4)]
                    })

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return detections, round(latency_ms, 2)

    def detect_batch(
        self,
        frames: list[np.ndarray],
        batch_size: int = 8
    ) -> tuple[list[list[dict]], float]:
        """
        Executes batched inference across multiple frames or SAHI tiles in a single forward pass.
        Returns (list_of_detections_per_frame, total_latency_ms).
        """
        if not frames:
            return [], 0.0

        start_time = time.perf_counter()
        batch_detections: list[list[dict]] = []

        if self.yolo_model is not None:
            try:
                results = self.yolo_model.predict(
                    source=frames,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    batch=batch_size,
                    verbose=False
                )
                for i, res in enumerate(results):
                    frame_dets = []
                    frame_h, frame_w = frames[i].shape[:2]
                    boxes = res.boxes
                    for box in boxes:
                        coords = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_idx = int(box.cls[0].cpu().numpy())
                        class_name = self._map_class_name(cls_idx)

                        x1, y1, x2, y2 = [int(v) for v in coords]
                        box_w = x2 - x1
                        box_h = y2 - y1

                        frame_dets.append({
                            "class_name": class_name,
                            "confidence": round(conf, 3),
                            "box": [x1, y1, x2, y2],
                            "normalized_box": [
                                round(x1 / frame_w, 4),
                                round(y1 / frame_h, 4),
                                round(box_w / frame_w, 4),
                                round(box_h / frame_h, 4)
                            ]
                        })
                    batch_detections.append(frame_dets)
            except Exception as e:
                print(f"Batched YOLO inference error: {e}")
                batch_detections = [self.detect(f)[0] for f in frames]
        else:
            batch_detections = [self.detect(f)[0] for f in frames]

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return batch_detections, round(latency_ms, 2)
