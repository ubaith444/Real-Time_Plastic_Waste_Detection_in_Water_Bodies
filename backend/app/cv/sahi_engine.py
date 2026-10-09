"""
Slicing Aided Hyper Inference (SAHI) Engine
Enables high-recall detection of small floating plastic debris from high-altitude aerial drones.
Slices high-resolution frames into overlapping windows, runs inference, and merges predictions via global NMS.
"""

import time
import numpy as np
from backend.app.cv.tracker import calculate_iou

class SAHIEngine:
    def __init__(
        self,
        slice_height: int = 512,
        slice_width: int = 512,
        overlap_height_ratio: float = 0.20,
        overlap_width_ratio: float = 0.20,
        nms_iou_threshold: float = 0.45
    ):
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.nms_iou_threshold = nms_iou_threshold

    def slice_frame(self, frame: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
        """
        Slices the frame into overlapping uniform sub-windows.
        Guarantees all tiles are slice_width x slice_height by shifting boundary windows backwards.
        Returns: list of (slice_image, x_offset, y_offset).
        """
        h, w = frame.shape[:2]
        step_x = max(1, int(self.slice_width * (1.0 - self.overlap_width_ratio)))
        step_y = max(1, int(self.slice_height * (1.0 - self.overlap_height_ratio)))

        y_coords = []
        y = 0
        while y < h:
            if y + self.slice_height >= h:
                y_coords.append(max(0, h - self.slice_height))
                break
            y_coords.append(y)
            y += step_y

        x_coords = []
        x = 0
        while x < w:
            if x + self.slice_width >= w:
                x_coords.append(max(0, w - self.slice_width))
                break
            x_coords.append(x)
            x += step_x

        slices = []
        for sy in y_coords:
            for sx in x_coords:
                cropped = frame[sy:sy + self.slice_height, sx:sx + self.slice_width]
                ch, cw = cropped.shape[:2]
                if ch < self.slice_height or cw < self.slice_width:
                    padded = np.zeros((self.slice_height, self.slice_width, cropped.shape[2]), dtype=cropped.dtype)
                    padded[:ch, :cw] = cropped
                    slices.append((padded, sx, sy))
                else:
                    slices.append((cropped, sx, sy))

        return slices

    def get_slice_count(self, frame_shape: tuple[int, int]) -> int:
        """
        Calculates the number of tiles generated for a given frame dimension (h, w).
        """
        h, w = frame_shape[:2]
        step_x = max(1, int(self.slice_width * (1.0 - self.overlap_width_ratio)))
        step_y = max(1, int(self.slice_height * (1.0 - self.overlap_height_ratio)))

        ny = 1 if h <= self.slice_height else int(np.ceil((h - self.slice_height) / step_y)) + 1
        nx = 1 if w <= self.slice_width else int(np.ceil((w - self.slice_width) / step_x)) + 1
        return ny * nx

    def apply_sahi_inference(
        self,
        frame: np.ndarray,
        base_detector,
        synthetic_detections: list[dict] | None = None
    ) -> tuple[list[dict], float]:
        """
        Executes batched sliced inference across all overlapping windows in a single forward pass
        and merges with global frame detections via global NMS.
        """
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        # If synthetic marine stream is providing ground truth
        if synthetic_detections is not None:
            filtered = [d for d in synthetic_detections if d["confidence"] >= base_detector.confidence_threshold]
            latency = (time.perf_counter() - t0) * 1000.0
            return filtered, round(latency, 2)

        raw_detections = []

        # 1. Full-frame inference (for macro/medium objects)
        full_dets, _ = base_detector.detect(frame)
        for d in full_dets:
            d["source"] = "macro_full_frame"
        raw_detections.extend(full_dets)

        # 2. Extract uniform slices
        slices = self.slice_frame(frame)
        if not slices:
            return raw_detections, round((time.perf_counter() - t0) * 1000.0, 2)

        slice_imgs = [s[0] for s in slices]
        slice_offsets = [(s[1], s[2]) for s in slices]

        # 3. Batched slice inference in a single forward pass
        if hasattr(base_detector, "detect_batch"):
            batch_results, _ = base_detector.detect_batch(slice_imgs, batch_size=len(slice_imgs))
        else:
            batch_results = [base_detector.detect(img)[0] for img in slice_imgs]

        # 4. Project coordinates from slice-local to global frame space
        for (offset_x, offset_y), slice_dets in zip(slice_offsets, batch_results):
            for det in slice_dets:
                sx1, sy1, sx2, sy2 = det["box"]
                gx1 = min(w, max(0, offset_x + sx1))
                gy1 = min(h, max(0, offset_y + sy1))
                gx2 = min(w, max(0, offset_x + sx2))
                gy2 = min(h, max(0, offset_y + sy2))

                box_w = max(1, gx2 - gx1)
                box_h = max(1, gy2 - gy1)

                raw_detections.append({
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "box": [gx1, gy1, gx2, gy2],
                    "normalized_box": [
                        round(gx1 / w, 4),
                        round(gy1 / h, 4),
                        round(box_w / w, 4),
                        round(box_h / h, 4)
                    ],
                    "source": "sahi_slice"
                })

        # 5. Global Non-Maximum Suppression (NMS) to eliminate duplicate boundary boxes
        merged_detections = self._global_nms(raw_detections)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return merged_detections, round(latency_ms, 2)

    def _global_nms(self, detections: list[dict]) -> list[dict]:
        if not detections:
            return []

        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        keep = []

        for candidate in sorted_dets:
            should_keep = True
            for existing in keep:
                if candidate["class_name"] == existing["class_name"]:
                    iou = calculate_iou(candidate["box"], existing["box"])
                    if iou > self.nms_iou_threshold:
                        should_keep = False
                        break
            if should_keep:
                keep.append(candidate)

        return keep
