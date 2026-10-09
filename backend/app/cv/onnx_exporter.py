"""
ONNX and TensorRT Model Exporter and Accelerated Inference Runtime
Enables ultra-fast edge inference via ONNX Runtime and TensorRT engine generation.
"""

from pathlib import Path
import time
import numpy as np
import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "data" / "models"

def export_model_to_onnx(model_path: str = "yolov8n.pt", imgsz: int = 640, opset: int = 17) -> Path:
    """
    Exports a PyTorch YOLO model (.pt) to ONNX format with dynamic batching.
    """
    model = YOLO(model_path)
    output_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=True,
        simplify=True
    )
    print(f"Exported model to ONNX: {output_path}")
    return Path(output_path)

def export_model_to_tensorrt(model_path: str = "yolov8n.pt", device: int = 0) -> str:
    """
    Exports YOLO model to TensorRT engine for NVIDIA Jetson / GPU deployment.
    """
    model = YOLO(model_path)
    output_engine = model.export(
        format="engine",
        device=device,
        half=True  # FP16 precision
    )
    print(f"Exported model to TensorRT engine: {output_engine}")
    return str(output_engine)

class ONNXPlasticDetector:
    """
    Inference runtime using ONNX Runtime for high-speed edge execution.
    """
    def __init__(self, onnx_model_path: str, confidence_threshold: float = 0.40):
        import onnxruntime as ort
        self.confidence_threshold = confidence_threshold
        self.session = ort.InferenceSession(
            onnx_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def predict(self, frame: np.ndarray) -> tuple[list[dict], float]:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        # Preprocessing: Resize to 640x640, normalize 0-1, BGR->RGB, NCHW format
        resized = cv2.resize(frame, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        outputs = self.session.run(None, {self.input_name: tensor})
        # Output shape is [1, 84, 8400] for standard YOLOv8
        preds = outputs[0][0]  # shape: (84, 8400)

        detections = []
        # Extract candidate boxes
        boxes = preds[:4, :].T  # cx, cy, w, h
        scores = preds[4:, :].T  # class scores

        for idx in range(boxes.shape[0]):
            max_class_id = int(np.argmax(scores[idx]))
            max_conf = float(scores[idx][max_class_id])

            if max_conf >= self.confidence_threshold:
                cx, cy, bw, bh = boxes[idx]
                x1 = int((cx - bw / 2) * (w / 640.0))
                y1 = int((cy - bh / 2) * (h / 640.0))
                x2 = int((cx + bw / 2) * (w / 640.0))
                y2 = int((cy + bh / 2) * (h / 640.0))

                detections.append({
                    "class_name": "plastic_bottle" if max_class_id == 0 else "plastic_bag",
                    "confidence": round(max_conf, 3),
                    "box": [max(0, x1), max(0, y1), min(w, x2), min(h, y2)],
                    "normalized_box": [round(x1/w, 4), round(y1/h, 4), round((x2-x1)/w, 4), round((y2-y1)/h, 4)]
                })

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return detections, round(latency_ms, 2)
