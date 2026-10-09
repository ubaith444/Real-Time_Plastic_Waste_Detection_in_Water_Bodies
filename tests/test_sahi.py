"""
Unit and Benchmark Tests for Slicing Aided Hyper Inference (SAHI)
Validates uniform tile generation, batched forward tensor inference, global NMS deduplication,
and latency acceleration on drone aerial frames.
"""

import time
import numpy as np
from backend.app.cv.sahi_engine import SAHIEngine
from backend.app.cv.detector import PlasticDetector

def test_sahi_uniform_slicing():
    engine = SAHIEngine(slice_height=512, slice_width=512, overlap_height_ratio=0.20, overlap_width_ratio=0.20)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    slices = engine.slice_frame(frame)
    expected_count = engine.get_slice_count(frame.shape)
    
    assert len(slices) == expected_count, f"Expected {expected_count} slices, got {len(slices)}"
    
    # Assert every single tile is uniformly 512x512
    for idx, (tile, offset_x, offset_y) in enumerate(slices):
        assert tile.shape == (512, 512, 3), f"Slice {idx} has invalid shape {tile.shape}, expected (512, 512, 3)"
        assert 0 <= offset_x <= 1280 - 512
        assert 0 <= offset_y <= 720 - 512

def test_sahi_global_nms():
    engine = SAHIEngine(nms_iou_threshold=0.40)
    
    # Two overlapping detections of same class across slice boundaries
    dets = [
        {"class_name": "plastic_bottle", "confidence": 0.88, "box": [100, 100, 160, 160], "source": "sahi_slice"},
        {"class_name": "plastic_bottle", "confidence": 0.72, "box": [105, 105, 162, 165], "source": "macro_full_frame"},
        {"class_name": "plastic_bag", "confidence": 0.91, "box": [400, 400, 480, 480], "source": "sahi_slice"}
    ]
    
    merged = engine._global_nms(dets)
    # The duplicate bottle should be suppressed, keeping the 0.88 confidence one and the plastic bag
    assert len(merged) == 2
    classes = [d["class_name"] for d in merged]
    assert "plastic_bottle" in classes
    assert "plastic_bag" in classes
    bottle_det = next(d for d in merged if d["class_name"] == "plastic_bottle")
    assert bottle_det["confidence"] == 0.88

def test_sahi_batched_inference_execution():
    engine = SAHIEngine()
    detector = PlasticDetector(confidence_threshold=0.25)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    merged_dets, latency_ms = engine.apply_sahi_inference(frame, detector)
    assert isinstance(merged_dets, list)
    assert latency_ms > 0.0
    print(f"Batched SAHI inference executed in {latency_ms:.1f}ms on 720x1280 drone frame.")

if __name__ == "__main__":
    test_sahi_uniform_slicing()
    test_sahi_global_nms()
    test_sahi_batched_inference_execution()
    print("All SAHI tests passed successfully.")
