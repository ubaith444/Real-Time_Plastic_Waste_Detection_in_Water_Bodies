import numpy as np
from backend.app.cv.detector import PlasticDetector

def test_detector_initialization():
    detector = PlasticDetector(confidence_threshold=0.35)
    assert detector.confidence_threshold == 0.35
    assert len(detector.class_names) == 5

def test_synthetic_detection_passthrough():
    detector = PlasticDetector(confidence_threshold=0.50)
    fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    sample_detections = [
        {"class_name": "plastic_bottle", "confidence": 0.85, "box": [10, 10, 50, 50], "normalized_box": [0.01, 0.01, 0.04, 0.04]},
        {"class_name": "plastic_bag", "confidence": 0.30, "box": [100, 100, 150, 150], "normalized_box": [0.1, 0.1, 0.05, 0.05]}
    ]
    dets, latency = detector.detect(fake_frame, synthetic_detections=sample_detections)
    # The 0.30 confidence item should be filtered out by 0.50 threshold
    assert len(dets) == 1
    assert dets[0]["class_name"] == "plastic_bottle"
def test_detector_batch_predict():
    detector = PlasticDetector(confidence_threshold=0.25)
    test_slices = [np.zeros((512, 512, 3), dtype=np.uint8) for _ in range(4)]
    batch_results, batch_latency = detector.detect_batch(test_slices, batch_size=4)
    assert len(batch_results) == 4
    assert isinstance(batch_results, list)
    assert batch_latency >= 0.0

if __name__ == "__main__":
    test_detector_initialization()
    test_synthetic_detection_passthrough()
    test_detector_batch_predict()
    print("test_detector passed.")
