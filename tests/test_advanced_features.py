"""
Unit Tests for Advanced Architecture Enhancements:
1. Dynamic GSD Altitude Link
2. Convex Hull Slick Dispersion and Marine Barrier Containment Boom Metric
3. Decoupled Background Task Worker Queue
4. Multi-Subscriber Telemetry Pub/Sub Bus
5. Active Learning False-Positive Flagging Loop
"""

import numpy as np
from backend.app.cv.segmentation_engine import PlasticAreaEstimator, WaterHomographyCalibrator
from backend.app.services.background_worker import BackgroundWorkerQueue
from backend.app.services.pubsub import TelemetryPubSub
from backend.app.cv.tracker import Track

def test_dynamic_gsd_calculation():
    # Test DJI Phantom 4 Pro / Mavic 3 optics: sensor=13.2mm, f=8.8mm, alt=25m, w=1280px
    gsd_25m = PlasticAreaEstimator.compute_dynamic_gsd(
        sensor_width_mm=13.2,
        altitude_m=25.0,
        focal_length_mm=8.8,
        image_width_px=1280
    )
    # (13.2 * 25 * 100) / (8.8 * 1280) = 33000 / 11264 = 2.9297 cm/pixel
    assert round(gsd_25m, 2) == 2.93

    # Higher altitude increases GSD (each pixel represents larger physical distance)
    gsd_50m = PlasticAreaEstimator.compute_dynamic_gsd(
        sensor_width_mm=13.2,
        altitude_m=50.0,
        focal_length_mm=8.8,
        image_width_px=1280
    )
    assert gsd_50m > gsd_25m
    assert round(gsd_50m, 2) == 5.86

def test_convex_hull_slick_and_containment_boom():
    estimator = PlasticAreaEstimator(default_gsd_cm_per_pixel=2.0)
    
    # Create 4 synthetic debris tracks distributed over a water area
    t1 = Track(1, {"class_name": "plastic_bottle", "box": [100, 100, 140, 150], "confidence": 0.90})
    t2 = Track(2, {"class_name": "plastic_container", "box": [300, 100, 350, 160], "confidence": 0.88})
    t3 = Track(3, {"class_name": "fishing_net_rope", "box": [320, 280, 400, 340], "confidence": 0.92})
    t4 = Track(4, {"class_name": "plastic_bag", "box": [110, 270, 160, 310], "confidence": 0.85})
    
    tracks = [t1, t2, t3, t4]
    frame_shape = (720, 1280, 3)

    coverage_res = estimator.compute_coverage(frame_shape, tracks)
    
    assert "slick_area_m2" in coverage_res
    assert "containment_boom_meters" in coverage_res
    assert "hull_points" in coverage_res
    
    # Slick area and containment boom must be non-zero for multi-debris slick
    assert coverage_res["slick_area_m2"] > 0.0
    assert coverage_res["containment_boom_meters"] > 0.0
    assert len(coverage_res["hull_points"]) >= 3

def test_water_homography_calibrator():
    # Test perspective rectification with 4 corner mapping
    src = [(0, 0), (640, 0), (640, 480), (0, 480)]
    dst = [(50, 20), (590, 20), (640, 480), (0, 480)]
    calibrator = WaterHomographyCalibrator(src, dst)
    assert calibrator.homography_matrix is not None
    
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rectified = calibrator.rectify_frame(test_frame)
    assert rectified.shape == (480, 640, 3)

def test_background_worker_queue():
    worker = BackgroundWorkerQueue(max_queue_size=100)
    stats_before = worker.get_stats()
    assert stats_before["is_alive"] is True
    
    # Enqueue harmless dummy events (empty list)
    worker.enqueue_events_and_alerts([], [])
    stats_after = worker.get_stats()
    assert stats_after["total_dropped"] == 0
    worker.stop()

def test_telemetry_pubsub():
    bus = TelemetryPubSub()
    q1 = bus.subscribe()
    q2 = bus.subscribe()
    assert bus.active_subscriber_count() == 2
    
    bus.publish_telemetry({"test": 123})
    assert q1.qsize() == 1
    assert q2.qsize() == 1
    
    bus.unsubscribe(q1)
    bus.unsubscribe(q2)
    assert bus.active_subscriber_count() == 0

if __name__ == "__main__":
    test_dynamic_gsd_calculation()
    test_convex_hull_slick_and_containment_boom()
    test_water_homography_calibrator()
    test_background_worker_queue()
    test_telemetry_pubsub()
    print("test_advanced_features passed successfully.")

