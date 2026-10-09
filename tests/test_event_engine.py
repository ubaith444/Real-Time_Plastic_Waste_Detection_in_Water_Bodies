import numpy as np
from backend.app.cv.tracker import Track
from backend.app.cv.event_engine import EventEngine

def test_duplicate_suppression():
    engine = EventEngine(suppression_window_sec=10.0)
    fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    det = {"class_name": "plastic_bottle", "confidence": 0.88, "box": [100, 100, 200, 200], "normalized_box": [0.1, 0.1, 0.1, 0.1]}
    track = Track(track_id=1, detection=det)
    track.hits = 2  # confirmed hits

    # First event processing
    events1, alerts1 = engine.process_tracks("cam-test", [track], fake_frame)
    assert len(events1) == 1
    assert events1[0]["track_id"] == 1

    # Immediate second frame with same track (should be suppressed by duplicate filter)
    events2, alerts2 = engine.process_tracks("cam-test", [track], fake_frame)
    assert len(events2) == 0  # suppressed!

def test_critical_alert_for_fishing_net():
    engine = EventEngine()
    fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    det = {"class_name": "fishing_net_rope", "confidence": 0.92, "box": [50, 50, 250, 250], "normalized_box": [0.05, 0.05, 0.2, 0.2]}
    track = Track(track_id=42, detection=det)
    track.hits = 3

    events, alerts = engine.process_tracks("boat-01", [track], fake_frame)
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "Critical"
    assert "fishing net" in alerts[0]["message"].lower()

if __name__ == "__main__":
    test_duplicate_suppression()
    test_critical_alert_for_fishing_net()
    print("test_event_engine passed.")
