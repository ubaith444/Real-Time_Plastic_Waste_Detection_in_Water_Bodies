from backend.app.cv.tracker import MarineTracker, calculate_iou, KalmanFilter2D

def test_iou_calculation():
    box1 = [0, 0, 10, 10]
    box2 = [0, 0, 10, 10]
    box3 = [20, 20, 30, 30]
    assert calculate_iou(box1, box2) > 0.99
    assert calculate_iou(box1, box3) == 0.0

def test_tracker_lifecycle():
    tracker = MarineTracker(max_misses=5, iou_threshold=0.20)

    # Frame 1: One bottle appears
    det1 = [{"class_name": "plastic_bottle", "confidence": 0.85, "box": [100, 100, 140, 140], "normalized_box": [0.1, 0.1, 0.04, 0.04]}]
    tracks = tracker.update(det1)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].hits == 1

    # Frame 2: Bottle moves slightly with wave drift
    det2 = [{"class_name": "plastic_bottle", "confidence": 0.86, "box": [103, 102, 143, 142], "normalized_box": [0.103, 0.102, 0.04, 0.04]}]
    tracks = tracker.update(det2)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].hits == 2

    # Frame 3: A second debris item enters
    det3 = [
        {"class_name": "plastic_bottle", "confidence": 0.86, "box": [105, 103, 145, 143], "normalized_box": [0.105, 0.103, 0.04, 0.04]},
        {"class_name": "plastic_bag", "confidence": 0.90, "box": [400, 400, 480, 480], "normalized_box": [0.4, 0.4, 0.08, 0.08]}
    ]
    tracks = tracker.update(det3)
    assert len(tracks) == 2
    track_ids = {t.track_id for t in tracks}
    assert 1 in track_ids
    assert 2 in track_ids

def test_kalman_filter_occlusion_continuity():
    tracker = MarineTracker(max_misses=8, iou_threshold=0.15, dist_threshold=100.0)

    # 1. Debris is detected drifting downstream for 3 frames
    tracker.update([{"class_name": "plastic_bottle", "confidence": 0.88, "box": [100, 100, 140, 140]}])
    tracker.update([{"class_name": "plastic_bottle", "confidence": 0.89, "box": [104, 102, 144, 142]}])
    tracker.update([{"class_name": "plastic_bottle", "confidence": 0.91, "box": [108, 104, 148, 144]}])

    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].track_id == 1
    assert tracker.tracks[0].hits == 3
    assert tracker.tracks[0].misses == 0

    # 2. Wave crest submerge occlusion: No detection for 2 frames
    tracker.update([])
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].misses == 1
    assert tracker.tracks[0].is_occluded is True

    tracker.update([])
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].misses == 2
    assert tracker.tracks[0].is_occluded is True

    # 3. Wave passes and bottle re-emerges downstream
    re_emerged_dets = [{"class_name": "plastic_bottle", "confidence": 0.87, "box": [118, 108, 158, 148]}]
    tracks_after = tracker.update(re_emerged_dets)

    # The tracker MUST associate with the existing Track ID 1 via Kalman prediction
    assert len(tracks_after) == 1
    assert tracks_after[0].track_id == 1, f"Expected track_id 1, got {tracks_after[0].track_id}"
    assert tracks_after[0].hits == 4
    assert tracks_after[0].misses == 0
    assert tracks_after[0].is_occluded is False

def test_kalman_velocity_and_smoothing():
    kf = KalmanFilter2D(initial_box=[50, 50, 100, 100], current_velocity=(2.0, 1.0))
    pred_box = kf.predict(water_current=(2.0, 1.0))
    assert len(pred_box) == 4
    # State should advance along velocity
    assert pred_box[0] > 50
    assert pred_box[1] > 50

    # Measurement update
    kf.update([55, 53, 105, 103])
    vx, vy = kf.get_velocity()
    assert isinstance(vx, float)
    assert isinstance(vy, float)

if __name__ == "__main__":
    test_iou_calculation()
    test_tracker_lifecycle()
    test_kalman_filter_occlusion_continuity()
    test_kalman_velocity_and_smoothing()
    print("test_tracker passed successfully.")
