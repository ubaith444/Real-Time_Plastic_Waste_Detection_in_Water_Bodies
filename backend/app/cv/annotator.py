"""
Computer Vision Visual Annotator
Renders clean bounding boxes, track ID labels, motion trajectory drift vectors,
and environmental monitoring HUD telemetry onto video frames.
"""

import time
import numpy as np
import cv2

CLASS_COLORS = {
    "plastic_bottle": (235, 99, 37),      # BGR for Blue
    "plastic_bag": (136, 148, 13),        # BGR for Teal
    "plastic_container": (6, 119, 217),   # BGR for Amber
    "fishing_net_rope": (38, 38, 220),    # BGR for Red
    "micro_macro_fragment": (234, 51, 147) # BGR for Purple
}

DEFAULT_COLOR = (200, 200, 200)

def annotate_frame(
    frame: np.ndarray,
    tracks: list,
    fps: float,
    latency_ms: float,
    camera_name: str,
    environment_type: str,
    gps_coords: tuple[float, float] | None = None
) -> np.ndarray:
    """
    Renders bounding boxes, drift trails, and telemetry HUD on the frame.
    Returns annotated frame copy.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # 1. Draw drift trajectory vectors and bounding boxes
    for track in tracks:
        color = CLASS_COLORS.get(track.class_name, DEFAULT_COLOR)
        x1, y1, x2, y2 = track.box

        # Draw trajectory history (drift path)
        if len(track.trajectory) > 1:
            for i in range(1, len(track.trajectory)):
                pt1 = track.trajectory[i - 1]
                pt2 = track.trajectory[i]
                alpha = i / len(track.trajectory)
                thickness = max(1, int(alpha * 3))
                cv2.line(annotated, pt1, pt2, color, thickness)

        # Draw bounding box (thinner if in Kalman prediction mode during wave occlusion)
        is_occluded = getattr(track, "is_occluded", False)
        box_thickness = 1 if is_occluded else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

        # Label pill
        if is_occluded:
            label = f"#{track.track_id} {track.class_name.replace('_', ' ').title()} [KF Predict]"
        else:
            vx, vy = getattr(track, "velocity", (0.0, 0.0))
            label = f"#{track.track_id} {track.class_name.replace('_', ' ').title()} {int(track.confidence * 100)}%"
        (lbl_w, lbl_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

        pill_y1 = max(0, y1 - lbl_h - 8)
        pill_y2 = y1
        pill_x2 = min(w, x1 + lbl_w + 10)

        # Background tag
        cv2.rectangle(annotated, (x1, pill_y1), (pill_x2, pill_y2), color, -1)
        # White text
        cv2.putText(
            annotated,
            label,
            (x1 + 5, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Dwell time indicator
        if track.dwell_time_sec > 2.0:
            dwell_str = f"{track.dwell_time_sec}s"
            cv2.putText(
                annotated,
                dwell_str,
                (x1, y2 + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    # 2. Top HUD Telemetry Banner
    hud_bg_height = 42
    cv2.rectangle(annotated, (0, 0), (w, hud_bg_height), (15, 23, 42), -1)

    # Telemetry strings
    status_text = f"CAM: {camera_name} [{environment_type.upper()}]"
    cv2.putText(annotated, status_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    fps_text = f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | Debris Count: {len(tracks)}"
    cv2.putText(annotated, fps_text, (w - 440, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 230, 255), 1, cv2.LINE_AA)

    # Bottom status bar
    bot_y = h - 12
    coords_text = f"GPS: {gps_coords[0]:.4f}, {gps_coords[1]:.4f}" if gps_coords else "GPS: N/A"
    clock_text = time.strftime("%Y-%m-%d %H:%M:%S UTC")
    cv2.putText(annotated, f"{coords_text} | {clock_text}", (15, bot_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)

    return annotated
