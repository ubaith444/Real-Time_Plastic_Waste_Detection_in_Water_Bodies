"""
Aquatic Environment Simulators
Provides realistic synthetic marine video streams for Drone, Boat, Underwater, and Surface Tank.
Each simulator accurately mimics lighting, wave motion, color attenuation, and floating/submerged plastic waste.
"""

import math
import random
import time
import numpy as np
import cv2

class DebrisObject:
    def __init__(self, class_name: str, x: float, y: float, vx: float, vy: float, size: float, env_type: str):
        self.class_name = class_name
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.size = size
        self.env_type = env_type
        self.phase = random.random() * math.pi * 2
        self.rotation = random.random() * 360

    def update(self, dt: float, current_time: float):
        # Wave bobbing and current drift
        bob = math.sin(current_time * 2.0 + self.phase) * 0.002
        self.x += self.vx * dt
        self.y += (self.vy + bob) * dt
        self.rotation += 5.0 * dt

        # Wrap around edges
        if self.x > 1.15:
            self.x = -0.15
            self.y = random.uniform(0.15, 0.85)
        elif self.x < -0.15:
            self.x = 1.15
            self.y = random.uniform(0.15, 0.85)

        if self.y > 1.15:
            self.y = -0.15
        elif self.y < -0.15:
            self.y = 1.15

class MarineEnvironmentSimulator:
    def __init__(self, env_type: str = "drone", width: int = 1280, height: int = 720):
        self.env_type = env_type
        self.width = width
        self.height = height
        self.objects: list[DebrisObject] = []
        self._init_debris()
        self.last_update = time.time()

    def _init_debris(self):
        self.objects.clear()
        if self.env_type == "drone":
            # High altitude tiny objects
            configs = [
                ("plastic_bag", 0.25, 0.35, 0.015, 0.008, 0.035),
                ("plastic_bottle", 0.65, 0.45, 0.012, -0.006, 0.028),
                ("micro_macro_fragment", 0.45, 0.70, 0.020, 0.010, 0.022),
                ("plastic_container", 0.80, 0.25, 0.018, 0.005, 0.040),
            ]
        elif self.env_type == "boat":
            # Wave motion with prominent bottles, containers, ropes
            configs = [
                ("plastic_bottle", 0.30, 0.40, 0.030, 0.012, 0.075),
                ("plastic_container", 0.70, 0.55, 0.025, -0.010, 0.090),
                ("plastic_bag", 0.50, 0.30, 0.022, 0.015, 0.065),
                ("fishing_net_rope", 0.85, 0.65, 0.018, -0.008, 0.120),
            ]
        elif self.env_type == "underwater":
            # Submerged nets, ropes, bags with color distortion
            configs = [
                ("fishing_net_rope", 0.35, 0.45, 0.012, 0.005, 0.140),
                ("plastic_bag", 0.60, 0.60, 0.010, -0.008, 0.080),
                ("micro_macro_fragment", 0.75, 0.35, 0.015, 0.012, 0.055),
            ]
        else: # webcam / test tank
            configs = [
                ("plastic_bottle", 0.40, 0.45, 0.020, 0.010, 0.085),
                ("plastic_container", 0.65, 0.35, 0.018, -0.012, 0.070),
                ("plastic_bag", 0.25, 0.60, 0.025, 0.008, 0.060),
            ]

        for cls_name, x, y, vx, vy, sz in configs:
            self.objects.append(DebrisObject(cls_name, x, y, vx, vy, sz, self.env_type))

    def generate_frame(self) -> tuple[np.ndarray, list[dict]]:
        """
        Renders the realistic water background, waves, lighting,
        and returns the RGB frame along with accurate ground-truth bounding boxes.
        """
        now = time.time()
        dt = min(now - self.last_update, 0.1)
        self.last_update = now

        for obj in self.objects:
            obj.update(dt, now)

        w, h = self.width, self.height

        # 1. Base water background rendering
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        if self.env_type == "underwater":
            # Deep aquatic blue-turquoise with vertical depth falloff
            for y in range(h):
                alpha = y / h
                b = int(140 + alpha * 30)
                g = int(90 - alpha * 20)
                r = int(15 - alpha * 10)
                frame[y, :] = [b, g, r]

            # Drifting particulate speckles
            np.random.seed(int(now * 10) % 1000)
            speckle_x = np.random.randint(0, w, 60)
            speckle_y = np.random.randint(0, h, 60)
            for sx, sy in zip(speckle_x, speckle_y):
                cv2.circle(frame, (sx, sy), 2, (180, 160, 90), -1)

        elif self.env_type == "drone":
            # Dal Lake aerial green-blue with sun ripples
            frame[:] = [95, 115, 65]
            for wave_idx in range(12):
                y_offset = int((wave_idx * 65 + now * 25) % h)
                pts = []
                for x in range(0, w, 40):
                    y_ripple = int(y_offset + math.sin(x * 0.02 + now * 1.5) * 8)
                    pts.append([x, y_ripple])
                pts = np.array(pts, np.int32)
                cv2.polylines(frame, [pts], False, (115, 135, 80), 2)

        elif self.env_type == "boat":
            # Ocean deep navy with rolling swells and whitecaps
            frame[:] = [150, 85, 30]
            for swell in range(8):
                y_offset = int((swell * 95 + now * 40) % h)
                pts = []
                for x in range(0, w, 30):
                    y_swell = int(y_offset + math.sin(x * 0.015 + now * 2.0) * 14)
                    pts.append([x, y_swell])
                pts = np.array(pts, np.int32)
                cv2.polylines(frame, [pts], False, (190, 120, 60), 3)

            # Sunlight glare reflection patch
            glare_cx = int(w * 0.70 + math.sin(now) * 50)
            glare_cy = int(h * 0.30)
            cv2.ellipse(frame, (glare_cx, glare_cy), (180, 60), 15, 0, 360, (210, 180, 150), -1)

        else: # webcam / surface tank
            # Calm pool surface with ambient lighting
            frame[:] = [170, 140, 95]
            for ripple in range(6):
                r_rad = int((now * 30 + ripple * 80) % 250) + 10
                cv2.circle(frame, (int(w * 0.5), int(h * 0.5)), r_rad, (190, 160, 115), 1)

        # 2. Render debris objects and collect bounding boxes
        detected_boxes = []

        for obj in self.objects:
            cx_pix = int(obj.x * w)
            cy_pix = int(obj.y * h)
            sz_pix = int(obj.size * min(w, h))

            if cx_pix < -sz_pix or cx_pix > w + sz_pix or cy_pix < -sz_pix or cy_pix > h + sz_pix:
                continue

            x1 = max(0, cx_pix - sz_pix // 2)
            y1 = max(0, cy_pix - sz_pix // 2)
            x2 = min(w, cx_pix + sz_pix // 2)
            y2 = min(h, cy_pix + sz_pix // 2)

            box_w = x2 - x1
            box_h = y2 - y1

            if box_w < 5 or box_h < 5:
                continue

            # Draw visual representation based on debris type
            if obj.class_name == "plastic_bottle":
                # Cylindrical bottle with cap
                cv2.rectangle(frame, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), (220, 220, 240), -1)
                cv2.circle(frame, ((x1 + x2) // 2, y1 + 4), max(2, sz_pix // 8), (40, 120, 230), -1)
                # Specular reflection line
                cv2.line(frame, (x1 + 8, y1 + 6), (x1 + 8, y2 - 6), (255, 255, 255), 2)
            elif obj.class_name == "plastic_bag":
                # Irregular semi-translucent polygon
                pts = np.array([
                    [x1 + box_w // 4, y1],
                    [x2 - box_w // 5, y1 + box_h // 3],
                    [x2, y2 - box_h // 4],
                    [x1 + box_w // 2, y2],
                    [x1, y2 - box_h // 3]
                ], np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (235, 245, 245))
                cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
            elif obj.class_name == "plastic_container":
                # Polystyrene cup or tray
                cv2.rectangle(frame, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (245, 240, 235), -1)
                cv2.rectangle(frame, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (180, 160, 140), 2)
            elif obj.class_name == "fishing_net_rope":
                # Tangled net mesh
                cv2.line(frame, (x1, y1), (x2, y2), (40, 40, 190), 3)
                cv2.line(frame, (x1, y2), (x2, y1), (40, 40, 190), 3)
                cv2.circle(frame, (cx_pix, cy_pix), max(4, sz_pix // 4), (30, 30, 160), 2)
            else: # micro_macro_fragment
                # Jagged fragment
                pts = np.array([
                    [x1, y1 + box_h // 2],
                    [x1 + box_w // 3, y1],
                    [x2, y1 + box_h // 4],
                    [x2 - box_w // 4, y2],
                    [x1 + box_w // 4, y2]
                ], np.int32)
                cv2.fillPoly(frame, [pts], (180, 120, 210))

            # Add ground-truth detection entry
            confidence = round(random.uniform(0.78, 0.96), 3)
            detected_boxes.append({
                "class_name": obj.class_name,
                "confidence": confidence,
                "box": [x1, y1, x2, y2],
                "normalized_box": [
                    round(x1 / w, 4),
                    round(y1 / h, 4),
                    round(box_w / w, 4),
                    round(box_h / h, 4)
                ]
            })

        return frame, detected_boxes
