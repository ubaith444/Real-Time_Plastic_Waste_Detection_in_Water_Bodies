"""
Polygon Instance Segmentation and Surface Area Coverage Estimation Engine
Calculates exact pixel-level and physical (m²) water surface area covered by plastic waste.
Classifies aquatic pollution density from clean surface to severe plastic choke.
"""

import numpy as np
import cv2

class PlasticAreaEstimator:
    def __init__(self, default_gsd_cm_per_pixel: float = 1.2):
        """
        gsd_cm_per_pixel: Ground Sampling Distance (GSD) in cm/pixel.
        Default 1.2 cm/px corresponds to typical drone altitude of 25-30 meters.
        """
        self.gsd_cm_per_pixel = default_gsd_cm_per_pixel

    @staticmethod
    def compute_dynamic_gsd(
        sensor_width_mm: float,
        altitude_m: float,
        focal_length_mm: float,
        image_width_px: int
    ) -> float:
        """
        Computes dynamic Ground Sampling Distance (GSD) in cm/pixel:
        GSD = (sensor_width * altitude * 100) / (focal_length * image_width)
        """
        if focal_length_mm <= 0 or image_width_px <= 0:
            return 1.20
        gsd = (sensor_width_mm * max(1.0, altitude_m) * 100.0) / (focal_length_mm * image_width_px)
        return round(float(gsd), 4)

    def compute_convex_hull_slick(
        self,
        frame_shape: tuple[int, int],
        tracks: list,
        gsd_cm_per_px: float | None = None
    ) -> dict:
        """
        Computes the bounding convex hull enclosing all active debris tracks.
        Estimates total dispersion slick area (m²) and barrier containment boom
        perimeter (meters) for marine cleanup crews.
        """
        gsd = gsd_cm_per_px or self.gsd_cm_per_pixel
        h, w = frame_shape[:2]

        if not tracks or len(tracks) < 2:
            return {
                "slick_area_m2": 0.0,
                "containment_boom_meters": 0.0,
                "hull_point_count": 0,
                "hull_points": []
            }

        # Gather all corner vertices of active debris bounding boxes
        points = []
        for t in tracks:
            x1, y1, x2, y2 = t.box
            points.extend([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ])

        pts_arr = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(pts_arr)

        area_px = cv2.contourArea(hull)
        perimeter_px = cv2.arcLength(hull, closed=True)

        # Convert cm² to m²: (pixels * (cm/px)^2) / 10,000
        slick_area_m2 = round((area_px * (gsd ** 2)) / 10000.0, 2)
        # Convert cm to meters: (pixels * cm/px) / 100
        containment_boom_meters = round((perimeter_px * gsd) / 100.0, 2)

        hull_coords = hull.reshape(-1, 2).tolist()

        return {
            "slick_area_m2": slick_area_m2,
            "containment_boom_meters": containment_boom_meters,
            "hull_point_count": len(hull_coords),
            "hull_points": hull_coords
        }

    def compute_coverage(
        self,
        frame_shape: tuple[int, int],
        tracks: list,
        polygon_masks: list[np.ndarray] | None = None,
        custom_gsd: float | None = None
    ) -> dict:
        """
        Calculates pixel area and physical square meter area covered by plastic,
        as well as debris cluster convex hull slick boundary metrics.
        """
        gsd = custom_gsd or self.gsd_cm_per_pixel
        h, w = frame_shape[:2]
        total_pixels = h * w

        plastic_pixels = 0
        per_class_pixels: dict[str, int] = {}

        if polygon_masks and len(polygon_masks) > 0:
            # Union of binary polygon masks
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for mask in polygon_masks:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            plastic_pixels = int(np.count_nonzero(combined_mask))
        else:
            # Estimate area from confirmed track bounding boxes with shape fill factors
            # (e.g. bottles ~ 0.70 fill factor, thin plastic bags ~ 0.55, containers ~ 0.85)
            fill_factors = {
                "plastic_bottle": 0.70,
                "plastic_bag": 0.55,
                "plastic_packaging": 0.85,
                "plastic_container": 0.85,
                "fishing_net_rope": 0.40,
                "plastic_fragment": 0.65,
                "other_floating_waste": 0.60
            }

            for t in tracks:
                x1, y1, x2, y2 = t.box
                bw = max(0, x2 - x1)
                bh = max(0, y2 - y1)
                box_area = bw * bh
                factor = fill_factors.get(t.class_name, 0.65)
                est_item_px = int(box_area * factor)
                plastic_pixels += est_item_px

                per_class_pixels[t.class_name] = per_class_pixels.get(t.class_name, 0) + est_item_px

        coverage_percentage = round((plastic_pixels / total_pixels) * 100.0, 3)

        # Convert cm² to m²: (pixels * (cm/px)^2) / 10,000
        cm_sq_per_px = gsd ** 2
        area_sq_meters = round((plastic_pixels * cm_sq_per_px) / 10000.0, 4)

        # Debris slick convex hull containment metric
        slick_hull = self.compute_convex_hull_slick(frame_shape, tracks, gsd_cm_per_px=gsd)

        # Determine environmental pollution density rating
        if coverage_percentage < 0.05:
            density_status = "Clean Surface / Trace"
            density_badge = "badge-low"
        elif coverage_percentage < 0.50:
            density_status = "Low Dispersion"
            density_badge = "badge-medium"
        elif coverage_percentage < 2.00:
            density_status = "Moderate Plastic Accumulation"
            density_badge = "badge-high"
        else:
            density_status = "Severe Plastic Choke"
            density_badge = "badge-critical"

        return {
            "total_pixels": total_pixels,
            "plastic_pixels": plastic_pixels,
            "coverage_percentage": coverage_percentage,
            "area_sq_meters": area_sq_meters,
            "density_status": density_status,
            "density_badge": density_badge,
            "per_class_pixels": per_class_pixels,
            "slick_area_m2": slick_hull["slick_area_m2"],
            "containment_boom_meters": slick_hull["containment_boom_meters"],
            "hull_points": slick_hull["hull_points"]
        }


class WaterHomographyCalibrator:
    """
    Computes and applies perspective homography matrix to rectify oblique
    camera angles (e.g. shoreline or boat mount) to an orthorectified top-down view.
    """
    def __init__(self, src_pts: list[tuple[float, float]] | None = None, dst_pts: list[tuple[float, float]] | None = None):
        self.homography_matrix: np.ndarray | None = None
        if src_pts and dst_pts and len(src_pts) == 4 and len(dst_pts) == 4:
            src = np.array(src_pts, dtype=np.float32)
            dst = np.array(dst_pts, dtype=np.float32)
            self.homography_matrix, _ = cv2.findHomography(src, dst)

    def rectify_frame(self, frame: np.ndarray, target_shape: tuple[int, int] | None = None) -> np.ndarray:
        if self.homography_matrix is None:
            return frame
        h, w = target_shape or frame.shape[:2]
        return cv2.warpPerspective(frame, self.homography_matrix, (w, h))

