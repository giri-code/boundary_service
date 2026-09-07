from typing import List, Dict, Any
import cv2
import numpy as np
from ..schemas.boundary import Point, BoundingBox
from ..config import settings


class ContourService:
    """Contour & Hole Extraction Engine with Douglas-Peucker Polygon Approximation."""

    @staticmethod
    def process_mask(
        binary_mask: np.ndarray, tolerance: float = 0.005
    ) -> Dict[str, Any]:
        """Convert binary mask to outer boundary polygon vertices and inner hole polygons.

        Args:
            binary_mask: 2D boolean or uint8 mask array (H, W). Any non-zero value is
                         treated as foreground.
            tolerance:   RDP polygon approximation tolerance multiplier (e.g. 0.005).
                         Higher values produce fewer vertices.

        Returns:
            Dict with keys:
                - outer_boundary: List[Point]
                - holes:          List[List[Point]]
                - bounding_box:   BoundingBox
                - area_pixels:    int
        """
        # FIX BUG-1: Explicit two-step cast — safe regardless of input dtype
        # (bool → uint8 is lossless; float masks are clamped to [0, 1] then scaled)
        if binary_mask.dtype == bool:
            mask_uint8 = binary_mask.view(np.uint8) * 255
        else:
            mask_uint8 = (np.clip(binary_mask, 0, 1).astype(np.uint8)) * 255

        # RETR_CCOMP: 2-level hierarchy — outer contours (parent == -1) and holes
        contours, hierarchy = cv2.findContours(
            mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        empty_result: Dict[str, Any] = {
            "outer_boundary": [],
            "holes": [],
            "bounding_box": BoundingBox(xmin=0, ymin=0, xmax=0, ymax=0),
            "area_pixels": 0,
        }

        if not contours or hierarchy is None:
            return empty_result

        hierarchy = hierarchy[0]  # shape: (N, 4) — [Next, Prev, First_Child, Parent]

        # ── Identify largest outer contour ────────────────────────────────────
        outer_indices = [i for i in range(len(contours)) if hierarchy[i][3] == -1]
        if not outer_indices:
            outer_indices = list(range(len(contours)))

        # FIX PERF-5 & BUG-5: compute area once per candidate and reuse it
        outer_areas = {i: cv2.contourArea(contours[i]) for i in outer_indices}
        main_outer_idx = max(outer_indices, key=lambda i: outer_areas[i])
        main_contour = contours[main_outer_idx]
        area_pixels = int(outer_areas[main_outer_idx])

        # FIX FUNC-2: guard against degenerate (zero-perimeter) contours
        arc_len = cv2.arcLength(main_contour, True)
        if arc_len == 0 or len(main_contour) < 3:
            return empty_result

        # ── Bounding box ──────────────────────────────────────────────────────
        x, y, bw, bh = cv2.boundingRect(main_contour)
        bounding_box = BoundingBox(
            xmin=int(x), ymin=int(y), xmax=int(x + bw), ymax=int(y + bh)
        )

        # ── Simplify outer boundary ───────────────────────────────────────────
        epsilon = max(1.0, tolerance * arc_len)
        approx_outer = cv2.approxPolyDP(main_contour, epsilon, True)
        outer_points = [
            Point(x=int(pt[0][0]), y=int(pt[0][1])) for pt in approx_outer
        ]

        # ── Extract inner holes (children of main_outer_idx) ─────────────────
        holes: List[List[Point]] = []
        for i in range(len(contours)):
            if hierarchy[i][3] != main_outer_idx:
                continue
            hole_contour = contours[i]
            if cv2.contourArea(hole_contour) <= settings.MIN_HOLE_AREA_PIXELS:  # skip noise below configured threshold
                continue
            # FIX PERF-5: compute arc length once per hole
            hole_arc = cv2.arcLength(hole_contour, True)
            if hole_arc == 0:
                continue
            hole_epsilon = max(1.0, tolerance * hole_arc)
            approx_hole = cv2.approxPolyDP(hole_contour, hole_epsilon, True)
            hole_points = [
                Point(x=int(pt[0][0]), y=int(pt[0][1])) for pt in approx_hole
            ]
            if len(hole_points) >= 3:
                holes.append(hole_points)

        return {
            "outer_boundary": outer_points,
            "holes": holes,
            "bounding_box": bounding_box,
            "area_pixels": area_pixels,
        }
