import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure app package is importable
MODULE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE_ROOT))

from app.storage.factory import StorageProviderFactory
from app.models.factory import SegmentationModelFactory
from app.services.contour_service import ContourService

def run_visual_test(image_path: str = None, click_x: int = 300, click_y: int = 200):
    if image_path is None:
        image_path = str(MODULE_ROOT / "storage" / "sample_hat.png")

    print(f"Loading image from: {image_path}")
    image = StorageProviderFactory.read_image(image_path)
    h, w = image.shape[:2]

    # Model engine prediction (MobileSAM-only)
    engine = SegmentationModelFactory.get_engine("mobile_sam")
    binary_mask, confidence = engine.predict_mask(image, (click_x, click_y), cache_key=image_path)

    # Contour processing
    result = ContourService.process_mask(binary_mask, tolerance=0.005)

    print(f"\n--- Segmentation Results ---")
    print(f"Model Engine: {engine.name}")
    print(f"Confidence: {confidence}")
    print(f"Outer Boundary Vertices: {len(result['outer_boundary'])}")
    print(f"Holes Found: {len(result['holes'])}")
    print(f"Bounding Box: {result['bounding_box']}")
    print(f"Area (pixels): {result['area_pixels']}")

    # Draw visual overlays
    visual = image.copy()

    # Draw click point (yellow target cross)
    cv2.circle(visual, (click_x, click_y), 6, (0, 255, 255), -1)
    cv2.drawMarker(visual, (click_x, click_y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    # Draw outer boundary polygon (bright neon green)
    if result["outer_boundary"]:
        outer_pts = np.array([[pt.x, pt.y] for pt in result["outer_boundary"]], np.int32)
        cv2.polylines(visual, [outer_pts], isClosed=True, color=(0, 255, 0), thickness=3)

        # Draw vertex points (bright red dots)
        for pt in result["outer_boundary"]:
            cv2.circle(visual, (pt.x, pt.y), 3, (0, 0, 255), -1)

    # Draw inner hole polygons (bright cyan)
    for hole in result["holes"]:
        hole_pts = np.array([[pt.x, pt.y] for pt in hole], np.int32)
        cv2.polylines(visual, [hole_pts], isClosed=True, color=(255, 255, 0), thickness=2)

    # Draw bounding box (magenta dashed)
    bbox = result["bounding_box"]
    cv2.rectangle(visual, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (255, 0, 255), 1)

    output_path = MODULE_ROOT / "storage" / "output_boundary_visual.png"
    cv2.imwrite(str(output_path), visual)
    print(f"\nVisual verification saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    run_visual_test()
