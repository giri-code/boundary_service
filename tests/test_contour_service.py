import numpy as np
from app.services.contour_service import ContourService

def test_contour_service_square():
    # 200x200 canvas with a 100x100 filled square at (50, 50)
    mask = np.zeros((200, 200), dtype=bool)
    mask[50:150, 50:150] = True

    result = ContourService.process_mask(mask, tolerance=0.01)

    assert len(result["outer_boundary"]) >= 4
    assert result["area_pixels"] > 9000
    assert result["bounding_box"].xmin == 50
    assert result["bounding_box"].ymin == 50
    assert result["bounding_box"].xmax == 150
    assert result["bounding_box"].ymax == 150

def test_contour_service_with_hole():
    # 200x200 canvas with a outer square and an inner hole
    mask = np.zeros((200, 200), dtype=bool)
    mask[20:180, 20:180] = True
    mask[70:130, 70:130] = False  # Cutout hole

    result = ContourService.process_mask(mask, tolerance=0.005)

    assert len(result["outer_boundary"]) >= 4
    assert len(result["holes"]) >= 1
    assert len(result["holes"][0]) >= 4
