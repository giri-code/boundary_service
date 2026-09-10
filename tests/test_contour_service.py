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

def test_contour_service_non_contiguous_and_float():
    # Non-contiguous (strided) and Fortran-order bool masks must not crash
    # or corrupt: .astype handles any layout, .view does not.
    base = np.zeros((200, 200), dtype=bool)
    base[50:150, 50:150] = True
    strided = base[::2, ::2]
    assert not strided.flags.c_contiguous
    result = ContourService.process_mask(strided, tolerance=0.01)
    assert len(result["outer_boundary"]) >= 4
    assert result["area_pixels"] > 2000

    fortran = np.asfortranarray(base)
    assert not fortran.flags.c_contiguous
    result_f = ContourService.process_mask(fortran, tolerance=0.01)
    assert result_f["bounding_box"].xmin == 50
    assert result_f["area_pixels"] > 9000

    # Float masks are clamped to [0, 1]
    result_fl = ContourService.process_mask(base.astype(np.float32), tolerance=0.01)
    assert result_fl["area_pixels"] > 9000
