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


def _two_island_mask():
    # Island A (larger): [10:110, 10:110] (100x100); Island B (smaller): [10:60, 140:190] (50x50)
    mask = np.zeros((200, 200), dtype=bool)
    mask[10:110, 10:110] = True
    mask[10:60, 140:190] = True
    return mask


def test_contour_service_click_point_selection():
    mask = _two_island_mask()

    # No click: legacy largest-island behavior (Island A)
    res_no_click = ContourService.process_mask(mask, tolerance=0.01)
    assert res_no_click["bounding_box"].xmin == 10
    assert res_no_click["bounding_box"].xmax == 110

    # Click on Island B selects it even though it is smaller (rear-leg bug shape)
    res_click_b = ContourService.process_mask(mask, tolerance=0.01, click_point=(160, 35))
    assert res_click_b["bounding_box"].xmin == 140
    assert res_click_b["bounding_box"].xmax == 190


def test_contour_service_click_speck_guard():
    # 2px noise speck containing the click must NOT hijack the large island
    mask = np.zeros((200, 200), dtype=bool)
    mask[20:170, 20:170] = True
    mask[100:102, 100:102] = False  # tiny hole, not a speck island; add speck nearby
    mask[5:7, 5:7] = True  # 2x2 speck far from the click
    res = ContourService.process_mask(mask, tolerance=0.01, click_point=(100, 100))
    assert res["bounding_box"].xmin == 20
    assert res["bounding_box"].xmax == 170


def test_contour_service_click_inside_speck_falls_back_to_largest():
    # Click enclosed ONLY by a sub-floor speck → guard rejects it → largest wins.
    # Main island lives at [20:70), so (151, 151) is outside it and inside nothing
    # but the 2x2 speck (area 4 < floor max(40, 0.5% of 2500)).
    mask = np.zeros((200, 200), dtype=bool)
    mask[20:70, 20:70] = True
    mask[150:152, 150:152] = True
    res = ContourService.process_mask(mask, tolerance=0.01, click_point=(151, 151))
    assert res["bounding_box"].xmin == 20
    assert res["bounding_box"].xmax == 70


def test_contour_service_click_nearest_within_threshold():
    mask = _two_island_mask()
    # Just outside Island B (edge at x=190): nearest within threshold wins
    res_near = ContourService.process_mask(mask, tolerance=0.01, click_point=(196, 35))
    assert res_near["bounding_box"].xmin == 140


def test_contour_service_click_far_falls_back_to_largest():
    mask = _two_island_mask()
    # Far from both islands (corner (199, 199), diag threshold ~2.8px on 200px canvas
    # vs distances > 60px) → legacy largest fallback
    res_far = ContourService.process_mask(mask, tolerance=0.01, click_point=(199, 199))
    assert res_far["bounding_box"].xmax == 110
