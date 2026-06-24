import numpy as np
from geometry import mask_to_box, box_area, box_iou, mask_iou

def _mask(x0, y0, x1, y1, h=10, w=10):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m

def test_box_area():
    # box_area uses pixel-inclusive math: (x2-x1+1)*(y2-y1+1)
    # box [0,0,2,3] -> w=(2-0+1)=3, h=(3-0+1)=4 -> 12
    assert box_area([0, 0, 2, 3]) == 12.0

def test_box_iou_identical_is_one():
    assert box_iou([0, 0, 4, 4], [0, 0, 4, 4]) == 1.0

def test_box_iou_disjoint_is_zero():
    assert box_iou([0, 0, 1, 1], [5, 5, 6, 6]) == 0.0

def test_mask_to_box_tight_bounds():
    # _mask(2,3,6,7) sets m[3:7, 2:6] so active xs=2..5, ys=3..6
    # mask_to_box returns (xs.min, ys.min, xs.max, ys.max) = (2,3,5,6)
    box = mask_to_box(_mask(2, 3, 6, 7))
    assert list(box) == [2.0, 3.0, 5.0, 6.0]

def test_mask_iou_identical_is_one():
    m = _mask(0, 0, 5, 5)
    assert mask_iou(m, m) == 1.0

def test_mask_iou_disjoint_is_zero():
    assert mask_iou(_mask(0, 0, 2, 2), _mask(5, 5, 8, 8)) == 0.0
