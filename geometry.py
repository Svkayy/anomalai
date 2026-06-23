import numpy as np
import cv2


def mask_to_box(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None  # empty
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

def box_area(box):
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1 + 1.0)
    h = max(0.0, y2 - y1 + 1.0)
    return w * h

def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1 + 1.0)
    ih = max(0.0, iy2 - iy1 + 1.0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = box_area(a) + box_area(b) - inter
    return inter / ua if ua > 0 else 0.0

def mask_iou(a_mask: np.ndarray, b_mask: np.ndarray):
    a = a_mask.astype(bool)
    b = b_mask.astype(bool)
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return (inter / union) if union > 0 else 0.0

def remove_small_regions(mask: np.ndarray, min_region_area: int = 0):
    """
    Removes very small connected components and fills very small holes.
    Works on boolean/binary mask.
    """
    m = (mask > 0).astype(np.uint8)

    # Remove small foreground components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    cleaned = np.zeros_like(m)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_region_area:
            cleaned[labels == i] = 1

    # Fill small holes by inverting and removing small components
    inv = (1 - cleaned).astype(np.uint8)
    num_labels_h, labels_h, stats_h, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    holes = np.zeros_like(inv)
    h_keep = np.ones(num_labels_h, dtype=bool)
    h_keep[0] = True  # background stays
    for i in range(1, num_labels_h):
        if stats_h[i, cv2.CC_STAT_AREA] < min_region_area:
            # this small background region is a hole; fill it
            holes[labels_h == i] = 1
            h_keep[i] = False
    filled = np.clip(cleaned + holes, 0, 1)
    return filled.astype(np.uint8)
