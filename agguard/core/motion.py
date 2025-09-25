from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2
import logging
from .roi import Roi

log = logging.getLogger(__name__)

@dataclass
class ChangeReading:
    score: float
    area_px: int
    total_px: int
    fgmask: np.ndarray
    bboxes: List[Tuple[int, int, int, int]]

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1); ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aA = (ax2 - ax1 + 1) * (ay2 - ay1 + 1); bA = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    return inter / float(aA + bA - inter)

def _merge_overlapping(boxes, iou_thresh=0.35, max_iters=5):
    boxes = boxes[:]
    for _ in range(max_iters):
        merged, used = [], [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            cur, changed = boxes[i], True
            while changed:
                changed = False
                for j in range(i + 1, len(boxes)):
                    if used[j]:
                        continue
                    if _iou(cur, boxes[j]) >= iou_thresh:
                        x1 = min(cur[0], boxes[j][0]); y1 = min(cur[1], boxes[j][1])
                        x2 = max(cur[2], boxes[j][2]); y2 = max(cur[3], boxes[j][3])
                        cur = (x1, y1, x2, y2)
                        used[j] = True
                        changed = True
            used[i] = True
            merged.append(cur)
        if len(merged) == len(boxes):
            return merged
        boxes = merged
    return boxes

class MotionGate:
    """
    Same API/output intent as your original: MOG2 on BGR + ROI + morphology + CC + merge,
    but runs on a downscaled frame for speed. Kernels and min-area are scaled so the
    *effective* behavior matches full-res.
    """

    def __init__(
        self,
        roi: Roi,
        history: int = 300,
        var_threshold: float = 16.0,
        shadow: bool = True,        # keep your default
        min_blob_area: int = 80,
        morph_open: int = 3,
        smooth_alpha: float = 0.2,
        dilate_px: int = 2,
        morph_close: int = 5,
        merge_iou: float = 0.35,
        target_max_dim: int = 480,  # downscale max side; set to 0/None to disable
    ):
        self.roi = roi
        self.min_blob_area = int(min_blob_area)
        self.morph_open = int(morph_open)
        self.smooth_alpha = float(smooth_alpha)
        self.dilate_px = int(dilate_px)
        self.morph_close = int(morph_close)
        self.merge_iou = float(merge_iou)
        self.target_max_dim = int(target_max_dim) if target_max_dim else 0

        self._ema: Optional[float] = None
        self._roi_mask = roi.mask()              # full-res mask
        self._roi_mask_small: Optional[np.ndarray] = None
        self._bg_size: Optional[Tuple[int, int]] = None

        self._bg_shadow = bool(shadow)
        self._history = int(history)
        self._var_threshold = float(var_threshold)
        self.bg = None  # created lazily when small size is known

    def _ensure_background(self, sh: int, sw: int):
        if self._bg_size != (sh, sw):
            self.bg = cv2.createBackgroundSubtractorMOG2(
                history=self._history,
                varThreshold=self._var_threshold,
                detectShadows=self._bg_shadow,
            )
            self._bg_size = (sh, sw)

    def _ensure_small_roi_mask(self, h: int, w: int, sh: int, sw: int):
        if (w, h) != self.roi.size:
            self.roi.size = (w, h)
            self._roi_mask = self.roi.mask()
            self._roi_mask_small = None
        if self._roi_mask_small is None or self._roi_mask_small.shape != (sh, sw):
            self._roi_mask_small = cv2.resize(
                self._roi_mask, (sw, sh), interpolation=cv2.INTER_NEAREST
            )

    def update(self, frame_bgr: np.ndarray) -> ChangeReading:
        h, w = frame_bgr.shape[:2]

        # scale factor (downscale for speed; 1.0 means no downscale)
        if self.target_max_dim and max(h, w) > self.target_max_dim:
            scale = max(h, w) / float(self.target_max_dim)
            sw, sh = int(round(w / scale)), int(round(h / scale))
        else:
            scale = 1.0
            sw, sh = w, h

        # prepare small BGR (keep BGR like your original)
        small = frame_bgr if scale == 1.0 else cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)

        self._ensure_background(sh, sw)
        self._ensure_small_roi_mask(h, w, sh, sw)

        # ---- MOG2 on BGR (like original) ----
        fg = self.bg.apply(small)  # if detectShadows=True, returns 0/127/255
        if self._bg_shadow:
            # keep your original binarying step
            _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        # ROI on small
        fg = cv2.bitwise_and(fg, self._roi_mask_small)

        # ---- scale morphology kernels to preserve behavior ----
        def k_odd(x: int) -> int:
            return x if x % 2 == 1 else max(1, x - 1)

        # kernels shrink by ~1/scale on the small image
        dilate_small = max(0, int(round(self.dilate_px / scale)))
        close_small  = max(0, int(round(self.morph_close / scale)))
        open_small   = max(0, int(round(self.morph_open / scale)))

        if dilate_small > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_odd(2*dilate_small + 1), k_odd(2*dilate_small + 1)))
            fg = cv2.dilate(fg, k, iterations=1)
        if close_small > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_odd(close_small), k_odd(close_small)))
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=1)
        if open_small > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_odd(open_small), k_odd(open_small)))
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)

        # CC on small; area threshold scaled by 1/scale^2
        min_area_small = max(1, int(round(self.min_blob_area / (scale * scale))))
        num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)

        boxes_small: List[Tuple[int, int, int, int]] = []
        clean_small = np.zeros_like(fg)
        for i in range(1, num):
            x, y, w0, h0, area = stats[i]
            if area < min_area_small:
                continue
            x1, y1, x2, y2 = x, y, x + w0 - 1, y + h0 - 1
            boxes_small.append((x1, y1, x2, y2))
            clean_small[y : y + h0, x : x + w0] = 255

        if len(boxes_small) > 1 and self.merge_iou > 0:
            boxes_small = _merge_overlapping(boxes_small, iou_thresh=self.merge_iou)

        # map back to full-res
        if scale != 1.0:
            bboxes = [
                (int(round(x1 * scale)), int(round(y1 * scale)),
                 int(round(x2 * scale)), int(round(y2 * scale)))
                for (x1, y1, x2, y2) in boxes_small
            ]
            clean = cv2.resize(clean_small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            bboxes = boxes_small
            clean = clean_small

        moving = int((clean == 255).sum())
        total = int((self._roi_mask == 255).sum())
        raw = float(moving) / float(max(total, 1))
        if self.smooth_alpha > 0:
            self._ema = raw if self._ema is None else (1 - self.smooth_alpha) * self._ema + self.smooth_alpha * raw
            score = float(self._ema)
        else:
            score = raw

        log.debug("MotionGate: score=%.3f, blobs=%d, area=%d/%d", score, len(bboxes), moving, total)
        return ChangeReading(score=score, area_px=moving, total_px=total, fgmask=clean, bboxes=bboxes)
