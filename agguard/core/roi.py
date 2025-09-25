from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple
import numpy as np
import cv2

Point = Tuple[float, float]

def _ensure_np(arr: Sequence[Point]) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("Expected Nx2 points")
    return a

@dataclass
class Roi:
    """Polygonal ROI stored in normalized coords [0,1] and sized in pixels."""
    poly_norm: np.ndarray
    size: Tuple[int, int]  # (w, h)

    @staticmethod
    def from_normalized(poly_norm: Sequence[Point], frame_size: Tuple[int, int]) -> "Roi":
        pn = _ensure_np(poly_norm)
        if (pn < 0).any() or (pn > 1).any():
            raise ValueError("ROI must be normalized to [0,1]")
        return Roi(pn, frame_size)

    @staticmethod
    def from_pixels(poly_px: Sequence[Point], frame_size: Tuple[int, int]) -> "Roi":
        w, h = frame_size
        pp = _ensure_np(poly_px)
        pn = np.stack([pp[:, 0] / w, pp[:, 1] / h], axis=1)
        return Roi(pn, frame_size)

    @property
    def poly_px(self) -> np.ndarray:
        w, h = self.size
        pn = self.poly_norm
        pp = np.stack([pn[:, 0] * w, pn[:, 1] * h], axis=1).astype(np.float32)
        return pp

    def as_cv2(self) -> np.ndarray:
        return self.poly_px.reshape((-1, 1, 2)).astype(np.int32)

    def mask(self) -> np.ndarray:
        w, h = self.size
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [self.as_cv2()], 255)
        return m
