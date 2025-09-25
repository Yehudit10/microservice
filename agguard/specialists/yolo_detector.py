from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
import logging
from ultralytics import YOLO
from agguard.core.types import Detection


log = logging.getLogger(__name__)

# COCO ids: person(0), car(2), truck(8 older/7 new), animals, etc. Adjust as needed.
DEFAULT_ALLOWED_CLASS_IDS = {0, 2, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}

class YoloDetector:
    """YOLO wrapper that runs on a cropped ROI for speed; returns List[Detection]."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self.model = YOLO(cfg.get("weights", "yolov8n.pt"))
        self.conf = float(cfg.get("conf", 0.25))
        self.imgsz = int(cfg.get("imgsz", 320))
        self.device = cfg.get("device", None)
        self.roi_pad = int(cfg.get("roi_pad", 16))
        self.allowed_ids = cfg.get("allowed_classes", None) or list(DEFAULT_ALLOWED_CLASS_IDS)
        self.names = getattr(self.model, "names", {})

    def _centroid_inside(self, box: Tuple[int, int, int, int], roi_mask: np.ndarray) -> bool:
        x1, y1, x2, y2 = box
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        h, w = roi_mask.shape[:2]
        return 0 <= cx < w and 0 <= cy < h and roi_mask[cy, cx] > 0

    def _crop_from_mask(self, frame: np.ndarray, roi_mask: np.ndarray):
        ys, xs = np.where(roi_mask > 0)
        if len(xs) == 0:
            return frame, (0, 0)
        x1, y1 = max(xs.min() - self.roi_pad, 0), max(ys.min() - self.roi_pad, 0)
        x2, y2 = min(xs.max() + self.roi_pad, frame.shape[1] - 1), min(ys.max() + self.roi_pad, frame.shape[0] - 1)
        return frame[y1 : y2 + 1, x1 : x2 + 1].copy(), (x1, y1)

    def detect(self, frame_bgr: np.ndarray, roi_mask: np.ndarray) -> List[Detection]:
        crop, (ox, oy) = self._crop_from_mask(frame_bgr, roi_mask)
        results = self.model.predict(
            source=crop,
            conf=self.conf,
            classes=self.allowed_ids,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        out: List[Detection] = []
        if not results:
            return out
        res = results[0]
        if res.boxes is None:
            return out
        for b, conf, cid in zip(
            res.boxes.xyxy.cpu().numpy(),
            res.boxes.conf.cpu().numpy(),
            res.boxes.cls.cpu().numpy().astype(int),
        ):
            x1, y1, x2, y2 = map(int, b)
            x1 += ox; x2 += ox; y1 += oy; y2 += oy
            box = (x1, y1, x2, y2)
            if not self._centroid_inside(box, roi_mask):
                continue
            name = self.names.get(int(cid), str(cid))
            out.append(Detection(name, float(conf), box))
        log.debug("Detector: %d detections", len(out))
        return out
