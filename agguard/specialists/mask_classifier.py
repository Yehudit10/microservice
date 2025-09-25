from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable
import cv2
import numpy as np
from ultralytics import YOLO

Box = Tuple[int, int, int, int]

@dataclass(frozen=True)
class MaskPrediction:
    box: Box
    label: str
    confidence: float
    raw: dict

class FaceMaskClassifier:
    """
    Runs a fine-tuned Ultralytics YOLO model on person crops;
    returns one MaskPrediction per input box (pick top detection).
    """

    def __init__(
        self,
        model_path: str,
        max_crop_size: int = 384,
        imgsz: int = 320,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        device: Optional[str | int] = None,
    ):
        if not model_path:
            raise ValueError("model_path is required")
        self.model = YOLO(model_path)
        self.max_crop_size = max_crop_size
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.names = getattr(self.model, "names", {})

    def classify(self, frame_bgr: np.ndarray, face_boxes_xyxy: Iterable[Box]) -> List[MaskPrediction]:
        H, W = frame_bgr.shape[:2]
        boxes, crops = [], []

        for (x1, y1, x2, y2) in face_boxes_xyxy:
            x1 = max(0, min(int(x1), W - 1)); x2 = max(0, min(int(x2), W - 1))
            y1 = max(0, min(int(y1), H - 1)); y2 = max(0, min(int(y2), H - 1))
            if x2 <= x1 or y2 <= y1:
                boxes.append((x1, y1, x2, y2)); crops.append(None); continue
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                boxes.append((x1, y1, x2, y2)); crops.append(None); continue
            crop = self._resize_long_edge(crop, self.max_crop_size)
            boxes.append((x1, y1, x2, y2))
            crops.append(crop)

        valid = [i for i, c in enumerate(crops) if c is not None]
        images = [crops[i] for i in valid]
        results = []
        if images:
            results = self.model.predict(
                source=images,
                imgsz=self.imgsz,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                device=self.device,
                verbose=False,
            )

        idx_to_result = {valid[i]: results[i] for i in range(len(valid))}
        out: List[MaskPrediction] = []

        for i, face_box in enumerate(boxes):
            if i not in idx_to_result:
                out.append(MaskPrediction(face_box, "invalid_box", 0.0, {}))
                continue
            r = idx_to_result[i]
            if r.boxes is None or len(r.boxes) == 0:
                out.append(MaskPrediction(face_box, "unknown", 0.0, {"detections": 0}))
                continue
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            k = int(confs.argmax())
            top_conf = float(confs[k])
            top_cls = int(clss[k])
            label = self.names.get(top_cls, str(top_cls))
            out.append(
                MaskPrediction(
                    face_box,
                    label,
                    top_conf,
                    {"class_id": top_cls, "class_name": label, "confidence": top_conf, "num_detections": int(len(confs))},
                )
            )
        return out

    @staticmethod
    def _resize_long_edge(img: np.ndarray, max_long: int) -> np.ndarray:
        if not max_long or max_long <= 0:
            return img
        h, w = img.shape[:2]
        m = max(h, w)
        if m <= max_long:
            return img
        s = max_long / float(m)
        return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
