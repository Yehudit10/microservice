from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import os
import numpy as np
import cv2
import logging

# If you keep it as-is in your project, imports match your original file:
from agguard.core.types import Detection

log = logging.getLogger(__name__)

# COCO ids: person(0), car(2), truck(7), animals, etc. Adjust as needed.
DEFAULT_ALLOWED_CLASS_IDS = {0, 2, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}


def _cap_long_side(img: np.ndarray, long_max: int) -> tuple[np.ndarray, float]:
    """Resize with aspect ratio if max(h,w) > long_max; return (img, scale)."""
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= long_max:
        return img, 1.0
    s = long_max / float(m)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), s


def _letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Square letterbox to (size,size). Return (canvas, scale, (dx,dy))."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=img.dtype)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, r, (left, top)


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, topk: int) -> List[int]:
    """Lightweight NMS on CPU. boxes: [N,4] in xyxy; returns keep indices."""
    if boxes.size == 0:
        return []
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0 and len(keep) < topk:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        idxs = rest[iou <= iou_thr]
    return keep


class YoloDetector:
    """
    YOLO wrapper (PyTorch or ONNX Runtime) that runs on a cropped ROI.
      - backend: "torch" (default) or "onnx"
      - If ONNX, set cfg["onnx"] to the exported model path.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self.backend = cfg.get("backend", "torch")  # "torch" | "onnx"
        self.weights = cfg.get("weights", "yolov8n.pt")
        self.onnx_path = cfg.get("onnx", self.weights.replace(".pt", ".onnx"))

        # Tunables
        self.conf = float(cfg.get("conf", 0.30))
        self.iou = float(cfg.get("iou", 0.45))
        self.imgsz = int(cfg.get("imgsz", 320))           # consider 256 on CPU
        self.long_cap = int(cfg.get("roi_long_cap", 640)) # cap cropped ROI long side for speed
        self.roi_pad = int(cfg.get("roi_pad", 16))
        self.max_det = int(cfg.get("max_det", 100))
        self.allowed_ids = list(cfg.get("allowed_classes", DEFAULT_ALLOWED_CLASS_IDS))
        self.device = cfg.get("device", "cpu")            # "cpu" (torch), ignored by onnx

        # Threading: keep CPU predictable
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        self.names: Dict[int, str] = {}
        if self.backend == "onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(os.getenv("ORT_THREADS", "1"))
            so.inter_op_num_threads = 1
            self.ort = ort.InferenceSession(self.onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
            # You can pass names in cfg if you want readable labels. Otherwise numeric ids.
            self.names = cfg.get("names", {})
            # Warm-up
            self._run_onnx(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8))
        else:
            from ultralytics import YOLO
            self.model = YOLO(self.weights)
            self.names = getattr(self.model, "names", {})
            # Warm-up
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(
                source=dummy,
                conf=self.conf,
                classes=self.allowed_ids,
                imgsz=self.imgsz,
                device=self.device,
                iou=self.iou,
                max_det=self.max_det,
                verbose=False,
            )

    # ----------------- helpers -----------------

    def _centroid_inside(self, box: Tuple[int, int, int, int], roi_mask: np.ndarray) -> bool:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        h, w = roi_mask.shape[:2]
        return 0 <= cx < w and 0 <= cy < h and roi_mask[cy, cx] > 0

    def _crop_from_mask(self, frame: np.ndarray, roi_mask: np.ndarray):
        ys, xs = np.where(roi_mask > 0)
        if xs.size == 0:
            return frame, (0, 0)
        x1 = max(int(xs.min()) - self.roi_pad, 0)
        y1 = max(int(ys.min()) - self.roi_pad, 0)
        x2 = min(int(xs.max()) + self.roi_pad, frame.shape[1] - 1)
        y2 = min(int(ys.max()) + self.roi_pad, frame.shape[0] - 1)
        return frame[y1 : y2 + 1, x1 : x2 + 1].copy(), (x1, y1)

    # ----------------- ONNX path -----------------

    def _run_onnx(self, img_bgr: np.ndarray):
        """
        Handles Ultralytics YOLOv8 ONNX outputs with either 84 or 85 columns and
        different layouts. Returns (boxes_xyxy, scores, classes, r, dx, dy).
        """
        # Letterbox -> NCHW float32 [0,1]
        lb, r, (dx, dy) = _letterbox(img_bgr, self.imgsz)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(rgb, (2, 0, 1))[None]  # (1,3,H,W)

        # Run ORT
        input_name = self.ort.get_inputs()[0].name
        out = self.ort.run(None, {input_name: inp})[0]

        if log.isEnabledFor(logging.DEBUG):
            try:
                log.debug("ORT out shape=%s", out.shape)
            except Exception:
                pass


        # Normalize to shape (N, C)
        if out.ndim == 3:
            # Either (1, C, N) or (1, N, C)
            if out.shape[0] == 1 and out.shape[1] in (84, 85):
                # (1, C, N) -> (N, C)
                out = out[0].transpose(1, 0)
            elif out.shape[0] == 1 and out.shape[2] in (84, 85):
                # (1, N, C) -> (N, C)
                out = out[0]
            else:
                log.warning("ONNX output unexpected 3D shape %s", out.shape)
                return (np.empty((0, 4), np.float32),
                        np.empty((0,), np.float32),
                        np.empty((0,), np.int32),
                        r, dx, dy)
        elif out.ndim == 2:
            # already (N, C)
            pass
        else:
            log.warning("ONNX output unexpected shape %s", out.shape)
            return (np.empty((0, 4), np.float32),
                    np.empty((0,), np.float32),
                    np.empty((0,), np.int32),
                    r, dx, dy)

        N, C = out.shape
        if C not in (84, 85):
            log.warning("ONNX output columns %d not in {84,85}", C)
            return (np.empty((0, 4), np.float32),
                    np.empty((0,), np.float32),
                    np.empty((0,), np.int32),
                    r, dx, dy)

        out = out.astype(np.float32, copy=False)
        xywh = out[:, :4]

        if C == 85:
            # xywh + obj + 80 cls
            obj = out[:, 4]
            cls_scores = out[:, 5:]
            cls_idx = np.argmax(cls_scores, axis=1).astype(np.int32)
            best_cls = cls_scores[np.arange(N), cls_idx]
            scores = obj * best_cls  # standard obj * class
        else:  # C == 84 (no objectness column)
            # xywh + 80 cls
            cls_scores = out[:, 4:]
            cls_idx = np.argmax(cls_scores, axis=1).astype(np.int32)
            scores = cls_scores[np.arange(N), cls_idx]

        # Early filtering
        keep = scores >= float(self.conf)
        if self.allowed_ids:
            # ensure dtype aligns
            allowed = np.array(self.allowed_ids, dtype=np.int32)
            keep &= np.isin(cls_idx, allowed)

        xywh = xywh[keep]
        scores = scores[keep]
        cls_idx = cls_idx[keep]

        if xywh.shape[0] == 0:
            return (np.empty((0, 4), np.float32),
                    np.empty((0,), np.float32),
                    np.empty((0,), np.int32),
                    r, dx, dy)

        # xywh -> xyxy in letterboxed space
        cx, cy, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        return boxes_xyxy, scores, cls_idx, r, dx, dy


    # ----------------- public API -----------------

    def detect(self, frame_bgr: np.ndarray, roi_mask: np.ndarray) -> List[Detection]:
        crop, (ox, oy) = self._crop_from_mask(frame_bgr, roi_mask)
        # Cap crop long side for CPU speed
        crop, s = _cap_long_side(crop, self.long_cap)        # 🚧 keep scale 's'

        if self.backend == "onnx":
            boxes_lx, scores, cls_idx, r, dx, dy = self._run_onnx(crop)
            if boxes_lx.size == 0:
                log.debug("Detector(onnx): 0 detections")
                return []

            # NMS in letterboxed space
            keep = _nms_xyxy(boxes_lx, scores, self.iou, self.max_det)
            boxes_lx = boxes_lx[keep]
            scores = scores[keep]
            cls_idx = cls_idx[keep]

            # Un-letterbox back to *resized* crop coords
            boxes = boxes_lx.copy().astype(np.float32)
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / r
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / r

            # 🚧 Undo the pre-letterbox downscale so we're back in ORIGINAL crop coords
            if s != 1.0:
                boxes /= float(s)

            # Then offset to full-frame coords
            boxes[:, 0::2] += ox
            boxes[:, 1::2] += oy

            out: List[Detection] = []
            H, W = frame_bgr.shape[:2]
            for b, sc, cid in zip(boxes, scores, cls_idx):
                x1, y1, x2, y2 = map(int, b)
                # (optional) clamp to image bounds
                x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
                y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
                box = (x1, y1, x2, y2)
                if not self._centroid_inside(box, roi_mask):
                    continue
                name = self.names.get(int(cid), str(int(cid)))
                out.append(Detection(name, float(sc), box))
            log.debug("Detector(onnx): %d detections", len(out))
            return out

        # ---- PyTorch (Ultralytics) path ----
        results = self.model.predict(
            source=crop,
            conf=self.conf,
            classes=self.allowed_ids,
            imgsz=self.imgsz,
            device=self.device,
            iou=self.iou,
            max_det=self.max_det,
            verbose=False,
        )
        out: List[Detection] = []
        if not results:
            return out
        res = results[0]
        if res.boxes is None:
            return out

        H, W = frame_bgr.shape[:2]
        for b, conf, cid in zip(
            res.boxes.xyxy.cpu().numpy(),
            res.boxes.conf.cpu().numpy(),
            res.boxes.cls.cpu().numpy().astype(int),
        ):
            b = b.astype(np.float32)

            # 🚧 Ultralytics returns coords in the coords of the 'source' (our *resized* crop)
            # Undo the pre-downscale to get ORIGINAL crop coordinates
            if s != 1.0:
                b /= float(s)

            # Now offset to full-frame
            x1, y1, x2, y2 = b
            x1 += ox; x2 += ox; y1 += oy; y2 += oy

            # (optional) clamp
            x1 = int(max(0, min(W - 1, x1))); x2 = int(max(0, min(W - 1, x2)))
            y1 = int(max(0, min(H - 1, y1))); y2 = int(max(0, min(H - 1, y2)))

            box = (x1, y1, x2, y2)
            if not self._centroid_inside(box, roi_mask):
                continue
            name = self.names.get(int(cid), str(cid))
            out.append(Detection(name, float(conf), box))

        log.debug("Detector(torch): %d detections", len(out))
        return out

