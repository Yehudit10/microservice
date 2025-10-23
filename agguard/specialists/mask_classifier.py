# agguard/specialists/mask_classifier.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import os
import logging
import numpy as np
import cv2

log = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]


@dataclass(frozen=True)
class MaskPrediction:
    box: Box
    label: str           # lowercased class name from the model
    confidence: float    # probability of that class
    raw: Dict[str, Any]  # backend-specific details (e.g., full probs)


def _ensure_rgb(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resize_square(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    s = np.sum(e, axis=axis, keepdims=True)
    return e / np.maximum(s, 1e-12)


class FaceMaskClassifier:
    """
    Per-person classifier using a YOLOv8-cls model.

    Backends (must be specified explicitly):
      - backend="torch": uses Ultralytics YOLO (expects *.pt)
      - backend="onnx":  uses onnxruntime (expects *.onnx)

    REQUIRED kwargs:
      - model_path: str
      - backend: "torch" | "onnx"

    OPTIONAL kwargs:
      - imgsz: int (default 224)
      - device: str (default "cpu")
      - class_names: List[str] (REQUIRED for ONNX unless your app supplies a sidecar)
                       For torch, names are read from the YOLO model.

    API:
      classify(frame_bgr, boxes) -> List[MaskPrediction]
    """

    def __init__(
        self,
        model_path: str,
        backend: str,
        imgsz: int = 224,
        device: str = "cpu",
        class_names: Optional[List[str]] = None,
    ):
        self.model_path = model_path
        self.backend = backend.strip().lower()
        self.imgsz = int(imgsz)
        self.device = device
        self._torch_model = None
        self._onnx_session = None
        self._onnx_input_name = None
        self._class_names = class_names  # For ONNX, you must pass this (or ship a sidecar & adjust code)

        if self.backend == "torch":
            self._init_torch()
        elif self.backend == "onnx":
            self._init_onnx()
        else:
            raise ValueError("backend must be 'torch' or 'onnx' (no auto).")

        if not self._class_names or len(self._class_names) < 2:
            raise ValueError(
                "class_names must be resolvable. "
                "For torch, they come from the Ultralytics model; "
                "for onnx, pass class_names explicitly."
            )

        # Normalize class names to lowercase for consistent matching in Aggregator
        self._class_names = [str(n).lower() for n in self._class_names]
        log.info("FaceMaskClassifier initialized (backend=%s, classes=%s)", self.backend, self._class_names)

    # ----------------- backends -----------------

    def _init_torch(self):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError("Ultralytics not available. Install with: pip install ultralytics") from e

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self._torch_model = YOLO(self.model_path)
        try:
            names = getattr(self._torch_model, "names", None)
            if isinstance(names, dict) and len(names) > 0:
                self._class_names = [names[i] for i in range(len(names))]
        except Exception:
            pass

        # Try to place on device (Ultralytics handles fallback)
        try:
            self._torch_model.to(self.device)
        except Exception:
            pass

    def _init_onnx(self):
        try:
            import onnxruntime as ort
        except Exception as e:
            raise RuntimeError("onnxruntime not available. Install with: pip install onnxruntime") from e

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self._onnx_session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        inp = self._onnx_session.get_inputs()[0]
        self._onnx_input_name = inp.name

        # For ONNX we require class_names from kwargs (keeps strict)
        if not self._class_names:
            raise ValueError("ONNX backend requires 'class_names' list (e.g., ['no_mask','mask']).")

        # Detect if the exported ONNX has a static batch size of 1
        shape = inp.shape  # e.g. [1, 3, 224, 224] or [None, 3, 224, 224]
        self._onnx_static_batch1 = False
        if isinstance(shape, (list, tuple)) and len(shape) >= 1:
            bdim = shape[0]
            if isinstance(bdim, int) and bdim == 1:
                self._onnx_static_batch1 = True

    # ----------------- public API -----------------

    def classify(self, frame_bgr: np.ndarray, boxes: List[Box]) -> List[MaskPrediction]:
        if not boxes:
            return []

        H, W = frame_bgr.shape[:2]
        crops_rgb: List[np.ndarray] = []
        kept: List[Box] = []

        for (x1, y1, x2, y2) in boxes:
            X1 = max(0, min(W - 1, int(x1)))
            Y1 = max(0, min(H - 1, int(y1)))
            X2 = max(0, min(W - 1, int(x2)))
            Y2 = max(0, min(H - 1, int(y2)))
            if X2 <= X1 or Y2 <= Y1:
                continue
            crop = frame_bgr[Y1:Y2, X1:X2]
            rgb = _ensure_rgb(crop)
            rgb = _resize_square(rgb, self.imgsz)
            crops_rgb.append(rgb)
            kept.append((X1, Y1, X2, Y2))

        if not crops_rgb:
            return []

        if self.backend == "torch":
            return self._infer_torch(crops_rgb, kept)
        else:
            return self._infer_onnx(crops_rgb, kept)

    # ----------------- inference helpers -----------------

    def _infer_torch(self, crops_rgb: List[np.ndarray], boxes: List[Box]) -> List[MaskPrediction]:
        results = self._torch_model.predict(
            source=crops_rgb,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device if hasattr(self._torch_model, "overrides") else None
        )
        names = self._class_names
        preds: List[MaskPrediction] = []

        for b, r in zip(boxes, results):
            if getattr(r, "probs", None) is None:
                # Highly unlikely for YOLO-cls; fallback to uniform
                probs = np.ones((len(names),), dtype=np.float32) / max(len(names), 1)
                top1 = int(np.argmax(probs))
                conf = float(probs[top1])
            else:
                probs = np.asarray(r.probs.data, dtype=np.float32)  # (C,)
                top1 = int(getattr(r.probs, "top1", int(np.argmax(probs))))
                conf = float(getattr(r.probs, "top1conf", probs[top1]))

            label = names[top1] if 0 <= top1 < len(names) else str(top1)
            preds.append(MaskPrediction(box=b, label=label.lower(), confidence=conf, raw={
                "probs": probs.tolist(),
                "top1": top1
            }))
        return preds

    def _infer_onnx(self, crops_rgb: List[np.ndarray], boxes: List[Box]) -> List[MaskPrediction]:
        import numpy as np
        import onnxruntime as ort

        def _to_nchw(arr: np.ndarray) -> np.ndarray:
            # (H, W, 3) RGB -> (1, 3, H, W) float32 in [0,1]
            x = (arr.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
            return x

        names = self._class_names
        preds: List[MaskPrediction] = []

        if getattr(self, "_onnx_static_batch1", False):
            # Run one-by-one (model expects batch dimension == 1)
            for rgb, b in zip(crops_rgb, boxes):
                X = _to_nchw(rgb)
                outputs = self._onnx_session.run(None, {self._onnx_input_name: X})
                logits = outputs[0]
                if logits.ndim == 1:
                    logits = logits[None, ...]  # (1, C)
                p = _softmax(logits, axis=-1)[0]
                top1 = int(np.argmax(p))
                conf = float(p[top1])
                label = names[top1] if 0 <= top1 < len(names) else str(top1)
                preds.append(MaskPrediction(box=b, label=label.lower(), confidence=conf, raw={
                    "probs": p.tolist(),
                    "top1": top1
                }))
            return preds

        # Vectorized path (dynamic batch models)
        X = np.stack([
            np.transpose((c.astype(np.float32) / 255.0), (2, 0, 1))
            for c in crops_rgb
        ], axis=0)  # (N, 3, H, W)

        outputs = self._onnx_session.run(None, {self._onnx_input_name: X})
        logits = outputs[0]
        if logits.ndim == 1:
            logits = logits[None, ...]
        probs = _softmax(logits, axis=-1)

        for i, b in enumerate(boxes):
            p = probs[i]
            top1 = int(np.argmax(p))
            conf = float(p[top1])
            label = names[top1] if 0 <= top1 < len(names) else str(top1)
            preds.append(MaskPrediction(box=b, label=label.lower(), confidence=conf, raw={
                "probs": p.tolist(),
                "top1": top1
            }))
        return preds

