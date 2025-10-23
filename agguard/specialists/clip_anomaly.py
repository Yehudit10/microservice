# agguard/specialists/clip_anomaly.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Dict, Any
import cv2
import numpy as np
from PIL import Image
import torch
import open_clip

Box = Tuple[int, int, int, int]

@dataclass(frozen=True)
class AnomalyPrediction:
    box: Box
    label: str            # lower-cased; matches Rule.attr_value
    confidence: float
    raw: dict             # extra info (topk, per-class probs, etc.)

class ClipAnomalyClassifier:
    """
    Zero-shot CLIP classifier over person/scene crops.
    - Prompt-ensembles text labels (templates) to form class embeddings.
    - Encodes each crop with CLIP image tower and scores against classes.
    - Returns one prediction per input box (top-1 by default).
    """

    DEFAULT_LABELS = [
        "shooting",
        "climbing a fence",
        "stealing or robbery",
        "explosion",
        "normal activity",
    ]

    DEFAULT_TEMPLATES = [
        "a CCTV frame of {}",
        "a surveillance video of {}",
        "a low-resolution photo of {}",
        "a person {}",
        "an action of {}",
        "a person is {}",
    ]

    def __init__(
        self,
        labels: Optional[List[str]] = None,
        templates: Optional[List[str]] = None,
        model_name: str = "ViT-B-32",                 # light & fast
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str | int] = None,
        input_size: int = 224,
        pad_ratio: float = 0.18,                      # extra context around the box
        batch_size: int = 16,
        aggregate: str = "meanmax",                   # mean | max | meanmax over template variants
        class_thresholds: Optional[Dict[str, float]] = None,  # per-class thresholding (optional)
        return_top1: bool = True,                     # if False: returns all labels >= threshold
        use_full_frame_if_no_boxes: bool = True,      # helpful for "explosion"
    ):
        self.labels = [s.strip().lower() for s in (labels or self.DEFAULT_LABELS)]
        self.templates = templates or self.DEFAULT_TEMPLATES
        self.model_name = model_name
        self.pretrained = pretrained
        self.input_size = int(input_size)
        self.pad_ratio = float(pad_ratio)
        self.batch_size = int(batch_size)
        self.aggregate = aggregate
        self.return_top1 = bool(return_top1)
        self.use_full_frame_if_no_boxes = bool(use_full_frame_if_no_boxes)
        self.class_thresholds = {k.lower(): float(v) for k, v in (class_thresholds or {}).items()}

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load CLIP (OpenCLIP)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        # Build text embeddings once
        self._class_emb, self._id2label = self._build_text_embeddings(self.labels, self.templates)

    # ---------------- internal helpers ----------------

    def _build_text_embeddings(self, labels: List[str], templates: List[str]):
        texts: List[str] = []
        offsets: List[int] = []
        for lab in labels:
            variants = [t.format(lab) for t in templates]
            offsets.append(len(texts))
            texts.extend(variants)

        tok = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            z = self.model.encode_text(tok)
            z = z / z.norm(dim=-1, keepdim=True)

        class_vecs = []
        for i in range(len(labels)):
            s = offsets[i]
            e = offsets[i + 1] if i + 1 < len(labels) else len(texts)
            v = z[s:e]
            if self.aggregate == "mean":
                v = v.mean(dim=0)
            elif self.aggregate == "max":
                v = v.max(dim=0).values
            else:  # meanmax
                v = 0.5 * (v.mean(dim=0) + v.max(dim=0).values)
            v = v / v.norm()
            class_vecs.append(v)
        emb = torch.stack(class_vecs, dim=0)  # [C, D]
        return emb, {i: labels[i] for i in range(len(labels))}

    @staticmethod
    def _pad_clip(box: Box, W: int, H: int, r: float) -> Box:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        X1 = max(0, int(round(x1 - r * w)))
        Y1 = max(0, int(round(y1 - r * h)))
        X2 = min(W - 1, int(round(x2 + r * w)))
        Y2 = min(H - 1, int(round(y2 + r * h)))
        if X2 <= X1: X2 = min(W - 1, X1 + 1)
        if Y2 <= Y1: Y2 = min(H - 1, Y1 + 1)
        return (X1, Y1, X2, Y2)

    def _encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        if not images:
            # Return empty tensor with correct feature dim
            with torch.no_grad():
                feat_dim = self.model.visual.output_dim
            return torch.empty(0, feat_dim, device=self.device)

        ims = [self.preprocess(im).unsqueeze(0) for im in images]
        ims = torch.cat(ims, dim=0).to(self.device)
        outs = []
        with torch.no_grad():
            for i in range(0, len(ims), self.batch_size):
                chunk = ims[i:i + self.batch_size]
                z = self.model.encode_image(chunk)
                z = z / z.norm(dim=-1, keepdim=True)
                outs.append(z)
        return torch.cat(outs, dim=0)

    # ---------------- public API (used by ClassDispatch) ----------------

    def classify(self, frame_bgr: np.ndarray, boxes_xyxy: Iterable[Box]) -> List[AnomalyPrediction]:
        H, W = frame_bgr.shape[:2]
        boxes = list(boxes_xyxy) if boxes_xyxy is not None else []

        # If no boxes and allowed, classify the whole frame (scene anomaly)
        if not boxes and self.use_full_frame_if_no_boxes:
            boxes = [(0, 0, W - 1, H - 1)]

        crops: List[Image.Image] = []
        valid_idx: List[int] = []

        # Build crops (with padding)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1 = max(0, min(int(x1), W - 1)); x2 = max(0, min(int(x2), W - 1))
            y1 = max(0, min(int(y1), H - 1)); y2 = max(0, min(int(y2), H - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            X1, Y1, X2, Y2 = self._pad_clip((x1, y1, x2, y2), W, H, self.pad_ratio)
            crop = frame_bgr[Y1:Y2, X1:X2]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(crop_rgb).resize((self.input_size, self.input_size), Image.BICUBIC)
            crops.append(im)
            valid_idx.append(i)

        if not crops:
            return [AnomalyPrediction((0, 0, 1, 1), "unknown", 0.0, {"reason": "no_valid_crops"})]

        img_emb = self._encode_images(crops)              # [N, D]
        with torch.no_grad():
            logits = 100.0 * (img_emb @ self._class_emb.T)  # [N, C]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        preds: List[AnomalyPrediction] = []
        for j, i_box in enumerate(valid_idx):
            box = tuple(map(int, boxes[i_box]))
            p = probs[j]
            top = int(p.argmax())
            top_lab = self._id2label[top]
            top_conf = float(p[top])

            # Apply optional per-class thresholds (suppress to "normal activity" if below)
            if self.class_thresholds:
                th = float(self.class_thresholds.get(top_lab, 0.0))
                if top_conf < th and "normal activity" in self.labels:
                    norm_idx = self.labels.index("normal activity")
                    top_lab = "normal activity"
                    top_conf = float(p[norm_idx])

            if self.return_top1:
                preds.append(
                    AnomalyPrediction(
                        box=box,
                        label=top_lab.lower(),
                        confidence=top_conf,
                        raw={
                            "top_idx": int(top),
                            "top_conf": top_conf,
                            "top_label": top_lab,
                            "probs": {self._id2label[k]: float(v) for k, v in enumerate(p)},
                        },
                    )
                )
            else:
                # multi-label style: emit all labels above per-class thresholds
                for k, lab in self._id2label.items():
                    conf = float(p[k])
                    th = float(self.class_thresholds.get(lab, 0.0)) if self.class_thresholds else 0.0
                    if conf >= th:
                        preds.append(
                            AnomalyPrediction(
                                box=box,
                                label=lab.lower(),
                                confidence=conf,
                                raw={"idx": int(k)}
                            )
                        )

        # If some input boxes were invalid, fill placeholders to keep one-per-input-box invariant (optional)
        missing = len(boxes) - len(valid_idx)
        for _ in range(max(0, missing)):
            preds.append(AnomalyPrediction((0, 0, 1, 1), "unknown", 0.0, {"reason": "invalid_box"}))

        return preds
