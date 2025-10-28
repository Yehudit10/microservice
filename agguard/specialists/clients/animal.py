from __future__ import annotations
import logging
from typing import List, Tuple, Optional
import cv2, numpy as np, grpc

from agguard.proto import mask_classifier_pb2 as pb
from agguard.proto import mask_classifier_pb2_grpc as pbrpc

log = logging.getLogger(__name__)
Box = Tuple[int, int, int, int]

StubClass = getattr(pbrpc, "ClassifierServiceStub", getattr(pbrpc, "ClassifierStub", None))
if StubClass is None:
    raise ImportError("mask_classifier_pb2_grpc missing ClassifierServiceStub/ClassifierStub")




class GrpcAnimalClassifierClient:
    def __init__(self, address: str, model_name: str = "yolo-cls",
                 timeout_sec: float = 12.0, jpeg_quality: int = 85):
        self.address = address
        self.model_name = model_name
        self.timeout = float(timeout_sec)
        self.jpeg_quality = int(jpeg_quality)
        self._chan = grpc.insecure_channel(self.address, options=[
            ("grpc.max_send_message_length", 32 * 1024 * 1024),
            ("grpc.max_receive_message_length", 32 * 1024 * 1024),
        ])
        self._stub = StubClass(self._chan)
        log.info("GrpcAnimalClassifierClient -> %s (model=%s)", self.address, self.model_name)

    @staticmethod
    def _safe_crop(frame_bgr: np.ndarray, box: Box) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_bgr[y1:y2, x1:x2]

    def classify(self, frame_bgr: np.ndarray, boxes: List[Box]):
        crops: List[pb.Crop] = []
        for b in boxes:
            crop = self._safe_crop(frame_bgr, b)
            if crop is None:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                continue
            crops.append(pb.Crop(jpeg=buf.tobytes(), x1=b[0], y1=b[1], x2=b[2], y2=b[3]))
        if not crops:
            return []

        req = pb.ClassifyRequest(model_name=self.model_name, crops=crops)
        try:
            resp = self._stub.Classify(req, timeout=self.timeout)
        except Exception as e:
            log.warning("gRPC classify failed (%s): %s", self.address, e)
            return []

        out = []
        for p in resp.preds:
            out.append({
                "box": (p.x1, p.y1, p.x2, p.y2),
                "label": p.label,
                "confidence": p.confidence
            })
        return out
