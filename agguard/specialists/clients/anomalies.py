from __future__ import annotations
from typing import List, Tuple, Optional
import cv2
import grpc
from agguard.proto import mask_classifier_pb2 as pb
from agguard.proto import mask_classifier_pb2_grpc as pbrpc

BBox = Tuple[int, int, int, int]

class GrpcClipClassifierClient:
    """
    Client for the CLIP climbing classifier microservice.

    classify(frame_bgr, boxes, subjects=None) -> List[pb.Pred]
    """

    def __init__(self, address: str, timeout_sec: float = 1.5, jpeg_quality: int = 85):
        self.address = address
        self.timeout = float(timeout_sec)
        self.jpeg_quality = int(jpeg_quality)

        self.channel = grpc.insecure_channel(address)
        self.stub = pbrpc.ClassifierServiceStub(self.channel) \
            if hasattr(pbrpc, "ClassifierServiceStub") else pbrpc.ClassifierStub(self.channel)

    def _encode_crop(self, frame_bgr, box: BBox) -> bytes:
        x1, y1, x2, y2 = box
        crop = frame_bgr[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
        if crop.size == 0:
            return b""
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        return bytes(buf) if ok else b""

    def classify(self, frame_bgr, boxes: List[BBox], subjects: Optional[List[str]] = None):
        req = pb.ClassifyRequest()
        subs = list(subjects) if subjects else ["object"] * len(boxes)

        if len(subs) < len(boxes):
            subs += ["object"] * (len(boxes) - len(subs))
        elif len(subs) > len(boxes):
            subs = subs[:len(boxes)]

        for i, b in enumerate(boxes):
            jpeg = self._encode_crop(frame_bgr, b)
            if not jpeg:
                continue
            c = req.crops.add(x1=int(b[0]), y1=int(b[1]), x2=int(b[2]), y2=int(b[3]), jpeg=jpeg)
            if hasattr(c, "subject"):
                setattr(c, "subject", subs[i])

        resp = self.stub.Classify(req, timeout=self.timeout)
        if hasattr(resp, "detections"):
            preds = list(resp.detections)
        elif hasattr(resp, "preds"):
            preds = list(resp.preds)
        elif hasattr(resp, "results"):
            preds = list(resp.results)
        else:
            # fallback for unexpected response types
            import logging
            logging.warning(f"[GrpcClipClassifierClient] Unknown response fields: {resp}")
            preds = []

        return preds
        return resp.preds
