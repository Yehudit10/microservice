from __future__ import annotations
import time, grpc, cv2
import numpy as np
from agguard.proto import mega_detector_pb2 as pb2
from agguard.proto import mega_detector_pb2_grpc as pb2_grpc

class Detection:
    def __init__(self, cls: str, conf: float, bbox: tuple[int, int, int, int]):
        self.cls, self.conf, self.bbox = cls, conf, bbox

class MegaDetectorClient:
    def __init__(self, cfg: dict):
        self.host = cfg.get("host", "mega-detector:50063")
        self.timeout = float(cfg.get("timeout", 5.0))
        self.channel = grpc.insecure_channel(self.host)
        self.stub = pb2_grpc.MegaDetectorStub(self.channel)
        print(f"[MegaDetectorClient] Connected to {self.host}")

    def detect(self, frame_bgr: np.ndarray, roi_mask: np.ndarray | None = None):
        if roi_mask is not None:
            frame_bgr = cv2.bitwise_and(frame_bgr, frame_bgr, mask=roi_mask.astype(np.uint8))
        ok, buf = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            return []

        req = pb2.ImageRequest(image_bytes=buf.tobytes())
        try:
            t0 = time.time()
            resp = self.stub.Detect(req, timeout=self.timeout)
            dt = time.time() - t0
        except grpc.RpcError as e:
            print(f"[MegaDetectorClient] gRPC failed: {e.code().name} - {e.details()}")
            return []

        dets = [
            Detection(d.cls, d.conf, (int(d.x1), int(d.y1), int(d.x2), int(d.y2)))
            for d in resp.detections
        ]
        print(f"[MegaDetectorClient] {len(dets)} detections in {dt:.2f}s")
        return dets
