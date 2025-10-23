# agguard/specialists/mask_service/server.py
from __future__ import annotations
import os, logging
from concurrent import futures
from typing import List

import cv2
import numpy as np
import grpc

from agguard.proto import mask_classifier_pb2 as pb
from agguard.proto import mask_classifier_pb2_grpc as pbrpc

from agguard.specialists.mask_classifier import FaceMaskClassifier, MaskPrediction

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

def _jpeg_to_rgb(j: bytes) -> np.ndarray:
    arr = np.frombuffer(j, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode JPEG")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def _resize_square(rgb: np.ndarray, size: int) -> np.ndarray:
    # match local FaceMaskClassifier._resize_square behavior
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)

# tolerate either service name in generated stubs
ServicerBase = getattr(pbrpc, "ClassifierServiceServicer",
               getattr(pbrpc, "ClassifierServicer", None))
if ServicerBase is None:
    raise ImportError("No Classifier{Service}Servicer in mask_classifier_pb2_grpc.py")

add_servicer = getattr(pbrpc, "add_ClassifierServiceServicer_to_server",
               getattr(pbrpc, "add_ClassifierServicer_to_server", None))
if add_servicer is None:
    raise ImportError("No add_Classifier{Service}Servicer_to_server in mask_classifier_pb2_grpc.py")

class ClassifierService(ServicerBase):
    def __init__(self, model: FaceMaskClassifier):
        self.model = model

    def Classify(self, request: pb.ClassifyRequest, context) -> pb.ClassifyResponse:
        crops_rgb: List[np.ndarray] = []
        boxes = []
        for c in request.crops:
            try:
                rgb = _jpeg_to_rgb(c.jpeg)
                # 🔧 CRITICAL: resize every crop to model.imgsz (e.g., 224x224)
                rgb = _resize_square(rgb, self.model.imgsz)
                crops_rgb.append(rgb)
                boxes.append((c.x1, c.y1, c.x2, c.y2))
            except Exception:
                # skip bad crop
                pass

        preds: List[MaskPrediction] = []
        if crops_rgb:
            if self.model.backend == "torch":
                # Ultralytics will still accept variable sizes, but we resized for parity
                preds = self.model._infer_torch(crops_rgb, boxes)
            else:
                # ONNX path expects inputs already at imgsz — now satisfied
                preds = self.model._infer_onnx(crops_rgb, boxes)

        out = pb.ClassifyResponse()
        for p in preds:
            out.preds.add(
                x1=p.box[0], y1=p.box[1], x2=p.box[2], y2=p.box[3],
                label=p.label.lower(), confidence=float(p.confidence)
            )
        return out

def serve():
    backend = os.environ.get("BACKEND", "onnx").lower()
    model_path = os.environ.get("MODEL_PATH", "/app/weights/mask_yolov8.onnx")
    imgsz = int(os.environ.get("IMGSZ", "224"))
    device = os.environ.get("DEVICE", "cpu")
    classes = os.environ.get("CLASSES")  # e.g., "no_mask,mask"
    class_names = [s.strip() for s in classes.split(",")] if classes else None

    model = FaceMaskClassifier(
        model_path=model_path, backend=backend, imgsz=imgsz, device=device, class_names=class_names
    )

    port = int(os.environ.get("PORT", "50061"))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    add_servicer(ClassifierService(model), server)
    server.add_insecure_port(f"[::]:{port}")
    log.info("MaskClassifier gRPC server listening on :%d", port)
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
