#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animal Classifier microservice — reusing mask-classifier.proto.
Maps model class names (e.g., "american_black_bear", "sloth_bear")
to unified labels (e.g., "bear"). Unrecognized classes → "other".
"""

from __future__ import annotations
import io, os, time, grpc
from concurrent import futures
from PIL import Image
from ultralytics import YOLO
from agguard.proto import mask_classifier_pb2 as pb2
from agguard.proto import mask_classifier_pb2_grpc as pb2_grpc


class AnimalClassifierServicer(pb2_grpc.ClassifierServiceServicer):
    def __init__(self):
        model_path = os.getenv("MODEL_PATH", "/app/weights/yolov8m-cls.pt")
        print(f"[AnimalClassifier] Loading {model_path} ...")
        self.model = YOLO(model_path)

        # ─────────────────────────────────────────────────────────────
        # Mapping: model class → unified label
        # ─────────────────────────────────────────────────────────────
        self.label_map = {
            # Bears
            "american_black_bear": "American_black_bear",
            "sloth_bear": "American_black_bear",
            "brown_bear": "American_black_bear",
            "gibbon":"American_black_bear",
            "siamang":"American_black_bear",
            # Foxes
            "red_fox": "fox",
            "grey_fox": "fox",
            # Others
            "wild_boar": "boar",
            "wolf": "wolf",
            "deer": "deer",
            "rabbit": "rabbit",
        }

        # ─────────────────────────────────────────────────────────────
        # Define all “intruding” classes (after mapping)
        # ─────────────────────────────────────────────────────────────
        self.intruding = set(self.label_map.values())

    # ─────────────────────────────────────────────────────────────
    # Run YOLO inference and apply label mapping
    # ─────────────────────────────────────────────────────────────
    def _predict(self, jpeg: bytes):
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        res = self.model.predict(img, verbose=False)[0]
        idx = res.probs.top1
        conf = float(res.probs.top1conf.item())
        raw_label = res.names[idx].lower().strip()

        # Map to unified label or mark as “other”
        label = self.label_map.get(raw_label, "other")

        # Only keep if considered intruding
        if label not in self.intruding:
            label = "other"

        print(f"[AnimalClassifier] raw={raw_label}, mapped={label}, conf={conf:.3f}")
        return label, conf

    # ─────────────────────────────────────────────────────────────
    # gRPC Classify endpoint
    # ─────────────────────────────────────────────────────────────
    def Classify(self, request, context):
        preds = []
        for crop in request.crops:
            try:
                label, conf = self._predict(crop.jpeg)
            except Exception as e:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Failed to classify crop: {e}")
                return pb2.ClassifyResponse()
            preds.append(pb2.Prediction(
                label=label,
                confidence=conf,
                x1=crop.x1, y1=crop.y1, x2=crop.x2, y2=crop.y2,
            ))

        print(f"[AnimalClassifier] → returning {len(preds)} predictions")
        return pb2.ClassifyResponse(preds=preds)


# ─────────────────────────────────────────────────────────────
# gRPC server setup
# ─────────────────────────────────────────────────────────────
def serve():
    port = int(os.getenv("PORT", "50064"))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pb2_grpc.add_ClassifierServiceServicer_to_server(AnimalClassifierServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    print(f"[AnimalClassifier] 🚀 gRPC server running on port {port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
