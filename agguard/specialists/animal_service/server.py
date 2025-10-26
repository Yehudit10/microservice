#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animal Classifier microservice — reusing mask-classifier.proto.
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
        model_path = os.getenv("MODEL_PATH", "/app/weights/yolov8n-cls.pt")
        print(f"[AnimalClassifier] Loading {model_path} ...")
        self.model = YOLO(model_path)
        self.intruding = {"wild boar", "boar", "pig", "bear", "snake", "rabbit", "wolf", "fox", "deer"}

    def _predict(self, jpeg: bytes):
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        res = self.model.predict(img, verbose=False)[0]
        idx = res.probs.top1
        conf = float(res.probs.top1conf.item())
        label = res.names[idx].lower().strip()
        if label not in self.intruding:
            label = "other"
        return label, conf

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
                label=label, confidence=conf,
                x1=crop.x1, y1=crop.y1, x2=crop.x2, y2=crop.y2
            ))
        return pb2.ClassifyResponse(preds=preds)


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
