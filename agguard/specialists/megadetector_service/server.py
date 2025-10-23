#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MegaDetector gRPC microservice — YOLOv5-based lightweight version.
Compatible with AgGuard pipeline (gRPC interface identical to other classifiers).
"""

from __future__ import annotations
import os
import io
import time
import grpc
from concurrent import futures
from PIL import Image
import numpy as np
import torch

from agguard.proto import mega_detector_pb2 as pb2
from agguard.proto import mega_detector_pb2_grpc as pb2_grpc


class SimpleMegaDetector:
    """
    Lightweight MegaDetector wrapper using YOLOv5 architecture.
    Loads the official md_v5a/v6a weights and returns detections identical to MegaDetector.
    """

    def __init__(self, model_path: str = "/app/weights/md_v5a.0.0.pt", conf: float = 0.25):
        self.model_path = model_path
        self.conf = conf
        print(f"[MegaDetector] Loading weights from {model_path} ...")
        t0 = time.time()

        # Load YOLOv5 model from ultralytics repo
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=self.model_path)
        self.model.conf = self.conf
        print(f"[MegaDetector] ✅ Model ready (load time {time.time() - t0:.1f}s)")

    def detect(self, img: np.ndarray):
        """Run inference on a numpy or PIL image and return list of detections."""
        # If PIL image, convert to numpy
        if isinstance(img, Image.Image):
            img = np.array(img)

        results = self.model(img)
        df = results.pandas().xyxy[0]  # xmin, ymin, xmax, ymax, conf, cls, name

        detections = []
        for _, row in df.iterrows():
            detections.append(
                {
                    "category": str(row["name"]),
                    "conf": float(row["confidence"]),
                    "bbox": [float(row["xmin"]), float(row["ymin"]),
                             float(row["xmax"] - row["xmin"]), float(row["ymax"] - row["ymin"])]
                }
            )
        return detections


class MegaDetectorServicer(pb2_grpc.MegaDetectorServicer):
    """gRPC servicer for MegaDetector microservice."""

    def __init__(self):
        model_name = os.getenv("MODEL_NAME", "md_v5a.0.0.pt")
        model_path = f"/app/weights/{model_name}"
        conf_thresh = float(os.getenv("CONF_THRESH", 0.25))

        self.detector = SimpleMegaDetector(model_path=model_path, conf=conf_thresh)

    def Detect(self, request, context):
        """Handle gRPC detection requests."""
        if not (request.image_bytes or request.image_path):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("No image data provided")
            return pb2.DetectionResponse()

        # Load image
        try:
            if request.image_bytes:
                img = Image.open(io.BytesIO(request.image_bytes)).convert("RGB")
            else:
                img = Image.open(request.image_path).convert("RGB")
        except Exception as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Failed to load image: {e}")
            return pb2.DetectionResponse()

        # Run inference
        t0 = time.time()
        try:
            detections_raw = self.detector.detect(img)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Inference failed: {e}")
            return pb2.DetectionResponse()
        dt = time.time() - t0

        detections = []
        for det in detections_raw:
            bbox = det["bbox"]  # [x, y, w, h]
            detections.append(
                pb2.Detection(
                    cls=det["category"],
                    conf=det["conf"],
                    x1=float(bbox[0]),
                    y1=float(bbox[1]),
                    x2=float(bbox[0] + bbox[2]),
                    y2=float(bbox[1] + bbox[3]),
                )
            )

        print(f"[MegaDetector] {len(detections)} detections in {dt:.2f}s")
        return pb2.DetectionResponse(detections=detections, inference_time=dt)


def serve():
    """Start gRPC server."""
    port = int(os.getenv("PORT", "50063"))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pb2_grpc.add_MegaDetectorServicer_to_server(MegaDetectorServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    print(f"[MegaDetector] 🚀 gRPC server listening on port {port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
