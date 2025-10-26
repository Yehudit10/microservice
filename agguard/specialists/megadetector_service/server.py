#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MegaDetector gRPC microservice — using the **official Microsoft MDv5A model**.

✅ Identical inference pipeline to Colab:
   from megadetector.detection import run_detector
   model = run_detector.load_detector("MDV5A")

Compatible with AgGuard pipeline (gRPC interface identical to other classifiers).
"""

from __future__ import annotations
import os
import io
import time
import grpc
import numpy as np
from concurrent import futures
from PIL import Image

# MegaDetector official loader
from megadetector.detection import run_detector

# Proto imports
from agguard.proto import mega_detector_pb2 as pb2
from agguard.proto import mega_detector_pb2_grpc as pb2_grpc


# ─────────────────────────────────────────────
# MegaDetector wrapper (official)
# ─────────────────────────────────────────────
class SimpleMegaDetector:
    """Wrapper around the official MegaDetector (v5A or v6A)."""

    CATEGORY_MAP = {"1": "animal", "2": "person", "3": "vehicle"}

    def __init__(self, model_name: str = "MDV5A", conf: float = 0.2):
        print(f"[MegaDetector] 🔹 Loading {model_name} ...")
        t0 = time.time()

        # Load model (downloads weights automatically if needed)
        self.model = run_detector.load_detector(model_name)
        self.conf = conf

        print(f"[MegaDetector] ✅ Model loaded in {time.time() - t0:.1f}s")

    def detect(self, img: Image.Image) -> list[dict]:
        """Run official MD inference on a PIL image."""
        if not isinstance(img, Image.Image):
            raise TypeError("Input must be a PIL.Image.Image")

        # Convert to numpy array
        image_np = np.array(img)

        # Run inference
        result = self.model.generate_detections_one_image(image_np)

        detections_raw = result.get("detections", [])
        detections = []
        for d in detections_raw:
            if d.get("conf", 0) < self.conf:
                continue
            bbox = d["bbox"]  # normalized [x, y, w, h]
            detections.append({
                "category": self.CATEGORY_MAP.get(str(d["category"]), str(d["category"])),
                "conf": float(d["conf"]),
                "bbox": bbox
            })
        return detections


# ─────────────────────────────────────────────
# gRPC Servicer
# ─────────────────────────────────────────────
class MegaDetectorServicer(pb2_grpc.MegaDetectorServicer):
    """gRPC servicer wrapping the official MegaDetector."""

    def __init__(self):
        model_name = os.getenv("MODEL_NAME", "MDV5A")
        conf_thresh = float(os.getenv("CONF_THRESH", "0.2"))
        self.detector = SimpleMegaDetector(model_name=model_name, conf=conf_thresh)

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

        # Convert normalized → absolute coords
        w, h = img.size
        detections = []
        for det in detections_raw:
            x, y, bw, bh = det["bbox"]
            x1 = x * w
            y1 = y * h
            x2 = (x + bw) * w
            y2 = (y + bh) * h
            detections.append(
                pb2.Detection(
                    cls=det["category"],
                    conf=det["conf"],
                    x1=x1, y1=y1, x2=x2, y2=y2,
                )
            )

        print(f"[MegaDetector] {len(detections)} detections in {dt:.2f}s")
        print(detections)
        return pb2.DetectionResponse(detections=detections, inference_time=dt)


# ─────────────────────────────────────────────
# Server bootstrap
# ─────────────────────────────────────────────
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
