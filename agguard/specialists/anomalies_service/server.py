from __future__ import annotations
import os, logging, warnings
from concurrent import futures
from typing import List, Tuple, Dict

import numpy as np
import cv2
import grpc
from PIL import Image
import torch
import open_clip

from agguard.proto import mask_classifier_pb2 as pb
from agguard.proto import mask_classifier_pb2_grpc as pbrpc

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

Box = Tuple[int, int, int, int]

# --- tolerate either service name in generated stubs (ClassifierService|Classifier) ---
ServicerBase = getattr(pbrpc, "ClassifierServiceServicer",
                       getattr(pbrpc, "ClassifierServicer", None))
if ServicerBase is None:
    raise ImportError("No Classifier{Service}Servicer in mask_classifier_pb2_grpc.py")

add_servicer = getattr(pbrpc, "add_ClassifierServiceServicer_to_server",
                       getattr(pbrpc, "add_ClassifierServicer_to_server", None))
if add_servicer is None:
    raise ImportError("No add_Classifier{Service}Servicer_to_server in mask_classifier_pb2_grpc.py")


def _jpeg_to_pil(jpeg_bytes: bytes, size: int) -> Image.Image:
    """Decode JPEG -> RGB PIL image resized to model input."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode JPEG")
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


class ClipClassifierServicer(ServicerBase):
    """
    gRPC service using OpenCLIP RN50 to estimate the probability that a given subject is climbing a fence.
    """

    def __init__(self):
        # --- Environment configuration ---
        self.model_name   = os.environ.get("CLIP_MODEL", "RN50")
        self.pretrained   = os.environ.get("CLIP_PRETRAINED", "openai")
        self.input_size   = int(os.environ.get("CLIP_INPUT_SIZE", "224"))
        self.temperature  = float(os.environ.get("CLIP_TEMPERATURE", "100.0"))
        self.batch_size   = int(os.environ.get("CLIP_BATCH", "32"))

        device_env = os.environ.get("DEVICE")
        self.device = torch.device(device_env if device_env else ("cuda" if torch.cuda.is_available() else "cpu"))

        # optional performance flags
        enable_mkldnn = os.environ.get("ENABLE_MKLDNN", "1").lower() not in ("0", "false")
        num_threads = os.environ.get("NUM_THREADS")
        if enable_mkldnn and self.device.type == "cpu":
            torch.backends.mkldnn.enabled = True
        if num_threads and self.device.type == "cpu":
            torch.set_num_threads(int(num_threads))

        warnings.filterwarnings("ignore", message="QuickGELU mismatch.*")

        # --- Load CLIP model (always RN50 openai) ---
        log.info("Loading OpenCLIP model %s/%s ...", self.model_name, self.pretrained)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model = self.model.to(self.device).eval()
        self._emb_dim = getattr(self.model.visual, "output_dim", 1024)
        log.info("Model ready on %s", self.device)

    # =====================================================
    # 🔹 Encode helper
    # =====================================================
    def _encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        if not images:
            return torch.empty(0, self._emb_dim, device=self.device)
        xs = [self.preprocess(im).unsqueeze(0) for im in images]
        x = torch.cat(xs, dim=0).to(self.device)
        with torch.inference_mode():
            feats = []
            for i in range(0, len(x), self.batch_size):
                z = self.model.encode_image(x[i:i + self.batch_size])
                z = z / z.norm(dim=-1, keepdim=True)
                feats.append(z)
            return torch.cat(feats, dim=0)

    # =====================================================
    # 🔹 Classification logic
    # =====================================================
    def _classify_one(self, image_emb: torch.Tensor, subject: str) -> float:
        """
        Compute probability that {subject} is climbing a fence.
        """
        subj = (subject or "object").strip().lower()
        positive_prompt = f"a {subj} climbing a fence"
        negative_prompt = f"an image of a {subj} not climbing a fence"

        text_tokens = self.tokenizer([positive_prompt, negative_prompt]).to(self.device)
        with torch.inference_mode():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            logits = (self.temperature * image_emb @ text_features.T).softmax(dim=-1)
            prob_climbing = float(logits[0, 0].item())  # probability of positive prompt
        return prob_climbing

    # =====================================================
    # 🔹 gRPC entrypoint
    # =====================================================
    def Classify(self, request: pb.ClassifyRequest, context) -> pb.ClassifyResponse:
        images: List[Image.Image] = []
        boxes: List[Box] = []
        subjects: List[str] = []

        for c in request.crops:
            try:
                im = _jpeg_to_pil(c.jpeg, self.input_size)
                images.append(im)
                boxes.append((c.x1, c.y1, c.x2, c.y2))
                subj = getattr(c, "subject", "") or "object"
                subjects.append(subj)
            except Exception as e:
                log.warning("Failed to decode crop: %s", e)

        resp = pb.ClassifyResponse()
        if not images:
            return resp

        img_emb = self._encode_images(images)

        with torch.inference_mode():
            for j, box in enumerate(boxes):
                prob = self._classify_one(img_emb[j:j+1], subjects[j])
                label = f"{subjects[j]} climbing a fence"
                resp.preds.add(
                    x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                    label=label,
                    confidence=prob
                )
        print(resp)
        return resp


def serve():
    port = int(os.environ.get("PORT", "50062"))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    add_servicer(ClipClassifierServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    log.info("CLIP climbing classifier gRPC server listening on :%d", port)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
