# agguard/app/grpc_server.py  ← move file here if you haven’t yet
from __future__ import annotations
import os, time, logging, yaml
from pathlib import Path
from typing import Optional
from concurrent import futures

import numpy as np  # optional; keep if you like the np.ndarray type hint
import grpc
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from agguard.logging_utils import setup_logging
from agguard.pipeline.manager import PipelineManager
from agguard.core.events.models import Rule
# CHANGED: use the service-to-service repo you asked for
from agguard.adapters.db.repo import DbRepository
from agguard.adapters.s3_client import S3Client, S3Config
# from agguard.proto_gen import ingest_pb2, ingest_pb2_grpc
from agguard.proto import ingest_pb2, ingest_pb2_grpc

log = logging.getLogger("agguard.grpc")
DEFAULT_CFG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
CFG_PATH = Path(os.getenv("AGGUARD_CFG", str(DEFAULT_CFG)))


def _build_session(total_retries: int = 3, backoff_factor: float = 0.3) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=total_retries,
        read=total_retries,
        connect=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset(["GET", "POST", "PUT", "PATCH"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    # sane default timeouts will still come from DbRepository._request
    return s


# -------------------------
# gRPC service
# -------------------------

class ImageIngestorServicer(ingest_pb2_grpc.ImageIngestorServicer):
    def __init__(self, cfg_path: Path = CFG_PATH):
        cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
        setup_logging(cfg.get("logging", {}).get("level", "INFO"),
                      cfg.get("logging", {}).get("file", None))

        # DbRepository (SERVICE-TO-SERVICE ONLY)
        api_cfg = cfg.get("api", {})
        service_token = api_cfg.get("service_token") or os.getenv("DB_API_SERVICE_TOKEN")

        # Optional: dev bootstrap (server must run with ENV=dev for /auth/_dev_bootstrap)
        dev_bootstrap = bool(api_cfg.get("dev_bootstrap", False))
        dev_service_name = api_cfg.get("dev_service_name", "ingestor")

        self.repo = DbRepository(
            api_base=api_cfg.get("base_url", "http://localhost:8080/api"),
            session=_build_session(),
            device_id=cfg.get("camera_id", os.getenv("CAMERA_ID", "dev-a")),
            mission_id=None,
            service_token=service_token,          # ← use raw service token
            dev_bootstrap=dev_bootstrap,          # ← optional in DEV only
            dev_service_name=dev_service_name,    # ← optional
            timeout_sec=int(api_cfg.get("timeout_sec", 15)),
        )

        # S3/MinIO client
        scfg = cfg.get("s3", {})
        self.s3 = S3Client(S3Config(
            region_name=scfg.get("region_name", "us-east-1"),
            aws_access_key_id=scfg.get("aws_access_key_id"),
            aws_secret_access_key=scfg.get("aws_secret_access_key"),
            endpoint_url=scfg.get("endpoint_url"),
            connect_timeout=float(scfg.get("connect_timeout", 3.0)),
            read_timeout=float(scfg.get("read_timeout", 10.0)),
            max_attempts=int(scfg.get("max_attempts", 3)),
        ))

        # Rules (same as your run.py example)
        rules = [
            # Rule(
            #     name="person.mask",
            #     target_cls="person",
            #     target_cls_id=0,
            #     attr_value="mask",
            #     min_conf=0.5,
            #     severity=4,
            #     min_consec=2,
            #     cooldown=9,
            # ),
    #             Rule(  # shooting
    #     name="person.shooting",
    #     target_cls="person",
    #     target_cls_id=0,
    #     attr_value="shooting",
    #     min_conf=1,     # tune
    #     min_consec=2,
    #     cooldown=20,
    # ),
    # Rule(  # climbing a fence
    #     name="climbing_fence",
    #     target_cls="person",
    #     target_cls_id=0,
    #     attr_value="object climbing a fence",
    #     severity=4,
    #     min_conf=0.5,     # tune
    #     min_consec=2,
    #     cooldown=12,
    # ),
    Rule(  # climbing a fence
        name="climbing_fence",
        target_cls="animal",
        target_cls_id= 1,
        attr_value="object climbing a fence",
        severity=4,
        min_conf=0.5,     # tune
        min_consec=2,
        cooldown=12,
    ),
    # Rule(  # robbery / stealing
    #     name="person.robbery",
    #     target_cls="person",
    #     target_cls_id=0,
    #     attr_value="stealing or robbery",
    #     min_conf=1,     # tune
    #     min_consec=2,
    #     cooldown=20,
    # )
        ]

        self.pm = PipelineManager(cfg, self.repo, rules)

    def _fetch_bgr_from_bucket_key(self, bucket: str, key: str) -> np.ndarray:
        # Uses your S3Client wrapper (MinIO/AWS handled in S3Client)
        return self.s3.fetch_image_bgr(bucket, key)

    # ---- RPC ----
    def ProcessImage(self, request: ingest_pb2.ProcessImageRequest, context):
        camera_id = request.camera_id or "unknown"
        ts_sec = (request.ts_millis / 1000.0) if request.ts_millis else time.time()
        frame_idx = request.frame_idx if request.frame_idx else None

        # Enforce S3-only input
        src_bucket: Optional[str] = request.s3_bucket.strip() if request.s3_bucket else None
        src_key: Optional[str] = request.s3_key.strip() if request.s3_key else None

        if not src_bucket or not src_key:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("s3_bucket and s3_key are required")
            return ingest_pb2.ProcessImageResponse()

        try:
            frame_bgr = self._fetch_bgr_from_bucket_key(src_bucket, src_key)
        except Exception as e:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(f"Failed to load image from S3: {e}")
            return ingest_pb2.ProcessImageResponse()

        # Run the pipeline
        result, boxes = self.pm.process(
            camera_id=camera_id,
            ts_sec=ts_sec,
            frame_idx=frame_idx,
            frame_bgr=frame_bgr,
            src_bucket=src_bucket,
            src_key=src_key,
            return_boxes=request.return_detections,
        )

        # Build response
        resp = ingest_pb2.ProcessImageResponse(
            camera_id=result["camera_id"],
            frame_idx=result["frame_idx"],
            change_score=result["change_score"],
            num_detections=result["num_detections"],
            num_tracks=result["num_tracks"],
            fps_estimate=result["fps_ema"],
            opened_incident_id=result["opened_incident_id"],
            updated_incident_id=result["updated_incident_id"],
            closed_incident_id=result["closed_incident_id"],
        )
        if request.return_detections:
            for b in boxes:
                resp.boxes.add(
                    x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"],
                    cls=b["cls"], conf=b["conf"], track_id=b["track_id"]
                )
        return resp


# -------------------------
# Entrypoint
# -------------------------

def serve(host: str = "0.0.0.0", port: int = 50052, workers: int = 4):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers))
    ingest_pb2_grpc.add_ImageIngestorServicer_to_server(ImageIngestorServicer(), server)
    server.add_insecure_port(f"{host}:{port}")
    log.info("gRPC server listening on %s:%d (cfg=%s)", host, port, CFG_PATH)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    serve()
