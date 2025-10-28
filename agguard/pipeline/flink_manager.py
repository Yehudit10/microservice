from __future__ import annotations
import time, logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2,os,json

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from agguard.core.roi import Roi
from agguard.core.motion import MotionGate
from agguard.specialists.clients.megadetector import MegaDetectorClient
from agguard.core.tracker import BoxMOTWrapper
from agguard.specialists.dispatch import ClassDispatch
from agguard.core.events.aggregator import IncidentAggregator
from agguard.core.events.models import Rule
from agguard.adapters.alerting.alertmanager_client import AlertmanagerClient
from agguard.adapters.db.repo import DbRepository
log = logging.getLogger(__name__)


# def _build_session(total_retries: int = 3, backoff_factor: float = 0.3) -> requests.Session:
#     s = requests.Session()
#     retry = Retry(
#         total=total_retries,
#         read=total_retries,
#         connect=total_retries,
#         backoff_factor=backoff_factor,
#         status_forcelist=(502, 503, 504),
#         allowed_methods=frozenset(["GET", "POST", "PUT", "PATCH"]),
#         raise_on_status=False,
#     )
#     adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
#     s.mount("http://", adapter)
#     s.mount("https://", adapter)
#     # sane default timeouts will still come from DbRepository._request
#     return s
class FlinkPipelineManager:
    """
    Stateful per-camera pipeline for Flink.
    Does NOT talk to DB or S3; emits events to be sent to Kafka.
    """
    def __init__(self, cfg: Dict[str, Any],s3, rules: List[Rule]):
        self.cfg = cfg
        self.rules = rules
        self.det = MegaDetectorClient(cfg.get("detector", {}))
        self.router = ClassDispatch(cfg.get("specialists", []))
        self.change_thresh = float(cfg.get("change_thresh", 0.02))
        self._states: Dict[str, Dict[str, Any]] = {}
        #added
        self.s3=s3
        am_cfg: Dict[str, Any] = cfg.get("alertmanager", {}) or {}
        am_url = am_cfg.get("base_url")
        if am_url:
            self.am = AlertmanagerClient(am_url, timeout=float(am_cfg.get("timeout", 3.0)))
    
        # DbRepository (SERVICE-TO-SERVICE ONLY)
        # api_cfg = cfg.get("api", {})
        # service_token = api_cfg.get("service_token") or os.getenv("DB_API_SERVICE_TOKEN")

        # # Optional: dev bootstrap (server must run with ENV=dev for /auth/_dev_bootstrap)
        # dev_bootstrap = bool(api_cfg.get("dev_bootstrap", False))
        # dev_service_name = api_cfg.get("dev_service_name", "ingestor")
        # self.repo = DbRepository(
        #     api_base=api_cfg.get("base_url", "http://localhost:8080/api"),
        #     session=_build_session(),
        #     device_id=cfg.get("camera_id", os.getenv("CAMERA_ID", "dev-a")),
        #     mission_id=None,
        #     service_token=service_token,          # ← use raw service token
        #     dev_bootstrap=dev_bootstrap,          # ← optional in DEV only
        #     dev_service_name=dev_service_name,    # ← optional
        #     timeout_sec=int(api_cfg.get("timeout_sec", 15)),
        # )

    def _get_or_create(self, camera_id: str, frame_shape) -> Dict[str, Any]:
        if camera_id in self._states:
            return self._states[camera_id]
        h, w = frame_shape[:2]
        roi_poly = Roi.from_normalized([(0,0),(1,0),(1,1),(0,1)], (w,h))
        gate = MotionGate(roi_poly)
        trk = BoxMOTWrapper()

        video_bucket = self.cfg.get("video_bucket", "imagery")
        media_base = self.cfg.get("media_base", "http://media-proxy:8080")

        aggregator = IncidentAggregator(
            rules=self.rules,
            camera_id=camera_id,
            # store=self.repo,                # ✅ DB repository
            s3=self.s3,                     # ✅ your S3 client (already passed into manager)
            video_bucket=video_bucket,      # ✅ bucket for uploads
            video_prefix="security/incidents",
            alert_client=getattr(self, "am", None),  # ✅ AlertmanagerClient
            media_base=media_base,          # ✅ Base URL for HLS/VOD
        )
        # aggregator = IncidentAggregator(self.rules, s3=self.s3, camera_id=camera_id)
        self._states[camera_id] = {
            "roi": roi_poly, "gate": gate, "trk": trk,
            "aggregator": aggregator, "fps_ema": None, "prev": time.perf_counter()
        }
        return self._states[camera_id]

    def process(self, camera_id: str, ts_sec: float,
                frame_idx: Optional[int], frame_bgr: np.ndarray) -> Optional[Dict[str,Any]]:
        
        # Start timing for the whole frame
        start_time = time.perf_counter()
        log.info("[FlinkPipeline] ▶ START frame_idx=%s cam=%s", frame_idx, camera_id)

        p = self._get_or_create(camera_id, frame_bgr.shape)
        gate, trk, aggregator = p["gate"], p["trk"], p["aggregator"]

        # ---- 1. Motion gate check ----
        reading = gate.update(frame_bgr)
        if reading.score < self.change_thresh:
            log.debug("[FlinkPipeline] frame_idx=%s cam=%s — skipped (static frame, score=%.4f)",
                    frame_idx, camera_id, reading.score)
            return None

        # ---- 2. Detection ----
        t_det = time.perf_counter()
        dets = self.det.detect(frame_bgr)
        log.debug("[FlinkPipeline] frame_idx=%s cam=%s — detected %d objects in %.3fs",
                frame_idx, camera_id, len(dets), time.perf_counter() - t_det)

        # ---- 3. Tracking ----
        t_trk = time.perf_counter()
        tracks = trk.update([(d.cls, d.conf, d.bbox) for d in dets], frame_bgr)
        log.debug("[FlinkPipeline] frame_idx=%s cam=%s — tracker updated %d tracks in %.3fs",
                frame_idx, camera_id, len(tracks), time.perf_counter() - t_trk)

        # ---- 4. Specialists (dispatchers) ----
        t_spec = time.perf_counter()
        outs = self.router.run(frame_bgr, dets)
        log.debug("[FlinkPipeline] frame_idx=%s cam=%s — all specialists done in %.3fs, outputs=%s",
                frame_idx, camera_id, time.perf_counter() - t_spec,
                {k: len(v) for k, v in outs.items()})

        # ---- 5. Aggregation ----
        t_agg = time.perf_counter()
        evt = aggregator.update(frame_idx, ts_sec, frame_bgr, tracks, outs)
        log.debug("[FlinkPipeline] frame_idx=%s cam=%s — aggregator done in %.3fs",
                frame_idx, camera_id, time.perf_counter() - t_agg)

        total_time = time.perf_counter() - start_time
        log.info("[FlinkPipeline] ✅ DONE frame_idx=%s cam=%s total=%.3fs evt=%s",
                frame_idx, camera_id, total_time,
                "none" if evt is None else ("open" if evt.opened_incident_id else "close" if evt.closed_incident_id else "other"))

    

        if not evt:
            return None

        # ───────────────────── OPEN INCIDENT ─────────────────────
        # inside FlinkPipelineManager.process()
        if evt.opened_incident_id:
            inc_id = evt.opened_incident_id
            data = getattr(evt, "opened_data", None)

            if not data:
                return None  # safety guard

            payload = {
                "incident_id": inc_id,
                "device_id": camera_id,
                "mission_id": None,
                "anomaly": "climbing_fence",
                "started_at": data.get("ts_iso"),
                "frame_start": data.get("frame_start"),
                "roi_pixels": {"roi": data.get("roi")} if data.get("roi") else None,
                "meta": {"kind": "climbing_fence"},
                "severity": getattr(evt, "severity", 0),
            }
            topic = "incidents.create"

        elif evt.closed_incident_id:
            inc_id = evt.closed_incident_id
            data = getattr(evt, "closed_data", None)

            if not data:
                return None

            payload = {
                "incident_id": inc_id,
                "ended_at": data.get("ended_at_iso"),
                "duration_sec": float(data.get("duration_sec", 0)),
                "frame_end": int(data.get("frame_end", 0)),
                "poster_file_id": (
                    int(data["poster_file_id"]) if data.get("poster_file_id") is not None else None
                ),
            }
            topic = "incidents.update"

        else:
            return None

        return (topic, json.dumps(payload))


