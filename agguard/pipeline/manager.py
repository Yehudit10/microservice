# agguard/pipeline/manager.py
from __future__ import annotations
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2

from agguard.core.roi import Roi
from agguard.core.motion import MotionGate
from agguard.specialists.yolo_detector import YoloDetector
from agguard.core.tracker import BoxMOTWrapper
from agguard.specialists.dispatch import ClassDispatch
from agguard.core.events.models import Rule
from agguard.core.events.aggregator import IncidentAggregator

# NEW: S3 client for HLS/MP4 upload
from agguard.adapters.s3_client import S3Client, S3Config
from agguard.adapters.alerting.alertmanager_client import AlertmanagerClient

import logging
log = logging.getLogger(__name__)

def _parse_roi(s: str):
    s = (s or "full").strip().lower()
    if s == "full":
        return [(0, 0), (1, 0), (1, 1), (0, 1)]
    return [tuple(map(float, p.split(","))) for p in s.split(";")]

class PipelineManager:
    def __init__(self, cfg: Dict[str, Any], repo, rules: List[Rule]):
        self.cfg = cfg
        self.repo = repo
        self.rules = rules

        self.det = YoloDetector(cfg.get("detector", {}))
        self.router = ClassDispatch(cfg.get("specialists", []))
        self.change_thresh = float(cfg.get("change_thresh", 0.02))
        self.motion_pad_px = int(cfg.get("motion_pad_px", 12))
        self._pipelines: Dict[str, Dict[str, Any]] = {}

        # -------- HLS/MP4 + S3 configuration --------
        video_cfg: Dict[str, Any] = cfg.get("video", {}) or {}
        s3_cfg_raw: Dict[str, Any] = cfg.get("s3", {}) or {}

        # Video/HLS knobs (defaults match our recommmendations)
        self.video_bucket: Optional[str] = video_cfg.get("bucket") or None
        self.video_prefix: str = (video_cfg.get("prefix") or "security/incidents").strip("/")
        self.video_fps: int = int(video_cfg.get("fps", 12))
        self.hls_segment_time: float = float(video_cfg.get("hls_segment_time", 3.0))
        self.hls_list_size: int = int(video_cfg.get("hls_list_size", 40))
        self.hls_use_cmaf: bool = bool(video_cfg.get("hls_use_cmaf", False))
        self.draw_thickness: int = int(video_cfg.get("draw_thickness", 3))

        # S3 client (optional; if no bucket -> disable video outputs)
        self.s3: Optional[S3Client] = None
        if self.video_bucket:
            # Build S3Config with safe defaults
            sc = S3Config(
                region_name=s3_cfg_raw.get("region_name", "us-east-1"),
                aws_access_key_id=s3_cfg_raw.get("aws_access_key_id"),
                aws_secret_access_key=s3_cfg_raw.get("aws_secret_access_key"),
                aws_session_token=s3_cfg_raw.get("aws_session_token"),
                endpoint_url=s3_cfg_raw.get("endpoint_url"),
                connect_timeout=float(s3_cfg_raw.get("connect_timeout", 3.0)),
                read_timeout=float(s3_cfg_raw.get("read_timeout", 10.0)),
                max_attempts=int(s3_cfg_raw.get("max_attempts", 3)),
            )
            try:
                self.s3 = S3Client(sc)
                log.info("S3 client initialized (endpoint=%s, region=%s, bucket=%s)",
                         sc.endpoint_url, sc.region_name, self.video_bucket)
            except Exception as e:
                # Fail soft: run without HLS if S3 init fails
                log.exception("Failed to init S3 client; disabling video outputs: %s", e)
                self.s3 = None
                self.video_bucket = None

        am_cfg: Dict[str, Any] = cfg.get("alertmanager", {}) or {}
        self.am = None
        am_url = am_cfg.get("base_url")
        if am_url:
            self.am = AlertmanagerClient(am_url, timeout=float(am_cfg.get("timeout", 3.0)))

    def _get_or_create(self, camera_id: str, frame_shape) -> Dict[str, Any]:
        if camera_id in self._pipelines:
            return self._pipelines[camera_id]

        h, w = frame_shape[:2]
        roi_poly = Roi.from_normalized(_parse_roi(self.cfg.get("roi")), (w, h))
        gate = MotionGate(
            roi_poly,
            min_blob_area=int(self.cfg.get("min_blob_area", 200)),
            morph_open=int(self.cfg.get("morph_open", 3)),
            target_max_dim=int(self.cfg.get("target_max_dim", 480)),
        )
        trk = BoxMOTWrapper()

        # ---- Build the IncidentAggregator with optional HLS/MP4 outputs ----
        aggregator = IncidentAggregator(
            self.rules,
            camera_id=camera_id,
            roi_pixels=roi_poly.as_cv2().reshape(-1, 2).tolist(),
            store=self.repo,
            assoc_iou=0.3,
            sample_every=1,

            # NEW (optional): live HLS + final MP4 on close
            s3=self.s3 if self.video_bucket else None,
            video_bucket=self.video_bucket,
            video_prefix=self.video_prefix,
            fps=self.video_fps,
            hls_segment_time=self.hls_segment_time,
            hls_list_size=self.hls_list_size,
            hls_use_cmaf=self.hls_use_cmaf,
            draw_thickness=self.draw_thickness,

            alert_client=self.am,                 # NEW
            media_base=f'{self.cfg["media_base"]}'.rstrip("/") if "media_base" in self.cfg else None,
            media_token=self.cfg.get("media_auth_token"),
        )

        state = {
            "roi_poly": roi_poly,
            "gate": gate,
            "trk": trk,
            "aggregator": aggregator,
            "fps_ema": None,
            "prev": time.perf_counter(),
            "frame_counter": 0,
            "full_mask": roi_poly.mask(),  # cache full mask once
        }
        self._pipelines[camera_id] = state
        return state

    def process(self, camera_id: str, ts_sec: float, frame_idx: Optional[int],
                frame_bgr: np.ndarray, src_bucket: Optional[str], src_key: Optional[str],
                return_boxes: bool) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:

        t0 = time.perf_counter()
        p = self._get_or_create(camera_id, frame_bgr.shape)

        if frame_idx is not None:
            p["frame_counter"] = int(frame_idx)
        else:
            p["frame_counter"] += 1
        frame_idx = p["frame_counter"]
        t1 = time.perf_counter()

        roi_poly, gate, trk, aggregator = (
            p["roi_poly"], p["gate"], p["trk"], p["aggregator"]
        )

        # --- Motion gate ---
        reading = gate.update(frame_bgr)
        t_gate_end = time.perf_counter()

        # --- ROI mask (currently full ROI) ---
        t_mask_start = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        curr_roi = p.get("full_mask", roi_poly.mask())
        t_mask_end = time.perf_counter()

        gate_dur = t_gate_end - t1
        mask_dur = t_mask_end - t_mask_start
        t2 = t_mask_end  # detect starts after mask

        # --- Detector / Tracker / Specialists ---
        if reading.score >= self.change_thresh:
            dets = self.det.detect(frame_bgr, curr_roi)
            t3 = time.perf_counter()
            tracks = trk.update([(d.cls, d.conf, d.bbox) for d in dets], frame_bgr)
            t4 = time.perf_counter()
            outs = self.router.run(frame_bgr, dets)
            t5 = time.perf_counter()
        else:
            dets = []
            tracks = trk.update([], frame_bgr)
            outs = {}
            t3 = t4 = t5 = time.perf_counter()

        # --- Aggregator (now also handles HLS live frames internally) ---
        agg_evt = aggregator.update(
            frame_idx=frame_idx, ts_sec=ts_sec,
            frame_bgr=frame_bgr, tracks=tracks, outputs=outs
        )
        t6 = time.perf_counter()

        # Persist per-frame detections (unchanged)
        aggregator.record_frame(
            frame_idx=frame_idx, source_bucket=src_bucket, source_key=src_key
        )
        t7 = time.perf_counter()

        now = time.perf_counter()
        inst = 1.0 / max(now - p["prev"], 1e-6)
        p["prev"] = now
        p["fps_ema"] = inst if p["fps_ema"] is None else (0.9 * p["fps_ema"] + 0.1 * inst)

        log.info(
            "timings camera=%s frame=%d init=%.3f gate=%.3f mask=%.3f detect=%.3f "
            "track=%.3f dispatch=%.3f agg_update=%.3f record=%.3f total=%.3f",
            camera_id, frame_idx,
            (t1 - t0), gate_dur, mask_dur, (t3 - t2), (t4 - t3),
            (t5 - t4), (t6 - t5), (t7 - t6), (now - t0),
        )

        boxes: List[Dict[str, Any]] = []
        if return_boxes:
            for t in tracks:
                boxes.append({
                    "x1": t.bbox[0], "y1": t.bbox[1],
                    "x2": t.bbox[2], "y2": t.bbox[3],
                    "cls": t.cls or "",
                    "conf": float(t.conf or 0.0),
                    "track_id": int(t.track_id if t.track_id is not None else -1),
                })

        result = {
            "camera_id": camera_id, "frame_idx": frame_idx,
            "change_score": float(reading.score),
            "num_detections": len(dets),
            "num_tracks": len(tracks),
            "fps_ema": float(p["fps_ema"]),
            "opened_incident_id": getattr(agg_evt, "opened_incident_id", "") if agg_evt else "",
            "updated_incident_id": getattr(agg_evt, "updated_incident_id", "") if agg_evt else "",
            "closed_incident_id": getattr(agg_evt, "closed_incident_id", "") if agg_evt else "",
        }
        log.debug(result)
        return result, boxes
