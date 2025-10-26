from __future__ import annotations
import time, logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2

from agguard.core.roi import Roi
from agguard.core.motion import MotionGate
from agguard.specialists.clients.megadetector import MegaDetectorClient
from agguard.core.tracker import BoxMOTWrapper
from agguard.specialists.dispatch import ClassDispatch
from agguard.core.events.aggregator import IncidentAggregator
from agguard.core.events.models import Rule

log = logging.getLogger(__name__)

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

    def _get_or_create(self, camera_id: str, frame_shape) -> Dict[str, Any]:
        if camera_id in self._states:
            return self._states[camera_id]
        h, w = frame_shape[:2]
        roi_poly = Roi.from_normalized([(0,0),(1,0),(1,1),(0,1)], (w,h))
        gate = MotionGate(roi_poly)
        trk = BoxMOTWrapper()
        aggregator = IncidentAggregator(self.rules, camera_id=camera_id)
        self._states[camera_id] = {
            "roi": roi_poly, "gate": gate, "trk": trk,
            "aggregator": aggregator, "fps_ema": None, "prev": time.perf_counter()
        }
        return self._states[camera_id]

    def process(self, camera_id: str, ts_sec: float,
                frame_idx: Optional[int], frame_bgr: np.ndarray) -> Optional[Dict[str,Any]]:
        p = self._get_or_create(camera_id, frame_bgr.shape)
        gate, trk, aggregator = p["gate"], p["trk"], p["aggregator"]

        reading = gate.update(frame_bgr)
        if reading.score < self.change_thresh:
            return None  # skip static frames

        dets = self.det.detect(frame_bgr)
        tracks = trk.update([(d.cls, d.conf, d.bbox) for d in dets], frame_bgr)
        outs = self.router.run(frame_bgr, dets)
        evt = aggregator.update(frame_idx, ts_sec, frame_bgr, tracks, outs)

        if not evt:
            return None

        if evt.opened_incident_id:
            type_ = "OPEN"; inc_id = evt.opened_incident_id
        elif evt.closed_incident_id:
            type_ = "CLOSE"; inc_id = evt.closed_incident_id
        else:
            type_ = "UPDATE"; inc_id = evt.updated_incident_id

        return {
            "type": type_,
            "camera_id": camera_id,
            "incident_id": inc_id,
            "severity": getattr(evt, "severity", 0),
            "ts_sec": ts_sec,
            "frame_idx": frame_idx,
            "num_detections": len(dets)
        }
