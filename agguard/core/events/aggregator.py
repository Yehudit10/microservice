# agguard/events/aggregator.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import uuid, datetime as _dt
import cv2, numpy as np

from .models import Rule, Incident, Box
from agguard.adapters.db.port import PersistencePort, IncidentOpenData
from agguard.media.hls_recorder import HlsRecorder, HlsConfig
from agguard.media.mp4_recorder import Mp4Recorder
@dataclass
class IncidentEvent:
    opened_incident_id: str | None = None
    updated_incident_id: str | None = None
    closed_incident_id: str | None = None

@dataclass
class _EventState:
    consec: int = 0
    cooldown_left: int = 0
    open_incident: Optional[Incident] = None
    last_seen_frame: int = -1
    last_seen_ts: float = 0.0
    # Only event-matching detections (for UI/HLS/DB)
    detections: List[Dict[str, Any]] = field(default_factory=list)
    total_tracks: int = 0
    total_frames: int = 0

def _iou(a: Box, b: Box) -> float:
    x1,y1,x2,y2 = a
    X1,Y1,X2,Y2 = b
    ix1,iy1 = max(x1,X1), max(y1,Y1)
    ix2,iy2 = min(x2,X2), min(y2,Y2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (x2-x1)*(y2-y1) + (X2-X1)*(Y2-Y1) - inter
    return inter / max(union, 1e-6)

def _clamp_box(box: Box, w: int, h: int) -> Box:
    x1,y1,x2,y2 = box
    x1 = max(0, min(w-1, int(x1))); y1 = max(0, min(h-1, int(y1)))
    x2 = max(0, min(w-1, int(x2))); y2 = max(0, min(h-1, int(y2)))
    if x2 < x1: x1,x2 = x2,x1
    if y2 < y1: y1,y2 = y2,y1
    return (x1,y1,x2,y2)

class IncidentAggregator:
    """Aggregates evidence and manages HLS recording."""
    def __init__(
        self,
        rules: List[Rule],
        camera_id: Optional[str] = None,
        roi_pixels: Optional[List[Tuple[int, int]]] = None,
        store: Optional[PersistencePort] = None,
        assoc_iou: float = 0.3,
        sample_every: int = 1,
        s3=None, video_bucket=None, video_prefix="security/incidents",
        fps=5, hls_segment_time=1.0, hls_list_size=240, hls_use_cmaf=False,
        draw_thickness=2, alert_client=None, media_base: Optional[str] = None, media_token: Optional[str] = None
    ):
        self.rules = rules
        self.camera_id = camera_id
        self.roi_pixels = roi_pixels
        self.store = store
        self.assoc_iou = float(assoc_iou)
        self.sample_every = int(sample_every)
        self._states: Dict[Tuple[str, str], _EventState] = {}

        self.s3 = s3
        self.video_bucket = video_bucket
        self.video_prefix = (video_prefix or "security/incidents").strip("/")
        self.fps = int(max(1, fps))
        self._hls_cfg = HlsConfig(
            fps=self.fps, segment_time=float(hls_segment_time) or 1.0,
            list_size=int(hls_list_size), use_cmaf=bool(hls_use_cmaf),
            preset="veryfast", crf=23, gop_segments=1, upload_interval_sec=0.02
        )

        self.draw_thickness = int(max(1, draw_thickness))
        self.alert_client = alert_client
        self.media_base = (media_base or "").rstrip("/")
        self.media_token = media_token or ""

    def _render_frame_with_boxes(self, frame_bgr, dets: List[Dict[str, Any]]):
        """Draw only provided detections (event-matching)."""
        out = frame_bgr.copy()
        t = self.draw_thickness
        for d in dets or []:
            x1,y1,x2,y2 = int(d["x1"]),int(d["y1"]),int(d["x2"]),int(d["y2"])
            cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), t)
            tid = d.get("track_id")
            if tid is not None and tid != -1:
                cv2.putText(out, str(tid), (x1, max(0,y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        return out

    def _hls_prefix(self, inc):
        cam = self.camera_id or "unknown"
        return f"{self.video_prefix}/{cam}/{inc.incident_id}"

    def set_camera(self, camera_id: Optional[str]) -> None: self.camera_id = camera_id
    def set_store(self, store: Optional[PersistencePort]) -> None: self.store = store
    def set_roi_pixels(self, roi_pixels: Optional[List[Tuple[int, int]]]) -> None: self.roi_pixels = roi_pixels

    def _key(self, rule: Rule) -> Tuple[str, str]:
        cam = self.camera_id or "unknown"
        return (cam, rule.name)

    @staticmethod
    def _class_match(t_cls: Any, rule: Rule) -> bool:
        t_name = str(t_cls).lower()
        by_name = (rule.target_cls and t_name == str(rule.target_cls).lower())
        by_id = (rule.target_cls_id is not None and str(t_cls) == str(rule.target_cls_id))
        return bool(by_name or by_id) if (rule.target_cls or rule.target_cls_id is not None) else True

    def _match_attr(self, rule: Rule, track_box: Box, preds: List) -> bool:
        best = None
        for p in preds or []:
            try:
                p_box = getattr(p, "box", (p.x1, p.y1, p.x2, p.y2))
                if _iou(track_box, p_box) >= self.assoc_iou:
                    print(p.confidence)
                    c = float(p.confidence)
                    if best is None or c > best:
                        best = c#(str(p.label).lower(), c)


            except Exception as e:
                print("error", e)
        if rule.attr_value is None:
            return True
        # want = str(rule.attr_value).lower()
        minc = getattr(rule, "min_conf", None)
        # bool(best and best[0] == want
        return  best is not None and (minc is None or best >= float(minc))

    def _open_incident(self, st: _EventState, rule: Rule, ts_sec: float, frame_idx: int, frame_bgr) -> None:
        import uuid

# Keep inc.incident_id as UUID object
        inc = Incident(
            incident_id=str(uuid.uuid4()),
            kind=rule.name,
            camera_id=self.camera_id,
            severity=rule.severity,
            started_ts=ts_sec,
            frame_start=frame_idx,
            roi=self.roi_pixels,
        )


        
        st.open_incident = inc
        st.cooldown_left = int(rule.cooldown)
        st.total_tracks = 0
        st.total_frames = 0

        if self.s3 and self.video_bucket:
            # --- Live HLS recorder ---
            st._hls = HlsRecorder(
                s3=self.s3, bucket=self.video_bucket,
                prefix=self._hls_prefix(inc), cfg=self._hls_cfg,
            )
            H, W = frame_bgr.shape[:2]
            st._hls.start((H, W))
            st._hls.write_bgr(self._render_frame_with_boxes(frame_bgr, st.detections))

            # --- MP4 recorder (for final video only) ---
            st._mp4 = Mp4Recorder(
                s3=self.s3, bucket=self.video_bucket,
                prefix=self._hls_prefix(inc), cfg=self._hls_cfg,
            )
            st._mp4.start((H, W))
            st._mp4.write_bgr(self._render_frame_with_boxes(frame_bgr, st.detections))


        if self.store:
            self.store.open_incident(
        IncidentOpenData(
            incident_id=inc.incident_id,  # ✅ serialize to string
            kind=rule.name,
            frame_start=frame_idx,
            ts_iso=_dt.datetime.utcfromtimestamp(ts_sec).isoformat() + "Z",
            camera_id=self.camera_id,
            roi=self.roi_pixels,
            severity=float(rule.severity or 0.0),
        )
    )

        if self.alert_client and self.media_base and hasattr(st, "_hls") and st._hls:
            st._hls.wait_ready(timeout=6.0)
            camera = inc.camera_id
            incident_id = inc.incident_id
            hls_url = f"{self.media_base}/hls/{camera}/{incident_id}/index.m3u8"
            vod_url = f"{self.media_base}/vod/{camera}/{incident_id}/final.mp4"
            self.alert_client.incident_open(
                camera_id=camera, incident_id=incident_id,
                anomaly=(inc.kind or "unknown"), severity=rule.severity,
                hls_url=hls_url, vod_url=vod_url,
                extra_annotations={"token_hint": "Authorization: Bearer <token>"},
            )

    def _close_incident(self, key: Tuple[str, str], st: _EventState, ts_sec: float, frame_idx: int):
        inc = st.open_incident
        if not inc: return
        inc.ended_ts = ts_sec
        inc.frame_end = frame_idx
        inc.duration_sec = max(0.0, inc.ended_ts - inc.started_ts)
        incident_severity = float(st.total_tracks) / max(st.total_frames, 1)
        inc.severity=inc.severity + round(min(3,incident_severity))-1
        self._states[key] = _EventState()
        if self.alert_client:
            self.alert_client.incident_close(
                inc.camera_id,
                inc.incident_id,
                anomaly=(inc.kind or "unknown"),
                severity=inc.severity
                # extra_labels=...  # pass if you added any at open
            )

        poster_file_id = None
        if self.s3 and self.video_bucket and hasattr(st, "_hls") and st._hls:
            try:
                mp4_key = st._mp4.finalize() if hasattr(st, "_mp4") and st._mp4 else None
                st._hls.cleanup_local()
                if self.store and hasattr(self.store, "_files_upsert_minimal") and hasattr(self.store, "_files_get_id"):
                    self.store._files_upsert_minimal(self.video_bucket, mp4_key, {"incident_id": inc.incident_id})
                    poster_file_id = self.store._files_get_id(self.video_bucket, mp4_key)
            except Exception:
                poster_file_id = None
        if self.store:
            self.store.close_incident(
        incident_id=inc.incident_id,
        ended_at_iso=_dt.datetime.utcfromtimestamp(ts_sec).isoformat() + "Z",
        duration_sec=inc.duration_sec,
        frame_end=frame_idx,
        poster_file_id=poster_file_id,
        severity=inc.severity,
    )

       

    def update(self, frame_idx: int, ts_sec: float, frame_bgr, tracks: List, outputs: Dict[str, List]) -> IncidentEvent:
        H, W = frame_bgr.shape[:2]
        by_cls = outputs or {}
        evt = IncidentEvent()

        for rule in self.rules:
            candidate_tracks = [t for t in tracks]# if self._class_match(t.cls, rule)]
            preds = by_cls.get(rule.target_cls, []) if rule.target_cls else []
            
            evidence = False
            det_event: List[Dict[str, Any]] = []
            for t in candidate_tracks:
                bx = _clamp_box(tuple(map(int, t.bbox)), W, H)
                matched = self._match_attr(rule, bx, preds)
                if matched:
                    evidence = True
                    x1, y1, x2, y2 = bx
                    # Ensure downstream fields always exist with safe types/defaults
                    try:
                        conf_val = float(getattr(t, "conf", 0.0))
                    except Exception:
                        conf_val = 0.0
                    cls_val = str(getattr(t, "cls", "") if getattr(t, "cls", None) is not None else "")
                    tid_val = int(getattr(t, "track_id", -1) if getattr(t, "track_id", None) is not None else -1)
                    det_event.append({
                        "track_id": tid_val,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": conf_val,
                        "cls": cls_val,
                    })

            st = self._states.setdefault(self._key(rule), _EventState())

            if st.open_incident is not None:
                rendered = self._render_frame_with_boxes(frame_bgr, st.detections)
                if hasattr(st, "_hls") and st._hls:
                    st._hls.write_bgr(rendered)  # continuous live feed
                if hasattr(st, "_mp4") and st._mp4:
                    st._mp4.write_bgr(rendered)  # only called when new frames arrive

                st.total_tracks += len(st.detections)
                st.total_frames += 1

            st.last_seen_frame = frame_idx
            st.last_seen_ts = ts_sec
            st.detections = det_event  # only event boxes

            st.consec = st.consec + 1 if evidence else 0
            key = self._key(rule)

            if st.open_incident is None and st.consec >= int(rule.min_consec or 1):
                self._open_incident(st, rule, ts_sec, frame_idx, frame_bgr)
                evt.opened_incident_id = st.open_incident.incident_id

            if st.open_incident is not None:
                if self.sample_every > 0 and (frame_idx % self.sample_every == 0) and st.detections:
                    bx = max(((d["x2"]-d["x1"])*(d["y2"]-d["y1"]), d) for d in st.detections)[1]
                    st.open_incident.boxes.append((bx["x1"],bx["y1"],bx["x2"],bx["y2"]))
                    st.open_incident.confs.append(1.0)

                st.cooldown_left = int(rule.cooldown) if evidence else (st.cooldown_left - 1)
                if not evidence and st.cooldown_left <= 0:
                    closed_id = st.open_incident.incident_id
                    self._close_incident(key, st, ts_sec, frame_idx)
                    evt.closed_incident_id = closed_id
                else:
                    evt.updated_incident_id = st.open_incident.incident_id

        return evt

    def visible_boxes(self) -> List[Dict[str, Any]]:
        """Union of event-matching boxes for this camera on the latest frame."""
        cam = (self.camera_id or "unknown")
        boxes: List[Dict[str, Any]] = []
        for (cam_id, _), st in self._states.items():
            if cam_id != cam:
                continue
            if st.detections:
                boxes.extend(st.detections)
        return boxes

    def record_frame(self, frame_idx: int, *, source_bucket: Optional[str], source_key: Optional[str]):
        if not self.store or not source_bucket or not source_key:
            return
        cam = (self.camera_id or "unknown")
        for (cam_id, _), st in list(self._states.items()):
            if cam_id != cam or st.open_incident is None:
                continue
            self.store.add_frame_ref(
                incident_id=st.open_incident.incident_id,
                frame_idx=frame_idx,
                ts_iso=_dt.datetime.utcfromtimestamp(st.last_seen_ts).isoformat() + "Z",
                detections=st.detections,  # event-only detections
                cls_name=None, cls_id=None,
                bucket=source_bucket, key=source_key,
            )

    def flush(self, ts_sec: float, frame_idx: int):
        for key, st in list(self._states.items()):
            if st.open_incident is not None:
                self._close_incident(key, st, ts_sec, frame_idx)
