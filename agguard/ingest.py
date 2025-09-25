# agguard/ingest.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import json, time

@dataclass(frozen=True)
class Msg:
    bucket: str
    key: str
    ts: float
    frame_idx: Optional[int]
    camera_id: Optional[str]

def parse_kafka_value(raw: bytes,
                      default_bucket: Optional[str] = None,
                      default_camera: Optional[str] = None) -> Msg:
    """
    Accepts:
      1) JSON: {"bucket":"imagery","key":"camera-01/.../image.png","ts":..., "frame_idx":..., "camera_id":...}
      2) Plain string like: "imagery/camera-01/.../image.png"
    """
    s = raw.decode("utf-8", errors="ignore").strip()
    if s and s[0] == "{":
        obj = json.loads(s)
        bucket = obj.get("bucket") or default_bucket
        if not bucket: raise ValueError("missing bucket")
        key = obj["key"]
        ts = float(obj.get("ts", time.time()))
        return Msg(bucket=bucket, key=key, ts=ts,
                   frame_idx=obj.get("frame_idx"),
                   camera_id=obj.get("camera_id", default_camera))

    # plain path: "imagery/camera-01/.../image.png"
    parts = s.split("/", 1)
    if len(parts) == 2:
        bucket, key = parts[0], parts[1]
    else:
        # no leading bucket; use default
        if not default_bucket:
            raise ValueError("plain path missing bucket and no default_bucket provided")
        bucket, key = default_bucket, s
    return Msg(bucket=bucket, key=key, ts=time.time(),
               frame_idx=None, camera_id=default_camera)
