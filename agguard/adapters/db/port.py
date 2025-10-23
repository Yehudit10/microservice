# agguard/persistence/port.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Protocol, List, Dict, Any
from uuid import UUID

Box = Tuple[int, int, int, int]

@dataclass
class IncidentOpenData:
    incident_id: str
    kind: str                    # → maps to anomaly (text)
    frame_start: int
    ts_iso: str                  # started_at as ISO8601
    camera_id: Optional[str]     # used as device_id on server
    roi: Optional[list] = None   # already JSON-serializable
    severity: Optional[float] = None

class PersistencePort(Protocol):
    def open_incident(self, data: IncidentOpenData) -> None: ...

    def add_frame_ref(
        self,
        incident_id: str,
        frame_idx: int,
        ts_iso: str,
        detections: List[Dict[str, Any]],
        cls_name: Optional[str],
        cls_id: Optional[str],
        bucket: str,
        key: str,
    ) -> Optional[int]: ...

    def close_incident(
        self,
        incident_id: str,
        ended_at_iso: str,
        duration_sec: float,
        frame_end: int,
        poster_file_id: Optional[int],
        severity: Optional[float] = None,
        is_real: Optional[bool] = None,
        ack: Optional[bool] = None,
    ) -> None: ...
