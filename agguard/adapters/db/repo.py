# agguard/persistence/db_repo.py
from __future__ import annotations

from typing import Optional, Dict, Any, List
import requests

from .port import PersistencePort, IncidentOpenData

JSON = Dict[str, Any]


class DbRepository(PersistencePort):
    """
    Persistence adapter for your FastAPI backend.

    Expected server routes:

      PUT   /incidents
            body: {
              incident_id, device_id, mission_id, anomaly_type_id?,
              started_at, frame_start, track_id?, roi_pixels?, meta?
            }

      PATCH /incidents/{incident_id}
            body: {
              ended_at?, duration_sec?, frame_end?, poster_file_id?, severity?, meta?
            }

      POST  /incidents/{incident_id}/frames
            body: {
              frame_idx, ts, detections: [
                {"x1":int,"y1":int,"x2":int,"y2":int,"conf":float|null,"track_id":int|null},
                ...
              ],
              conf?, cls_name?, cls_id?, file_id?, meta?
            }

      POST  /files
      GET   /files/{bucket}/{object_key}
    """

    def __init__(
        self,
        api_base: str,
        session: requests.Session,
        device_id: str,
        mission_id: Optional[int],
        token: Optional[str] = None,
    ):
        self.api = api_base #.rstrip("/")
        self.http = session
        self.device_id = device_id
        self.mission_id = mission_id
        if token:
            self.http.headers.update({"Authorization": f"Bearer {token}"})

    # ---------- files helpers ----------

    def _files_upsert_minimal(
        self, bucket: str, key: str, metadata: Optional[JSON] = None
    ) -> None:
        """
        Ensure a Files row exists for the original frame stored in object storage.
        """
        payload: JSON = {
            "bucket": bucket,
            "object_key": key,
            "content_type": None,
            "size_bytes": None,
            "etag": None,
            "mission_id": self.mission_id,
            "device_id": self.device_id,
            "metadata": metadata or {},
        }
        r = self.http.post(f"{self.api}/files", json=payload, timeout=10)
        r.raise_for_status()

    def _files_get_id(self, bucket: str, key: str) -> Optional[int]:
        r = self.http.get(f"{self.api}/files/{bucket}/{key}", timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        # expected JSON: {"file_id": <int>, ...}
        try:
            return int(r.json().get("file_id"))
        except Exception:
            return None

    # ---------- PersistencePort implementation ----------

    def open_incident(self, data: IncidentOpenData) -> None:
        """
        Upsert an incident (open/start). Idempotent for the same incident_id.
        """
        payload: JSON = {
            "incident_id": data.incident_id,
            "device_id": (data.camera_id or self.device_id),
            "mission_id": self.mission_id,
            "anomaly_type_id": None,  # keep if you map "kind" -> anomaly type elsewhere
            "started_at": data.ts_iso,
            "frame_start": data.frame_start,
            "roi_pixels": ({"roi": data.roi} if data.roi else None),
            "meta": {"kind": data.kind} if data.kind else {},
        }
        r = self.http.put(f"{self.api}/incidents", json=payload, timeout=15)
        r.raise_for_status()

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
    ) -> Optional[int]:
        """
        Add a per-frame record carrying ALL detections (bbox + conf + track_id) in one row.

        Returns the file_id of the source frame (if resolvable), or None.
        """
        # Ensure /files row for the ORIGINAL frame
        self._files_upsert_minimal(
            bucket,
            key,
            {
                "incident_id": incident_id,
                "frame_idx": frame_idx,
            },
        )
        file_id = self._files_get_id(bucket, key)
        print(file_id)
        # Normalize detection objects to the expected JSON schema
        norm_dets: List[JSON] = []
        for d in detections or []:
            try:
                x1 = int(d.get("x1", 0))
                y1 = int(d.get("y1", 0))
                x2 = int(d.get("x2", 0))
                y2 = int(d.get("y2", 0))
                c_raw = d.get("conf", None)
                c_val = float(c_raw) if c_raw is not None else None
                tid_raw = d.get("track_id", None)
                tid_val = int(tid_raw) if tid_raw is not None else None
                norm_dets.append(
                    {"track_id": tid_val,"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": c_val, }
                )
            except Exception:
                # skip malformed entries, but continue persisting valid ones
                continue

        frames_payload: JSON = {
            "frame_idx": frame_idx,
            "ts": ts_iso,
            "detections": norm_dets,  # <-- all detections in this frame
            "cls_name": cls_name,
            "cls_id": cls_id,
            "file_id": int(file_id) if file_id is not None else 0,
            "meta": {},
        }

        r = self.http.post(
            f"{self.api}/incidents/{incident_id}/frames", json=frames_payload, timeout=15
        )
        r.raise_for_status()
        return file_id

    def close_incident(
        self,
        incident_id: str,
        ended_at_iso: str,
        duration_sec: float,
        frame_end: int,
        poster_file_id: Optional[int],
        severity: Optional[float] = None,
    ) -> None:
        """
        Close/update an incident. Optionally include severity (mean tracks/frame).
        """
        patch: JSON = {
            "ended_at": ended_at_iso,
            "duration_sec": float(duration_sec),
            "frame_end": int(frame_end),
            "poster_file_id": (int(poster_file_id) if poster_file_id is not None else None),
        }
        if severity is not None:
            # If your API doesn't yet accept 'severity' directly, move it under 'meta'
            patch["severity"] = float(severity)

        r = self.http.patch(f"{self.api}/incidents/{incident_id}", json=patch, timeout=15)
        r.raise_for_status()
