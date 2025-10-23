# agguard/persistence/db_repo.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
import os
import requests

from .port import PersistencePort, IncidentOpenData

JSON = Dict[str, Any]


class _ServiceAuth:
    """
    Service-to-service auth using X-Service-Token only.
    Optionally bootstraps a service token in DEV via POST /auth/_dev_bootstrap.
    """

    def __init__(
        self,
        api_base: str,
        session: requests.Session,
        service_token: Optional[str] = None,
        dev_bootstrap: bool = False,
        dev_service_name: Optional[str] = None,
        timeout_sec: int = 10,
    ):
        self.api = api_base.rstrip("/")
        self.http = session
        self.timeout = timeout_sec

        token = service_token
        if (token is None) and dev_bootstrap:
            body = {
                "service_name": (dev_service_name or os.getenv("DEV_SA_NAME") or "db-api"),
                "rotate_if_exists": False,
            }
            r = self.http.post(f"{self.api}/auth/_dev_bootstrap", json=body, timeout=self.timeout)
            r.raise_for_status()
            data = r.json() or {}
            token = data.get("service_account", {}).get("raw_token") or None
            if not token:
                raise RuntimeError(
                    "Dev bootstrap returned no raw_token (service may already exist). "
                    "Provide SERVICE_TOKEN explicitly."
                )

        if not token:
            raise RuntimeError(
                "Service token required. Pass service_token=... or enable dev_bootstrap=True (DEV only)."
            )

        # Pin header for every request on this session
        self.http.headers.update({"X-Service-Token": token})

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout
        return self.http.request(method, url, **kwargs)


class DbRepository(PersistencePort):
    """
    Persistence adapter that talks to your FastAPI DB API using *only* service auth.

    Expected API:
      PUT   /incidents
      PATCH /incidents/{incident_id}
      POST  /incidents/{incident_id}/frames
      POST  /files
      GET   /files/{bucket}/{object_key}   -> 404 when not found
    """

    def __init__(
        self,
        api_base: str,
        session: requests.Session,
        device_id: str,
        mission_id: Optional[int],
        *,
        # auth (service only)
        service_token: Optional[str] = None,
        dev_bootstrap: bool = False,
        dev_service_name: Optional[str] = None,
        # http defaults
        timeout_sec: int = 15,
    ):
        self.api = api_base.rstrip("/")
        self.http = session
        self.device_id = device_id
        self.mission_id = mission_id
        self.timeout = timeout_sec

        self._auth = _ServiceAuth(
            api_base=self.api,
            session=self.http,
            service_token=service_token,
            dev_bootstrap=dev_bootstrap,
            dev_service_name=dev_service_name,
            timeout_sec=min(timeout_sec, 10),
        )

    # ---------------- internal HTTP helpers ----------------

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.api}{path}" if not path.startswith("http") else path
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout
        r = self._auth.request(method, url, **kwargs)
        if not r.ok:
            print("\n\n[DB ERROR]", r.status_code, url)
            print(r.text)   # 👈 ADD THIS LINE
        r.raise_for_status()
        return r

    def _post(self, path: str, json: Optional[JSON] = None) -> requests.Response:
        return self._request("POST", path, json=json)

    def _put(self, path: str, json: Optional[JSON] = None) -> requests.Response:
        return self._request("PUT", path, json=json)

    def _patch(self, path: str, json: Optional[JSON] = None) -> requests.Response:
        return self._request("PATCH", path, json=json)

    def _get_maybe_404(self, path: str) -> requests.Response:
        """
        GET that allows 404 (used for /files lookup). Does not raise on 404.
        Raises on other 4xx/5xx.
        """
        url = f"{self.api}{path}" if not path.startswith("http") else path
        r = self._auth.request("GET", url, timeout=self.timeout)
        if r.status_code == 404:
            return r
        r.raise_for_status()
        return r

    # ---------------- files helpers ----------------

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
            "device_id": self.device_id,
            "metadata": metadata or {},
        }

        if self.mission_id is not None:
            payload["mission_id"] = self.mission_id
        self._post("/files", json=payload)

    def _files_get_id(self, bucket: str, key: str) -> Optional[int]:
        r = self._get_maybe_404(f"/files/{bucket}/{key}")
        if r.status_code in (404, 204):
            return None
        try:
            return int((r.json() or {}).get("file_id"))
        except Exception:
            return None

    # ---------------- PersistencePort implementation ----------------

    def open_incident(self, data: IncidentOpenData) -> None:
        """
        Upsert an incident (open/start). Idempotent for the same incident_id.
        """
        payload: JSON = {
            "incident_id": data.incident_id,
            "device_id": (data.camera_id or self.device_id),
            "mission_id": self.mission_id,
            "anomaly": data.kind,  # ✅ matches DB column (text)
            "started_at": data.ts_iso,
            "frame_start": data.frame_start,
            "roi_pixels": ({"roi": data.roi} if data.roi else None),
            "meta": {"kind": data.kind} if data.kind else {},
        }

        # ✅ Include severity if provided
        if data.severity is not None:
            payload["severity"] = float(data.severity)

        self._put("/incidents", json=payload)

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

        # Normalize detection objects
        norm_dets: List[JSON] = []
        for d in detections or []:
            try:
                x1 = int(d.get("x1", 0)); y1 = int(d.get("y1", 0))
                x2 = int(d.get("x2", 0)); y2 = int(d.get("y2", 0))
                c_raw = d.get("conf", None)
                c_val = float(c_raw) if c_raw is not None else None
                tid_raw = d.get("track_id", None)
                tid_val = int(tid_raw) if tid_raw is not None else None
                norm_dets.append(
                    {"track_id": tid_val, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": c_val}
                )
            except Exception:
                continue

        frames_payload: JSON = {
            "frame_idx": frame_idx,
            "ts": ts_iso,
            "detections": norm_dets,
            "cls_name": cls_name,
            "cls_id": cls_id,
            "file_id": int(file_id) if file_id is not None else 0,
            "meta": {},
        }

        self._post(f"/incidents/{incident_id}/frames", json=frames_payload)
        return file_id

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
    ) -> None:
        """
        Close/update an incident. Optionally include severity, is_real, ack.
        """
        patch: JSON = {
            "ended_at": ended_at_iso,
            "duration_sec": float(duration_sec),
            "frame_end": int(frame_end),
            "poster_file_id": (int(poster_file_id) if poster_file_id is not None else None),
        }
        if severity is not None:
            patch["severity"] = float(severity)
        if is_real is not None:
            patch["is_real"] = bool(is_real)
        if ack is not None:
            patch["ack"] = bool(ack)

        self._patch(f"/incidents/{incident_id}", json=patch)
