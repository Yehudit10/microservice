from __future__ import annotations
import json, time, logging
from typing import Dict, Any, Optional, Sequence
import urllib.request

log = logging.getLogger(__name__)

class AlertmanagerClient:
    """
    Minimal POSTer for Alertmanager v2: /api/v2/alerts
    """
    def __init__(self, base_url: str, timeout: float = 3.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def send(self, alerts: Sequence[Dict[str, Any]]) -> None:
        url = f"{self.base_url}/api/v2/alerts"
        data = json.dumps(list(alerts)).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 (controlled host)
                if resp.status >= 300:
                    raise RuntimeError(f"AM non-2xx: {resp.status}")
        except Exception as e:
            log.warning("Alertmanager send failed: %s", e)

    def incident_open(
        self,
        camera_id: str,
        incident_id: str,
        anomaly: str,
        severity: int,
        hls_url: str,
        vod_url: Optional[str] = None,
        extra_labels: Optional[Dict[str, str]] = None,
        extra_annotations: Optional[Dict[str, str]] = None,
    ) -> None:
        now = int(time.time())
        labels = {
            "alertname": "agguard_incident_open",
            "camera": camera_id,
            "incident_id": incident_id,
            "anomaly": anomaly,
            "severity": str(severity),       
            **(extra_labels or {}),
        }
        annotations = {
            "summary": f"[{camera_id}] {anomaly} detected",
            "hls": hls_url,
            "vod": vod_url or "",
            **(extra_annotations or {}),
        }
        self.send([{
            "labels": labels,
            "annotations": annotations,
            "startsAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }])
    

    def incident_close(
    self,
    camera_id: str,
    incident_id: str,
    *,
    anomaly: str,
    severity: int,
    extra_labels: Optional[Dict[str, str]] = None,
) -> None:
        print(severity)
        now = int(time.time())
        labels = {
            "alertname": "agguard_incident_open",
            "camera": camera_id,
            "incident_id": incident_id,
            "anomaly": anomaly,          # ← include same labels as OPEN
            "severity": str(severity),        # ← include same labels as OPEN
            **(extra_labels or {}),
        }
        self.send([{
            "labels": labels,
            "annotations": {"summary": "Incident resolved"},
            "startsAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 1)),
            "endsAt":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }])


    # def incident_close(self, camera_id: str, incident_id: str) -> None:
    #     # Alertmanager resolves by sending the same labels and a "endsAt"
    #     now = int(time.time())
    #     labels = {
    #         "alertname": "agguard_incident_open",
    #         "camera": camera_id,
    #         "incident_id": incident_id,
    #     }
    #     self.send([{
    #         "labels": labels,
    #         "annotations": {"summary": "Incident resolved"},
    #         "startsAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 1)),
    #         "endsAt":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
    #     }])
