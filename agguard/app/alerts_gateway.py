# agguard/app/alerts_gateway.py

from __future__ import annotations
import os, asyncio, json, logging
from typing import Set, Dict, Tuple
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)
app = FastAPI(title="AgGuard Alerts Gateway", version="1.0")

CLIENTS: Set[WebSocket] = set()
# key = (camera, incident_id) -> canonical payload dict (only FIRING items live here)
ACTIVE: Dict[Tuple[str, str], dict] = {}

@app.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket):
    await ws.accept()
    CLIENTS.add(ws)
    try:
        # Always send a snapshot of currently active (firing) incidents
        await _broadcast_one(ws, {"type": "snapshot", "items": list(ACTIVE.values())})
        while True:
            await ws.receive_text()  # keepalive / ignore client messages
    except WebSocketDisconnect:
        CLIENTS.discard(ws)

async def _broadcast(payload: dict) -> None:
    dead = []
    txt = json.dumps(payload, ensure_ascii=False)
    for ws in list(CLIENTS):
        try:
            await ws.send_text(txt)
        except Exception:
            dead.append(ws)
    for d in dead:
        CLIENTS.discard(d)

async def _broadcast_one(ws: WebSocket, payload: dict) -> None:
    try:
        await ws.send_text(json.dumps(payload, ensure_ascii=False))
    except Exception:
        CLIENTS.discard(ws)

@app.get("/alerts/snapshot")
async def http_snapshot():
    """Optional: lets the client fetch a snapshot via HTTP if it wants."""
    return {"items": list(ACTIVE.values())}

@app.post("/am/webhook")
async def am_webhook(req: Request):
    try:
        data = await req.json()
    except Exception:
        return JSONResponse({"status": "bad json"}, status_code=400)

    alerts = data.get("alerts", []) or []
    out = []

    for a in alerts:
        labels = (a.get("labels") or {})
        ann    = (a.get("annotations") or {})

        # Robust status: some AM setups omit per-alert status; infer from endsAt
        per_alert_status = (a.get("status") or "").lower()
        if not per_alert_status:
            per_alert_status = "resolved" if a.get("endsAt") else "firing"

        raw_sev = labels.get("severity", "1")
        try:
            severity = int(raw_sev)
        except ValueError:
            severity = {"critical": 3, "warning": 2, "info": 1}.get(raw_sev.lower(), 1)

        item = {
            "alertname": labels.get("alertname"),
            "camera":    labels.get("camera") or ann.get("camera"),
            "incident_id": labels.get("incident_id") or ann.get("incident_id"),
            "anomaly":   labels.get("anomaly") or ann.get("anomaly"),
            "severity":  severity,
            
            "status":    per_alert_status,
            "hls":       ann.get("hls"),
            "vod":       ann.get("vod"),
            "summary":   ann.get("summary"),
            "startsAt":  a.get("startsAt"),
            "endsAt":    a.get("endsAt"),
        }
        out.append(item)

        cam = item.get("camera") or ""
        inc = item.get("incident_id") or ""
        if not cam or not inc:
            # Without a stable key we can't reconcile; log & skip ACTIVE update.
            log.warning("Webhook alert missing camera/incident_id; skipping ACTIVE update: %s", item)
            continue

        key = (cam, inc)
        if item["status"] == "firing":
            ACTIVE[key] = item
        else:
            ACTIVE.pop(key, None)

    # 1) Send the raw delta (useful for UIs that do fine-grained merges)
    await _broadcast({"type": "am_alerts", "items": out})
    # 2) Immediately follow with a stateful snapshot (only FIRING). This guarantees convergence.
    await _broadcast({"type": "snapshot", "items": list(ACTIVE.values())})

    return {"status": "ok", "count": len(out)}
