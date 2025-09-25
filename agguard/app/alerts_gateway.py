# agguard/app/alerts_gateway.py


from __future__ import annotations 
import os, asyncio, json, logging 
from typing import Set, Dict, Tuple
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request 
from fastapi.responses import JSONResponse 
log = logging.getLogger(__name__) 
app = FastAPI(title="AgGuard Alerts Gateway", version="1.0")


CLIENTS: Set[WebSocket] = set()
# key = (camera, incident_id) -> canonical payload dict (like your "out" items)
ACTIVE: Dict[Tuple[str, str], dict] = {}

@app.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket):
    await ws.accept()
    CLIENTS.add(ws)
    try:
        # send snapshot of currently active incidents
        if ACTIVE:
            await ws.send_text(json.dumps({"type": "snapshot", "items": list(ACTIVE.values())}, ensure_ascii=False))
        while True:
            await ws.receive_text()
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

@app.post("/am/webhook")
async def am_webhook(req: Request):
    try:
        data = await req.json()
    except Exception:
        return JSONResponse({"status": "bad json"}, status_code=400)

    alerts = data.get("alerts", [])
    out = []
    for a in alerts:
        labels = a.get("labels", {}) or {}
        ann = a.get("annotations", {}) or {}
        item = {
            "alertname": labels.get("alertname"),
            "camera": labels.get("camera"),
            "incident_id": labels.get("incident_id"),
            "anomaly": labels.get("anomaly"),
            "severity": labels.get("severity"),
            "status": a.get("status"),
            "hls": ann.get("hls"),
            "vod": ann.get("vod"),
            "summary": ann.get("summary"),
            "startsAt": a.get("startsAt"),
            "endsAt": a.get("endsAt"),
        }
        out.append(item)

        cam = item.get("camera")
        inc = item.get("incident_id")
        key = (cam, inc)
        if item.get("status") == "firing":
            ACTIVE[key] = item
        elif item.get("status") == "resolved" and key in ACTIVE:
            del ACTIVE[key]

    await _broadcast({"type": "am_alerts", "items": out})
    return {"status": "ok", "count": len(out)}
