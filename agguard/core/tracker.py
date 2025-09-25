# from __future__ import annotations
# from dataclasses import dataclass
# from enum import Enum, auto
# from typing import Dict, List, Tuple, Optional, Set
# import logging

from .types import Track #as BaseTrack, BBox

# log = logging.getLogger(__name__)

# # Optional tiny appearance features via OpenCV (HSV hist)
# try:
#     import cv2, numpy as np
#     _HAS_CV = True
# except Exception:
#     _HAS_CV = False

# def _iou(a: BBox, b: BBox) -> float:
#     ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
#     ix1, iy1 = max(ax1, bx1), max(ay1, by1); ix2, iy2 = min(ax2, bx2), min(ay2, by2)
#     iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
#     inter = iw * ih
#     if inter <= 0:
#         return 0.0
#     aA = (ax2 - ax1 + 1) * (ay2 - ay1 + 1); bA = (bx2 - bx1 + 1) * (by2 - by1 + 1)
#     return inter / float(aA + bA - inter)

# class TState(Enum):
#     TENTATIVE = auto()
#     CONFIRMED = auto()

# @dataclass
# class Track(BaseTrack):
#     state: TState = TState.TENTATIVE
#     vx: float = 0.0
#     vy: float = 0.0
#     feat: Optional["np.ndarray"] = None  # type: ignore[name-defined]

# class RobustSORT:
#     """
#     Robust SORT-style tracker:
#       - tentative/confirmed with min_hits
#       - two-stage association (high/low conf)
#       - EMA bbox smoothing
#       - constant-velocity center model
#       - optional tiny appearance cue (HSV hist)
#       - predict_only() for quiet frames (no detection)
#     """

#     def __init__(
#         self,
#         iou_thresh=0.5,
#         max_miss=20,
#         ema=0.25,
#         min_hits=3,
#         high_conf=0.5,
#         appearance_alpha=0.75,
#         center_blend=0.15,
#     ):
#         self.iou_thresh = float(iou_thresh)
#         self.max_miss = int(max_miss)
#         self.ema = float(ema)
#         self.min_hits = int(min_hits)
#         self.high_conf = float(high_conf)
#         self.appearance_a = float(appearance_alpha)
#         self.center_blend = float(center_blend)

#         self._next = 1
#         self._tracks: Dict[int, Track] = {}
#         self._frame: Optional["np.ndarray"] = None  # type: ignore[name-defined]

#     # public API
#     def set_frame(self, frame_bgr: "np.ndarray") -> None:  # type: ignore[name-defined]
#         self._frame = frame_bgr if _HAS_CV else None

#     def update(self, dets: List[Tuple[str, float, BBox]]) -> List[Track]:
#         # 1) predict
#         for tid, tr in list(self._tracks.items()):
#             self._tracks[tid].bbox = self._predict(tr)

#         # 2) split detections
#         hi, lo = [], []
#         for j, (cls, conf, bb) in enumerate(dets):
#             (hi if conf >= self.high_conf else lo).append((j, (cls, conf, bb)))

#         used: Set[int] = set()
#         matched = set()
#         matched |= self._associate_greedy(hi, dets, used, only_confirmed=True)
#         matched |= self._associate_greedy(lo, dets, used, only_confirmed=False)

#         matched_tids = {tid for tid, _ in matched}
#         # 3) update matched tracks
#         for tid, j in matched:
#             cls, conf, bb = dets[j]
#             self._update(self._tracks[tid], cls, conf, bb)

#         # 4) age unmatched
#         for tid, tr in list(self._tracks.items()):
#             if tid not in matched_tids:
#                 tr.miss += 1

#         # 5) create new tracks for leftover HIGH-conf detections (fixed logic)
#         hi_indices = [j for (j, _) in hi]
#         for j in hi_indices:
#             if j not in used:
#                 cls, conf, bb = dets[j]
#                 self._create(cls, conf, bb)

#         # 6) drop stale + promote tentative
#         for tid in list(self._tracks.keys()):
#             if self._tracks[tid].miss > self.max_miss:
#                 del self._tracks[tid]

#         for tr in self._tracks.values():
#             if tr.state is TState.TENTATIVE and tr.hits >= self.min_hits:
#                 tr.state = TState.CONFIRMED

#         return list(self._tracks.values())

#     def predict_only(self) -> List[Track]:
#         for tid, tr in list(self._tracks.items()):
#             self._tracks[tid].bbox = self._predict(tr)
#             tr.miss += 1
#         for tid in list(self._tracks.keys()):
#             if self._tracks[tid].miss > self.max_miss:
#                 del self._tracks[tid]
#         return list(self._tracks.values())

#     # internals
#     def _predict(self, tr: Track) -> BBox:
#         x1, y1, x2, y2 = tr.bbox
#         cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
#         w, h = (x2 - x1), (y2 - y1)
#         cxp, cyp = cx + tr.vx, cy + tr.vy
#         cx = (1 - self.center_blend) * cx + self.center_blend * cxp
#         cy = (1 - self.center_blend) * cy + self.center_blend * cyp
#         return (int(cx - w * 0.5), int(cy - h * 0.5), int(cx + w * 0.5), int(cy + h * 0.5))

#     def _smooth(self, a: BBox, b: BBox, t: float) -> BBox:
#         if t <= 0:
#             return b
#         ax1, ay1, ax2, ay2 = a
#         bx1, by1, bx2, by2 = b
#         return (
#             int((1 - t) * ax1 + t * bx1),
#             int((1 - t) * ay1 + t * by1),
#             int((1 - t) * ax2 + t * bx2),
#             int((1 - t) * ay2 + t * by2),
#         )

#     def _update(self, tr: Track, cls: str, conf: float, bb: BBox) -> None:
#         ax1, ay1, ax2, ay2 = tr.bbox
#         bx1, by1, bx2, by2 = bb
#         acx, acy = (ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5
#         bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
#         tr.vx = 0.5 * tr.vx + 0.5 * (bcx - acx)
#         tr.vy = 0.5 * tr.vy + 0.5 * (bcy - acy)
#         tr.bbox = self._smooth(tr.bbox, bb, self.ema)
#         tr.cls = cls
#         tr.conf = conf
#         tr.hits += 1
#         tr.miss = 0
#         if self._frame is not None and _HAS_CV:
#             tr.feat = self._feat(self._frame, tr.bbox)

#     def _create(self, cls: str, conf: float, bb: BBox) -> None:
#         tid = self._next
#         self._next += 1
#         tr = Track(track_id=tid, cls=cls, conf=conf, bbox=bb, hits=1, miss=0, state=TState.TENTATIVE)
#         if self._frame is not None and _HAS_CV:
#             tr.feat = self._feat(self._frame, bb)
#         self._tracks[tid] = tr

#     def _associate_greedy(self, det_pack, dets, used_det: Set[int], only_confirmed: bool):
#         # group by class
#         by_cls: Dict[str, Dict[str, list]] = {}
#         for j, (cls, conf, bb) in det_pack:
#             d = by_cls.setdefault(cls, {"js": [], "bbs": []})
#             d["js"].append(j)
#             d["bbs"].append(bb)

#         matches = set()

#         for cls, pack in by_cls.items():
#             js, bbs = pack["js"], pack["bbs"]
#             cand = [
#                 (tid, tr)
#                 for tid, tr in self._tracks.items()
#                 if tr.cls == cls and (tr.state is TState.CONFIRMED or not only_confirmed)
#             ]
#             if not cand or not bbs:
#                 continue

#             # cost = blend(IoU, appearance)
#             T, D = len(cand), len(bbs)
#             cost = [[1.0 for _ in range(D)] for _ in range(T)]
#             for ti, (_, tr) in enumerate(cand):
#                 for dj, bb in enumerate(bbs):
#                     iou = _iou(tr.bbox, bb)
#                     if iou < self.iou_thresh:
#                         cost[ti][dj] = 1e6
#                         continue
#                     if tr.feat is not None and self._frame is not None and _HAS_CV:
#                         df = self._feat(self._frame, bb)
#                         ad = self._dist(tr.feat, df) if df is not None else 1.0
#                         ad = max(0.0, min(1.0, ad))
#                         c = self.appearance_a * (1.0 - iou) + (1.0 - self.appearance_a) * ad
#                     else:
#                         c = (1.0 - iou)
#                     cost[ti][dj] = c

#             # greedy bipartite matching
#             used_t = set()
#             while True:
#                 best = (None, None, 1e9)
#                 for ti in range(len(cand)):
#                     if ti in used_t:
#                         continue
#                     for dj in range(len(bbs)):
#                         jg = js[dj]
#                         if jg in used_det:
#                             continue
#                         c = cost[ti][dj]
#                         if c < best[2]:
#                             best = (ti, dj, c)
#                 ti, dj, c = best
#                 if ti is None or c >= 1e5:
#                     break
#                 tid = cand[ti][0]
#                 jg = js[dj]
#                 matches.add((tid, jg))
#                 used_det.add(jg)
#                 used_t.add(ti)

#         return matches

#     # appearance helpers
#     def _feat(self, frame, bb):
#         if not _HAS_CV:
#             return None
#         x1, y1, x2, y2 = bb
#         h, w = frame.shape[:2]
#         x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
#         y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
#         if x2 <= x1 or y2 <= y1:
#             return None
#         crop = frame[y1:y2, x1:x2]
#         hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#         hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
#         hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
#         import numpy as np
#         feat = np.concatenate([hist_h, hist_s]).astype(np.float32)
#         n = np.linalg.norm(feat) + 1e-6
#         feat /= n
#         return feat

#     def _dist(self, f1, f2) -> float:
#         if f1 is None or f2 is None:
#             return 1.0
#         import numpy as np
#         return float(1.0 - float((f1 @ f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-6)))
# tiny adapter: keep your Detection/Track types unchanged
import numpy as np
from boxmot import ByteTrack  # or BoTSORT, DeepOCSORT, StrongSORT

class BoxMOTWrapper:
    def __init__(self, method="bytetrack", **kwargs):
        self.trk = ByteTrack(**kwargs)  # swap class for other methods

    def update(self, dets, frame):
        # dets: List[(cls:str, conf:float, (x1,y1,x2,y2))]
        if not dets:
            # most BoxMOT trackers accept empty input; they will age-out/predict
            out = self.trk.update(np.empty((0,6), dtype=float), frame)  # shape Mx(x1,y1,x2,y2,conf,cls)
            return []
        boxes = np.array([d[2] for d in dets], dtype=float)
        confs = np.array([d[1] for d in dets], dtype=float)[:, None]
        # map your class names to ints as needed; or 0 for single-class
        clss  = np.zeros((len(dets), 1), dtype=float)
        detections = np.concatenate([boxes, confs, clss], axis=1)
        # returns Mx(x1,y1,x2,y2,id,conf,cls,ind)
        tracks = self.trk.update(detections, frame)
        return [
            Track(track_id=int(t[4]), cls=str(int(t[6])), conf=float(t[5]),
                  bbox=(int(t[0]), int(t[1]), int(t[2]), int(t[3])))
            for t in tracks
        ]
