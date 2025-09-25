from __future__ import annotations
from typing import Dict, List, Callable
import importlib
import logging

from agguard.core.types import Detection, BBox

log = logging.getLogger(__name__)
Specialist = Callable[[object, List[BBox]], object]  # (frame_bgr, boxes) -> predictions

def _load(dotted_path: str):
    mod, obj = dotted_path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), obj)

class ClassDispatch:
    """Map class name -> callable(frame, boxes) using dotted paths from config."""

    def __init__(self, specs_cfg: List[dict]):
        self._by_class: Dict[str, Specialist] = {}
        for spec in specs_cfg or []:
            cls_name = spec["for_class"].lower()
            ctor = _load(spec["dotted_path"])
            inst = ctor(**(spec.get("kwargs") or {}))
            # Expect a `classify(frame, boxes)` method (like FaceMaskClassifier)
            fn: Specialist = lambda frame, boxes, _inst=inst: _inst.classify(frame, boxes)
            self._by_class[cls_name] = fn
            log.info("Registered specialist for class '%s' -> %s", cls_name, spec["dotted_path"])

    def run(self, frame_bgr, dets: List[Detection]) -> Dict[str, object]:
        buckets: Dict[str, List[BBox]] = {}
        for d in dets:
            key = d.cls.lower()
            if key in self._by_class:
                buckets.setdefault(key, []).append(d.bbox)
        outputs: Dict[str, object] = {}
        for key, boxes in buckets.items():
            try:
                outputs[key] = self._by_class[key](frame_bgr, boxes)
            except Exception as e:
                outputs[key] = {"error": str(e), "boxes": boxes}
                log.exception("Specialist for '%s' failed", key)
        return outputs
