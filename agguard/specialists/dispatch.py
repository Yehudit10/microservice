from __future__ import annotations
from typing import Dict, List, Callable, Any
import importlib
import logging

from agguard.core.types import Detection, BBox

log = logging.getLogger(__name__)

# (frame_bgr, boxes, subjects=None) -> predictions
Specialist = Callable[[object, List[BBox], List[str] | None], object]


def _load(dotted_path: str):
    mod, obj = dotted_path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), obj)


class ClassDispatch:
    """
    Map class name -> [specialist callables].
    Each class can have multiple specialists (gRPC or local).
    The dispatcher will call them all and unify their predictions.
    """

    def __init__(self, specs_cfg: List[dict]):
        self._by_class: Dict[str, List[Specialist]] = {}

        for spec in specs_cfg or []:
            cls_name = str(spec["for_class"]).lower()

            if "grpc" in spec:
                grpc_cfg = spec["grpc"] or {}
                kind = str(grpc_cfg.get("kind", "mask")).lower().strip()
                address = grpc_cfg.get("address", "127.0.0.1:50061")
                timeout = float(grpc_cfg.get("timeout_sec", 1.5))
                jpeg_q = int(grpc_cfg.get("jpeg_quality", 85))

                if kind == "anomalies":
                    from agguard.specialists.clients.anomalies import GrpcClipClassifierClient
                    client = GrpcClipClassifierClient(
                        address=address,
                        timeout_sec=timeout,
                        jpeg_quality=jpeg_q,
                    )
                    fn: Specialist = lambda frame, boxes, subjects=None, _c=client, _cls=cls_name: \
                        _c.classify(frame, boxes, subjects or ([_cls]))
                else:
                    from agguard.specialists.clients.mask import GrpcMaskClassifierClient
                    client = GrpcMaskClassifierClient(
                        address=address,
                        model_name="mask",
                        timeout_sec=timeout,
                        jpeg_quality=jpeg_q,
                    )
                    fn: Specialist = lambda frame, boxes, subjects=None, _c=client: \
                        _c.classify(frame, boxes)

                self._by_class.setdefault(cls_name, []).append(fn)
                log.info(
                    "Registered gRPC specialist for class '%s' -> %s (%s)",
                    cls_name, address, kind
                )

            else:
                dotted = spec["dotted_path"]
                ctor = _load(dotted)
                inst = ctor(**(spec.get("kwargs") or {}))

                def _call(frame, boxes, subjects=None, _inst=inst):
                    try:
                        return _inst.classify(frame, boxes, subjects=subjects)
                    except TypeError:
                        return _inst.classify(frame, boxes)

                self._by_class.setdefault(cls_name, []).append(_call)
                log.info(
                    "Registered local specialist for class '%s' -> %s",
                    cls_name, dotted
                )

    def run(self, frame_bgr, dets: List[Detection]) -> Dict[str, object]:
        """
        For each detection class:
        - Group boxes by class
        - Call all specialists registered for that class
        - Merge all their outputs into a unified list
        """
        buckets: Dict[str, List[BBox]] = {}
        for d in dets:
            key = str(d.cls).lower()
            if key in self._by_class:
                buckets.setdefault(key, []).append(d.bbox)

        outputs: Dict[str, Any] = {}
        for key, boxes in buckets.items():
            merged = []
            subjects = None#[key] * len(boxes)

            for fn in self._by_class.get(key, []):
                try:
                    preds = fn(frame_bgr, boxes, subjects) or []
                    if isinstance(preds, list):
                        merged.extend(preds)
                    else:
                        merged.append(preds)
                except Exception as e:
                    log.exception("Specialist for '%s' failed: %s", key, e)

            # Unified merged predictions for that class
            outputs[key] = merged

        return outputs
