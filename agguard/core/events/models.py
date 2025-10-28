from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional, Union
from datetime import datetime

Box = Tuple[int,int,int,int]

@dataclass
class Rule:
    name: str
    target_cls: str                          # e.g. "person"
    target_cls_id: Optional[Union[str,int]] = None  # e.g. 0, "0"
    match_classes: List[str] = field(default_factory=list)  # ← new field
    severity:  int = 3                
    min_conf: float = 0.6
    min_consec: int = 5
    cooldown: int = 12


@dataclass
class Incident:
    incident_id: str
    kind: str
    camera_id: Optional[str] = None
    started_ts: float = 0.0
    ended_ts: float = 0.0
    duration_sec: float = 0.0
    frame_start: int = 0
    frame_end: int = 0
    severity:  int = 1
    track_id: Optional[int] = None
    roi: Optional[List[Tuple[int,int]]] = None
    boxes: List[Box] = field(default_factory=list)
    confs: List[float] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    snapshot_path: Optional[str] = None
    artifacts: Dict[str,str] = field(default_factory=dict)
    meta: Dict[str, str|int|float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["started_iso"] = datetime.utcfromtimestamp(self.started_ts).isoformat()+"Z"
        d["ended_iso"]   = datetime.utcfromtimestamp(self.ended_ts).isoformat()+"Z"
        return d
