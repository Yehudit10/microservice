from __future__ import annotations

import os
import time
import shutil
import tempfile
import subprocess
import pathlib
import threading
import re
from dataclasses import dataclass, field
from typing import Optional, Tuple
from urllib.parse import urlparse
from agguard.media.hls_recorder import HlsConfig

import numpy as np  # for blank/held frames
@dataclass
class Mp4Recorder:
    """Simpler recorder used to build final MP4 only from new frames."""
    s3: any
    bucket: str
    prefix: str
    cfg: HlsConfig = field(default_factory=HlsConfig)
    _tmpdir: Optional[str] = None
    _proc: Optional[subprocess.Popen] = None

    def start(self, frame_size: Tuple[int, int]) -> None:
        H, W = frame_size
        self._tmpdir = tempfile.mkdtemp(prefix="mp4_")
        out_path = pathlib.Path(self._tmpdir) / "frames_pipe.mp4"

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "warning",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s:v", f"{W}x{H}",
            "-r", str(self.cfg.fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-movflags", "+faststart",
            str(out_path),
        ]

        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, cwd=self._tmpdir)

    def write_bgr(self, frame_bgr) -> None:
        if self._proc and self._proc.poll() is None and frame_bgr is not None:
            try:
                self._proc.stdin.write(frame_bgr.tobytes())
            except BrokenPipeError:
                pass

    def finalize(self) -> str:
        if self._proc:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.wait(timeout=10)
            except Exception:
                pass
        tmp = pathlib.Path(self._tmpdir or ".")
        out_mp4 = tmp / "final.mp4"

        # Rename if needed (ffmpeg already wrote to frames_pipe.mp4)
        pipe_out = tmp / "frames_pipe.mp4"
        if pipe_out.exists():
            pipe_out.rename(out_mp4)

        mp4_key = f"{self.prefix}/final.mp4"
        try:
            self.s3.put_file(self.bucket, mp4_key, str(out_mp4), content_type="video/mp4")
        except Exception:
            pass

        shutil.rmtree(self._tmpdir, ignore_errors=True)
        return mp4_key
