# agguard/media/hls_recorder.py
from __future__ import annotations
import os, time, shutil, tempfile, subprocess, pathlib, threading
from dataclasses import dataclass, field
from typing import Optional, Tuple

_CT = {
    ".m3u8": "application/vnd.apple.mpegurl",
    ".ts":   "video/MP2T",
    ".m4s":  "video/mp4",
    ".mp4":  "video/mp4",
}

@dataclass
class HlsConfig:
    fps: int = 12
    segment_time: float = 3.0         # seconds (use 1.0–2.0 for snappier start)
    list_size: int = 20               # sliding window length
    use_cmaf: bool = False            # False => .ts; True => CMAF .m4s
    preset: str = "veryfast"
    crf: int = 23
    gop_segments: int = 1             # exact 1 GOP per segment for low-latency live
    upload_interval_sec: float = 0.25
    target_width: Optional[int] = None  # e.g. 1920 to downscale; None to keep source
    add_silent_audio: bool = True       # add silent AAC to maximize compatibility

@dataclass
class HlsRecorder:
    s3: any
    bucket: str
    prefix: str                        # e.g. "security/incidents/<cam>/<incident>"
    cfg: HlsConfig = field(default_factory=HlsConfig)

    _tmpdir: Optional[str] = None
    _proc: Optional[subprocess.Popen] = None
    _stdin_lock: threading.Lock = field(default_factory=threading.Lock)
    _sync_thread: Optional[threading.Thread] = None
    _stop_evt: threading.Event = field(default_factory=threading.Event)
    _uploaded_keys: set = field(default_factory=set)
    _ready_evt: threading.Event = field(default_factory=threading.Event)

    def start(self, frame_size: Tuple[int, int]) -> None:
        H, W = frame_size
        self._tmpdir = tempfile.mkdtemp(prefix="hls_")
        out = pathlib.Path(self._tmpdir)

        seg_ext = ".m4s" if self.cfg.use_cmaf else ".ts"
        seg_pattern = str(out / f"segment_%05d{seg_ext}")
        m3u8_path = str(out / "index.m3u8")

        fps = max(1, int(self.cfg.fps))
        # exactly 1 GOP per segment to align keyframes with segment boundaries
        g_exact = max(1, int(round(fps * float(self.cfg.segment_time) * max(1, self.cfg.gop_segments))))

        # Build a video filter (vf) chain: optional downscale + force yuv420p
        vf_parts = []
        if self.cfg.target_width and W > self.cfg.target_width:
            vf_parts.append(f"scale={self.cfg.target_width}:-2")
        vf_parts.append("format=yuv420p")
        vf = ",".join(vf_parts)

        cmd = [
            "ffmpeg", "-loglevel", "warning",

            # video input from raw BGR frames
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s:v", f"{W}x{H}", "-r", str(fps), "-i", "pipe:0",
        ]

        # Optional silent audio input (keeps players that expect audio happy)
        if self.cfg.add_silent_audio:
            cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
            map_args = ["-map", "0:v:0", "-map", "1:a:0"]
        else:
            map_args = ["-map", "0:v:0"]

        cmd += [
            *map_args,

            # Encoder settings — MAIN profile, yuv420p, stable GOP cadence
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", self.cfg.preset, "-crf", str(self.cfg.crf),
            "-profile:v", "main", "-level:v", "4.1",
            "-tune", "zerolatency",
            "-g", str(g_exact), "-keyint_min", str(g_exact),
            "-sc_threshold", "0",
        ]

        if self.cfg.add_silent_audio:
            cmd += ["-c:a", "aac", "-ar", "48000", "-b:a", "128k"]

        # HLS muxing: SLIDING list (not append). Write segments atomically.
        cmd += [
            "-f", "hls",
            "-hls_time", str(self.cfg.segment_time),
            "-hls_list_size", str(self.cfg.list_size),
            "-hls_flags", "delete_segments+program_date_time+independent_segments+temp_file",
            "-hls_segment_filename", seg_pattern,
        ]

        if self.cfg.use_cmaf:
            cmd += ["-hls_segment_type", "fmp4", "-hls_fmp4_init_filename", "init.mp4"]

        # moov front-load (for quicker progressive MP4s if/when remuxed)
        cmd += ["-movflags", "faststart", m3u8_path]

        # launch ffmpeg
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, cwd=self._tmpdir)
        # start uploader
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()

    def write_bgr(self, frame_bgr) -> None:
        if not self._proc or self._proc.poll() is not None:
            return
        with self._stdin_lock:
            try:
                self._proc.stdin.write(frame_bgr.tobytes())
            except BrokenPipeError:
                pass

    def finalize_to_mp4(self) -> str:
        # stop ffmpeg and the sync thread
        if self._proc:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=15)
            except Exception:
                pass
            self._proc = None

        self._stop_evt.set()
        if self._sync_thread:
            try:
                self._sync_thread.join(timeout=10)
            except Exception:
                pass

        tmp = pathlib.Path(self._tmpdir or ".")
        m3u8 = tmp / "index.m3u8"
        out_mp4 = tmp / "final.mp4"

        def _upload_and_cleanup(mp4_path: pathlib.Path) -> str:
            mp4_key = f"{self.prefix}/final.mp4"
            self.s3.put_file(self.bucket, mp4_key, str(mp4_path), content_type=_CT[".mp4"])
            # delete the HLS objects we uploaded
            try:
                if hasattr(self.s3, "delete_prefix"):
                    self.s3.delete_prefix(self.bucket, self.prefix)
                else:
                    for k in list(self._uploaded_keys):
                        try:
                            self.s3.delete_object(self.bucket, k)
                        except Exception:
                            pass
            except Exception:
                pass
            self._cleanup_local()
            return mp4_key

        # Preferred: remux from playlist (handles both .ts and CMAF)
        if m3u8.exists():
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-protocol_whitelist", "file,crypto,data,pipe",
                        "-i", str(m3u8),
                        "-fflags", "+genpts",
                        "-c", "copy",
                        "-bsf:a", "aac_adtstoasc",
                        "-movflags", "+faststart",
                        str(out_mp4),
                    ],
                    check=True, cwd=str(tmp)
                )
                return _upload_and_cleanup(out_mp4)
            except subprocess.CalledProcessError:
                pass  # fall back below

        # Fallback: concat demuxer on the segments
        segs = sorted([p for p in tmp.iterdir() if p.suffix.lower() in (".ts", ".m4s")])
        if not segs:
            self._cleanup_local()
            return f"{self.prefix}/final.mp4"

        list_txt = tmp / "list.txt"
        with list_txt.open("w", encoding="utf-8") as f:
            for p in segs:
                f.write(f"file '{p.name}'\n")

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", str(list_txt),
                "-fflags", "+genpts",
                "-c", "copy",
                "-bsf:a", "aac_adtstoasc",
                "-movflags", "+faststart",
                str(out_mp4),
            ],
            check=True, cwd=str(tmp)
        )
        return _upload_and_cleanup(out_mp4)

    # ---------- internals ----------

    def _sync_loop(self) -> None:
        """Continuously mirror new/changed local HLS files to S3 with high-res mtime."""
        if not self._tmpdir:
            return
        root = pathlib.Path(self._tmpdir)
        seen: dict[str, tuple[int, int]] = {}
        ready_set = False

        interval = max(0.05, float(self.cfg.upload_interval_sec or 0.25))
        while not self._stop_evt.is_set():
            try:
                for p in root.iterdir():
                    if not p.is_file():
                        continue
                    ext = p.suffix.lower()
                    if ext not in (".m3u8", ".ts", ".m4s", ".mp4", ".mp3", ".aac", ".wav"):
                        continue

                    stat = p.stat()
                    mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))
                    sig = (stat.st_size, mtime_ns)
                    key = f"{self.prefix}/{p.name}"
                    if seen.get(key) == sig:
                        continue

                    self.s3.put_file(self.bucket, key, str(p), _CT.get(ext, "application/octet-stream"))
                    seen[key] = sig
                    self._uploaded_keys.add(key)

                    if not ready_set and p.name == "index.m3u8" and stat.st_size > 0:
                        ready_set = True
                        self._ready_evt.set()
            except Exception:
                # best-effort sync
                pass

            time.sleep(interval)

    def wait_ready(self, timeout: float = 6.0) -> bool:
        """Block until index.m3u8 has been uploaded at least once (or timeout)."""
        return self._ready_evt.wait(timeout)

    def _cleanup_local(self) -> None:
        try:
            if self._tmpdir and os.path.isdir(self._tmpdir):
                shutil.rmtree(self._tmpdir, ignore_errors=True)
        finally:
            self._tmpdir = None
