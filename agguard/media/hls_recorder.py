# agguard/media/hls_recorder.py
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

import numpy as np  # for blank/held frames

_CT = {
    ".m3u8": "application/vnd.apple.mpegurl",
    ".ts": "video/MP2T",
    ".m4s": "video/mp4",
    ".mp4": "video/mp4",
}

@dataclass
class HlsConfig:
    # Fixed cadence; 4–6 fps works well for security UIs
    fps: int = 5
    # 1.0s segments: good LIVE smoothness + 1s DVR seeks
    segment_time: float = 1.0
    # Event playlist grows; proxy trims for LIVE
    list_size: int = 240
    # Stick to TS for VLC; switch True for CMAF if wanted
    use_cmaf: bool = False
    # x264 knobs
    preset: str = "veryfast"
    crf: int = 23
    # one GOP per segment => clean random access
    gop_segments: int = 1
    # S3 mirror cadence
    upload_interval_sec: float = 0.02
    # Optional downscale (keeps aspect)
    target_width: Optional[int] = None
    # Some players behave better with silent audio
    add_silent_audio: bool = True


@dataclass
class HlsRecorder:
    s3: any
    bucket: str
    prefix: str
    cfg: HlsConfig = field(default_factory=HlsConfig)

    _tmpdir: Optional[str] = None
    _proc: Optional[subprocess.Popen] = None

    # Uploader state
    _sync_thread: Optional[threading.Thread] = None
    _stop_evt: threading.Event = field(default_factory=threading.Event)
    _uploaded_keys: set = field(default_factory=set)
    _ready_evt: threading.Event = field(default_factory=threading.Event)

    # Pacing/feeder state
    _feeder_thread: Optional[threading.Thread] = None
    _feeder_stop: threading.Event = field(default_factory=threading.Event)
    _last_frame: Optional[np.ndarray] = None
    _last_frame_lock: threading.Lock = field(default_factory=threading.Lock)
    _frame_shape: Optional[Tuple[int, int]] = None
    _first_frame_evt: threading.Event = field(default_factory=threading.Event)

    # ────────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────────
    def start(self, frame_size: Tuple[int, int]) -> None:
        """
        Start ffmpeg HLS writer + background uploader + paced feeder.
        This variant waits for the FIRST call to write_bgr() before feeding,
        so we don't encode initial black frames.
        """
        H, W = frame_size
        self._frame_shape = (H, W)
        self._tmpdir = tempfile.mkdtemp(prefix="hls_")
        out = pathlib.Path(self._tmpdir)

        seg_ext = ".m4s" if self.cfg.use_cmaf else ".ts"
        seg_pattern = str(out / f"segment_%05d{seg_ext}")
        m3u8_path = str(out / "index.m3u8")

        fps = max(1, int(self.cfg.fps))
        seg_time = float(self.cfg.segment_time)
        g_exact = max(1, int(round(fps * seg_time * max(1, self.cfg.gop_segments))))

        vf_parts = []
        if self.cfg.target_width and W > self.cfg.target_width:
            vf_parts.append(f"scale={self.cfg.target_width}:-2")
        vf_parts.append("format=yuv420p")
        vf = ",".join(vf_parts)

        cmd = [
            "ffmpeg",
            "-loglevel", "warning",
            # raw BGR frames on stdin; feeder will pace these
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s:v", f"{W}x{H}",
            "-r", str(fps),
            "-i", "pipe:0",
        ]

        # Optional silent audio (improves compatibility)
        if self.cfg.add_silent_audio:
            cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
            map_args = ["-map", "0:v:0", "-map", "1:a:0"]
        else:
            map_args = ["-map", "0:v:0"]

        cmd += [
            *map_args,
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", self.cfg.preset,
            "-crf", str(self.cfg.crf),
            "-profile:v", "main",
            "-level:v", "4.1",
            "-tune", "zerolatency",
            "-g", str(g_exact),
            "-keyint_min", str(g_exact),
            "-sc_threshold", "0",
            "-force_key_frames", f"expr:gte(t,n_forced*{seg_time})",
        ]

        if not self.cfg.use_cmaf:
            # Make TS segments self-contained
            cmd += ["-mpegts_flags", "resend_headers+initial_discontinuity"]

        if self.cfg.add_silent_audio:
            cmd += ["-c:a", "aac", "-ar", "48000", "-b:a", "128k"]

        # HLS muxing — EVENT playlist (grows); we’ll freeze it at finalize
        hls_flags = "append_list+program_date_time+independent_segments+temp_file+split_by_time"
        cmd += [
            "-f", "hls",
            "-hls_time", str(seg_time),
            "-hls_list_size", str(self.cfg.list_size),
            "-hls_flags", hls_flags,
            "-hls_segment_filename", seg_pattern,
            "-hls_playlist_type", "event",
            "-mpegts_flags", "resend_headers+initial_discontinuity",
            "-movflags", "faststart",
            m3u8_path,
        ]

        if self.cfg.use_cmaf:
            cmd += [
                "-hls_segment_type", "fmp4",
                "-hls_fmp4_init_filename", "init.mp4",
                "-hls_part_size", "0.2",
            ]

        # Launch ffmpeg
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, cwd=self._tmpdir)

        # We have not received any real frame yet
        self._first_frame_evt.clear()

        # Start uploader (mirrors local files to S3 fast)
        self._stop_evt.clear()
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()

        # Start paced feeder
        self._feeder_stop.clear()
        self._feeder_thread = threading.Thread(target=self._feeder_loop, daemon=True)
        self._feeder_thread.start()

    def write_bgr(self, frame_bgr) -> None:
        """Update the latest processed frame; feeder thread handles pacing."""
        if frame_bgr is None:
            return
        with self._last_frame_lock:
            self._last_frame = np.ascontiguousarray(frame_bgr)
        # If the feeder was waiting for the first frame, release it immediately
        if not self._first_frame_evt.is_set():
            self._first_frame_evt.set()

    # def finalize_to_mp4(self) -> str:
    #     """Stop feeder + ffmpeg + uploader; remux playlist to MP4; upload + cleanup."""
    #     # ─── Stop feeder + ffmpeg cleanly ───────────────────────────────
    #     self._feeder_stop.set()
    #     if self._feeder_thread:
    #         try:
    #             self._feeder_thread.join(timeout=5)
    #         except Exception:
    #             pass
    #         self._feeder_thread = None

    #     if self._proc:
    #         try:
    #             if self._proc.stdin:
    #                 try:
    #                     self._proc.stdin.close()
    #                 except Exception:
    #                     pass
    #         finally:
    #             try:
    #                 self._proc.wait(timeout=15)
    #             except Exception:
    #                 pass
    #         self._proc = None

    #     # ─── Stop uploader ─────────────────────────────────────────────
    #     self._stop_evt.set()
    #     if self._sync_thread:
    #         try:
    #             self._sync_thread.join(timeout=10)
    #         except Exception:
    #             pass
    #         self._sync_thread = None

    #     tmp = pathlib.Path(self._tmpdir or ".")
    #     out_mp4 = tmp / "final.mp4"

    #     # ─── Collect all local segments ────────────────────────────────
    #     segs = sorted([p for p in tmp.iterdir() if p.suffix.lower() in (".ts", ".m4s")])
    #     if not segs:
    #         self._cleanup_local()
    #         return f"{self.prefix}/final.mp4"

    #     # ─── Build concat list for ffmpeg ──────────────────────────────
    #     list_txt = tmp / "list.txt"
    #     with list_txt.open("w", encoding="utf-8") as f:
    #         for p in segs:
    #             f.write(f"file '{p.name}'\n")

    #     # ─── Convert to MP4 and remove duplicate frames ────────────────
    #     try:
    #         subprocess.run(
    #             [
    #                 "ffmpeg",
    #                 "-y",
    #                 "-loglevel", "warning",
    #                 "-f", "concat",
    #                 "-safe", "0",
    #                 "-i", str(list_txt),
    #                 # Apply duplicate-frame removal
    #                 "-vf", "mpdecimate=hi=64*15:lo=64*5:frac=0.1,setpts=N/FRAME_RATE/TB",
    #                 "-c:v", "libx264",
    #                 "-preset", "veryfast",
    #                 "-crf", "23",
    #                 "-c:a", "aac",
    #                 "-b:a", "128k",
    #                 "-movflags", "+faststart",
    #                 str(out_mp4),
    #             ],
    #             check=True,
    #             cwd=str(tmp),
    #         )
    #     except subprocess.CalledProcessError as e:
    #         print(f"[HlsRecorder] finalize_to_mp4 failed: {e}")
    #         self._cleanup_local()
    #         return f"{self.prefix}/final.mp4"

    #     # ─── Upload and cleanup ────────────────────────────────────────
    #     mp4_key = f"{self.prefix}/final.mp4"
    #     try:
    #         self.s3.put_file(self.bucket, mp4_key, str(out_mp4), content_type=_CT[".mp4"])
    #         for k in list(self._uploaded_keys):
    #             if not k.endswith("/final.mp4"):
    #                 try:
    #                     self.s3.delete_object(self.bucket, k)
    #                 except Exception:
    #                     pass
    #     except Exception:
    #         pass

    #     self._cleanup_local()
    #     return mp4_key










    # ────────────────────────────────────────────────────────────────────────
    # Internals
    # ────────────────────────────────────────────────────────────────────────
    def _feeder_loop(self):
        """Write frames into ffmpeg stdin at a fixed cadence; hold last frame when idle."""
        fps = max(1, int(self.cfg.fps))
        period = 1.0 / float(fps)

        # Wait for the first real frame to avoid initial black.
        # Since your aggregator calls write_bgr() immediately after start(),
        # this should trigger right away.
        self._first_frame_evt.wait(timeout=3.0)

        # If absolutely nothing arrived, we can still fall back to blank.
        blank = None
        if self._frame_shape:
            H, W = self._frame_shape
            blank = np.zeros((H, W, 3), dtype=np.uint8)

        while not self._feeder_stop.is_set():
            t0 = time.perf_counter()
            with self._last_frame_lock:
                frame = self._last_frame
            if frame is None:
                frame = blank
            if frame is not None and self._proc and self._proc.poll() is None:
                try:
                    self._proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    self._feeder_stop.set()
                    break
                except Exception:
                    pass
            dt = time.perf_counter() - t0
            time.sleep(max(0.0, period - dt))
    

    

    def _parse_tail_refs(self, m3u8_path: pathlib.Path) -> list[str]:
        """
        Return the list of segment filenames (URIs) in order, last item is the freshest.
        Works for both TS and CMAF (fMP4) playlists produced by ffmpeg.
        """
        try:
            txt = m3u8_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []
        uris = []
        for line in txt.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # keep raw line (local filename)
            uris.append(os.path.basename(s))
        return uris
    def _make_publishable_index(self, m3u8_path: pathlib.Path, exist_names: set[str]) -> Optional[pathlib.Path]:
        """
        Create a temp 'index.publish.m3u8' that is identical to the local m3u8
        but with any trailing (#EXTINF, URI) pairs removed if their URI file
        does not exist in 'exist_names'. Keeps headers & MAP intact.
        Returns path to the temp publishable file, or None if nothing to publish yet.
        """
        try:
            lines = m3u8_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return None

        headers: list[str] = []
        body_pairs: list[tuple[str, str]] = []  # (extinf, uri)
        pending_extinf: Optional[str] = None
        map_line: Optional[str] = None

        for raw in lines:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("#"):
                if s.startswith("#EXTINF:"):
                    pending_extinf = s
                elif s.startswith("#EXT-X-MAP:"):
                    map_line = s
                elif s.startswith("#EXT-X-ENDLIST"):
                    # drop ENDLIST for live/event publishing
                    continue
                else:
                    headers.append(s)
            else:
                # media URI
                uri = os.path.basename(s)
                if pending_extinf is None:
                    # defensive: if EXTINF missing, synthesize a 1s tag
                    pending_extinf = "#EXTINF:1.000,"
                body_pairs.append((pending_extinf, uri))
                pending_extinf = None

        # Trim from the tail until last URI exists
        while body_pairs and body_pairs[-1][1] not in exist_names:
            body_pairs.pop()

        if not body_pairs:
            return None

        out: list[str] = ["#EXTM3U"]
        # normalize a few important headers; keep original others
        have_version = any(h.startswith("#EXT-X-VERSION:") for h in headers)
        have_indep   = any(h.startswith("#EXT-X-INDEPENDENT-SEGMENTS") for h in headers)
        have_target  = any(h.startswith("#EXT-X-TARGETDURATION:") for h in headers)

        if not have_version:
            out.append("#EXT-X-VERSION:6")
        for h in headers:
            out.append(h)
        if not have_indep:
            out.append("#EXT-X-INDEPENDENT-SEGMENTS")
        if not have_target:
            out.append("#EXT-X-TARGETDURATION:1")

        if map_line:
            out.append(map_line)

        for extinf, uri in body_pairs:
            out.append(extinf)
            out.append(uri)

        tmp = m3u8_path.parent / "index.publish.m3u8"
        tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
        return tmp

    def _sync_loop(self) -> None:
        """Continuously mirror new/changed local HLS files to S3 in a safe order.

        Order:
          1) init.mp4 (if CMAF)
          2) segments (.ts or .m4s)
          3) audio files (rare)
          4) index.m3u8 (LAST, and only after newest referenced segment is uploaded)
        """
        if not self._tmpdir:
            return
        root = pathlib.Path(self._tmpdir)
        seen: dict[str, tuple[int, int]] = {}
        have_index = False
        have_init = not self.cfg.use_cmaf

        interval = float(max(0.005, self.cfg.upload_interval_sec))
        while not self._stop_evt.is_set():
            try:
                # Snapshot current files
                files = [p for p in root.iterdir() if p.is_file()]
                # Partition by type
                init_files = [p for p in files if p.name == "init.mp4"]
                seg_files_ts = [p for p in files if p.suffix.lower() == ".ts"]
                seg_files_m4s = [p for p in files if p.suffix.lower() == ".m4s"]
                audio_files = [p for p in files if p.suffix.lower() in (".aac", ".mp3", ".wav")]
                m3u8_files = [p for p in files if p.name == "index.m3u8"]

                # 1) init.mp4 first (CMAF)
                for p in init_files:
                    stat = p.stat()
                    sig = (stat.st_size, getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))
                    key = f"{self.prefix}/{p.name}"
                    if seen.get(key) != sig:
                        self.s3.put_file(self.bucket, key, str(p), _CT.get(".mp4", "video/mp4"))
                        seen[key] = sig
                        self._uploaded_keys.add(key)
                        have_init = True

                # 2) segments, sorted by name (sequence)
                segs = sorted(seg_files_ts + seg_files_m4s, key=lambda x: x.name)
                for p in segs:
                    stat = p.stat()
                    sig = (stat.st_size, getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))
                    key = f"{self.prefix}/{p.name}"
                    if seen.get(key) == sig:
                        continue
                    # Because ffmpeg uses -hls_flags temp_file, a segment appears atomically
                    # when fully written; we can upload immediately.
                    self.s3.put_file(self.bucket, key, str(p), _CT.get(p.suffix.lower(), "application/octet-stream"))
                    seen[key] = sig
                    self._uploaded_keys.add(key)

                # 3) any audio
                for p in audio_files:
                    stat = p.stat()
                    sig = (stat.st_size, getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))
                    key = f"{self.prefix}/{p.name}"
                    if seen.get(key) != sig:
                        self.s3.put_file(self.bucket, key, str(p), _CT.get(p.suffix.lower(), "application/octet-stream"))
                        seen[key] = sig
                        self._uploaded_keys.add(key)

                # 4) finally index.m3u8 — publish a trimmed version so it never points ahead
                for p in m3u8_files:
                    stat = p.stat()
                    if stat.st_size <= 0:
                        continue

                    # Build set of locally existing segment names (what we can publish)
                    existing = {q.name for q in segs}  # segs you already built in step 2
                    if self.cfg.use_cmaf:
                        existing |= {q.name for q in init_files}  # init.mp4 is allowed in MAP

                    publishable = self._make_publishable_index(p, existing)
                    if not publishable:
                        # Nothing safe to publish yet; try next loop
                        continue

                    pub_stat = publishable.stat()
                    sig = (pub_stat.st_size, getattr(pub_stat, "st_mtime_ns", int(pub_stat.st_mtime * 1e9)))
                    key = f"{self.prefix}/index.m3u8"
                    if seen.get(key) != sig:
                        self.s3.put_file(self.bucket, key, str(publishable), _CT[".m3u8"])
                        seen[key] = sig
                        self._uploaded_keys.add(key)
                        have_index = True


                if have_index and have_init and not self._ready_evt.is_set():
                    self._ready_evt.set()

            except Exception:
                # best-effort sync
                pass

            time.sleep(interval)


    def wait_ready(self, timeout: float = 6.0) -> bool:
        """Block until the playlist (+ init for CMAF) has been uploaded once (or timeout)."""
        return self._ready_evt.wait(timeout)

    def _cleanup_local(self) -> None:
        try:
            if self._tmpdir and os.path.isdir(self._tmpdir):
                shutil.rmtree(self._tmpdir, ignore_errors=True)
        finally:
            self._tmpdir = None

    # ────────────────────────────────────────────────────────────────────────
    # Playlist freezing / reconstruction for finalize
    # ────────────────────────────────────────────────────────────────────────
    def _build_temp_playlist(
        self,
        workdir: pathlib.Path,
        segs_ts: list[pathlib.Path],
        segs_m4s: list[pathlib.Path],
        have_cmaf: bool,
    ) -> Optional[pathlib.Path]:
        """Create a local m3u8 referencing all segments we have."""
        if have_cmaf:
            if not segs_m4s:
                return None
            td = 1
            p = workdir / "reconstructed.m3u8"
            with p.open("w", encoding="utf-8") as f:
                f.write("#EXTM3U\n#EXT-X-VERSION:6\n#EXT-X-PLAYLIST-TYPE:EVENT\n")
                f.write("#EXT-X-INDEPENDENT-SEGMENTS\n")
                f.write(f"#EXT-X-TARGETDURATION:{td}\n")
                if (workdir / "init.mp4").exists():
                    f.write('#EXT-X-MAP:URI="init.mp4"\n')
                for s in segs_m4s:
                    f.write("#EXTINF:1.000,\n")
                    f.write(f"{s.name}\n")
            return p
        else:
            if not segs_ts:
                return None
            td = 1
            p = workdir / "reconstructed.m3u8"
            with p.open("w", encoding="utf-8") as f:
                f.write("#EXTM3U\n#EXT-X-VERSION:6\n#EXT-X-PLAYLIST-TYPE:EVENT\n")
                f.write("#EXT-X-INDEPENDENT-SEGMENTS\n")
                f.write(f"#EXT-X-TARGETDURATION:{td}\n")
                for s in segs_ts:
                    f.write("#EXTINF:1.000,\n")
                    f.write(f"{s.name}\n")
            return p

    def _freeze_local_playlist(self, m3u8_path: pathlib.Path, workdir: pathlib.Path) -> pathlib.Path:
        """
        Produce a 'finalize.m3u8' that:
          - keeps essential headers (normalized),
          - rewrites URIs to local filenames,
          - appends #EXT-X-ENDLIST,
          - avoids HTTP or proxy (/seg?u=) paths so ffmpeg won't poll.
        """
        def _maybe_localize_uri(uri: str) -> str:
            m = re.search(r'[?&]u=([^&]+)', uri)
            inner = m.group(1) if m else uri
            u = urlparse(inner)
            candidate = os.path.basename(u.path if u.scheme in ("http", "https") else inner)
            return candidate  # ffmpeg will read from cwd

        lines = m3u8_path.read_text(encoding="utf-8", errors="ignore").splitlines()

        version = 6
        target = 1
        have_independent = False

        body: list[str] = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                if s.startswith("#EXT-X-VERSION:"):
                    version = 6  # normalize
                elif s.startswith("#EXT-X-TARGETDURATION:"):
                    try:
                        target = max(target, int(float(s.split(":", 1)[1])))
                    except Exception:
                        pass
                elif s.startswith("#EXT-X-INDEPENDENT-SEGMENTS"):
                    have_independent = True
                elif s.startswith("#EXT-X-MAP:"):
                    m = re.search(r'URI="([^"]+)"', s)
                    if m:
                        body.append(f'#EXT-X-MAP:URI="{_maybe_localize_uri(m.group(1))}"')
                elif s.startswith("#EXTINF:"):
                    body.append(s)
                # ignore PROGRAM-DATE-TIME, PLAYLIST-TYPE, MEDIA-SEQUENCE, etc.
                continue
            # media URI
            body.append(_maybe_localize_uri(s))

        frozen = ["#EXTM3U", f"#EXT-X-VERSION:{version}"]
        if not have_independent:
            frozen.append("#EXT-X-INDEPENDENT-SEGMENTS")
        frozen.append(f"#EXT-X-TARGETDURATION:{max(1, int(target))}")
        frozen.extend(body)
        frozen.append("#EXT-X-ENDLIST")

        out_path = workdir / "finalize.m3u8"
        out_path.write_text("\n".join(frozen) + "\n", encoding="utf-8")
        return out_path
