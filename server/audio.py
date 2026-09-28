"""Audio decode and segmentation.

Everything the recognizer sees is 16 kHz mono float32. Decoding goes through
ffmpeg (the container ships it) because uploads arrive as mp3/m4a/aac/webm/etc
and the Python decoders are unreliable on those containers.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

import numpy as np

from .config import SAMPLE_RATE


class AudioError(RuntimeError):
    """The upload could not be decoded."""


@dataclass
class Clip:
    """One recognizer job: a slice of the source audio, in seconds."""

    start: float
    end: float
    samples: np.ndarray


def _run_ffmpeg(args: list[str], *, capture_stdout: bool) -> bytes:
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=1800,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        raise AudioError(f"ffmpeg failed: {tail[-1] if tail else 'no stderr'}")
    return proc.stdout or b""


def probe_duration(path: str) -> float:
    """Length in seconds, or 0.0 when ffprobe cannot tell."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    if proc.returncode != 0:
        return 0.0
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return 0.0


def decode(path: str) -> np.ndarray:
    """Decode any container ffmpeg understands to 16 kHz mono float32.

    The decoder streams raw float samples (`-f f32le`) instead of a WAV file:
    `frombuffer` then wraps the buffer zero-copy, so a 1-hour upload peaks at
    one 230 MB array rather than PCM-plus-decoded-copy.
    """
    raw = _run_ffmpeg(
        [
            "ffmpeg", "-nostdin", "-v", "error",
            "-i", path,
            "-vn", "-sn", "-dn",
            "-acodec", "pcm_f32le",
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-f", "f32le", "-",
        ],
        capture_stdout=True,
    )
    if not raw:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(raw, dtype="<f4")


def silence_bounds(path: str, *, noise_db: int = -35, min_silence: float = 0.6) -> list[tuple[float, float]]:
    """Speech spans from ffmpeg's silencedetect, as (start, end) in seconds.

    This is the cheap stand-in for a VAD model: it drops music/room-tone
    stretches so the CPU never transcribes them. For audio where it detects no
    silence at all the caller falls back to fixed windows.
    """
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "info",
            "-i", path,
            "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
            "-f", "null", "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=1800,
    )
    events: list[tuple[str, float]] = []
    for line in (proc.stderr or b"").decode("utf-8", "replace").splitlines():
        if "silence_start:" in line:
            events.append(("start", float(line.rsplit("silence_start:", 1)[1].strip())))
        elif "silence_end:" in line:
            value = line.rsplit("silence_end:", 1)[1].split("|")[0].strip()
            events.append(("end", float(value)))

    spans: list[tuple[float, float]] = []
    cursor: float | None = None
    for kind, at in events:
        if kind == "start" and cursor is None:
            cursor = at
        elif kind == "end" and cursor is not None:
            if at > cursor:
                spans.append((cursor, at))
            cursor = None
    if cursor is not None:
        spans.append((cursor, float("inf")))
    return spans


def _windows(duration: float, segment_s: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        out.append((start, min(start + segment_s, duration)))
        start += segment_s
    return out


def plan_clips(path: str, samples: np.ndarray, segment_ms: int, noise_db: int = -35) -> list[Clip]:
    """Split the decoded audio into recognizer-sized clips.

    Prefers speech spans when silencedetect found any; otherwise cuts fixed
    windows so arbitrarily long audio still fits one recognizer call at a time.
    """
    duration = len(samples) / SAMPLE_RATE
    if duration <= 0:
        return []
    segment_s = max(1.0, segment_ms / 1000.0)

    spans = silence_bounds(path, noise_db=noise_db)
    if not spans:
        spans = [(0.0, duration)]

    clips: list[Clip] = []
    for span_start, span_end in spans:
        end = duration if span_end == float("inf") else min(span_end, duration)
        # Long speech spans are cut further; a single recognizer call should
        # not hold minutes of audio on a low-power CPU.
        for win_start, win_end in _windows(max(0.0, end - span_start), segment_s):
            start = span_start + win_start
            stop = span_start + win_end
            if stop - start < 0.2:  # drop slivers the VAD boundary can leave
                continue
            clips.append(
                Clip(
                    start=start,
                    end=stop,
                    samples=samples[int(start * SAMPLE_RATE) : int(stop * SAMPLE_RATE)],
                )
            )
    return clips
