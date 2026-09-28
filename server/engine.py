"""SenseVoice recognizer, wrapped for sequential clip-by-clip use.

sherpa-onnx recognizers are not thread-safe, so a small pool hands each
concurrent request its own recognizer. On a 4-core NAS the pool stays at 1-2:
more workers than cores just adds contention.
"""

from __future__ import annotations

import queue
import re
import threading
from dataclasses import dataclass, field

import numpy as np
import sherpa_onnx

from .config import Config, SAMPLE_RATE

# SenseVoice emits inline markers like <|zh|><|NEUTRAL|><|Speech|><|woitn|>.
# The transcript keeps only the words.
_MARKER = re.compile(r"<\|[^|]*\|>")


def clean_text(raw: str) -> str:
    return _MARKER.sub("", raw).strip()


def marker_value(raw: str) -> str:
    """'<|zh|>' -> 'zh'; plain strings pass through untouched."""
    return raw.replace("<|", "").replace("|>", "").strip()


@dataclass
class ClipResult:
    text: str
    language: str = ""
    emotion: str = ""
    event: str = ""


@dataclass
class Recognizer:
    """One sherpa-onnx SenseVoice recognizer, guarded by its own lock."""

    inner: "sherpa_onnx.OfflineRecognizer"
    lock: threading.Lock = field(default_factory=threading.Lock)

    def recognize(self, samples: np.ndarray) -> ClipResult:
        with self.lock:
            stream = self.inner.create_stream()
            stream.accept_waveform(SAMPLE_RATE, samples)
            self.inner.decode_stream(stream)
            result = stream.result
        return ClipResult(
            text=clean_text(result.text or ""),
            language=marker_value(getattr(result, "lang", "") or ""),
            emotion=marker_value(getattr(result, "emotion", "") or ""),
            event=marker_value(getattr(result, "event", "") or ""),
        )


def _build(config: Config) -> "sherpa_onnx.OfflineRecognizer":
    if not config.model_file.exists():
        raise SystemExit(f"model not found: {config.model_file}")
    if not config.tokens_file.exists():
        raise SystemExit(f"tokens not found: {config.tokens_file}")
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(config.model_file),
        tokens=str(config.tokens_file),
        num_threads=config.num_threads,
        use_itn=True,
        language=config.language,
        debug=False,
        provider="cpu",
    )


class Pool:
    """A fixed set of recognizers plus a semaphore-like checkout queue."""

    def __init__(self, config: Config):
        self._free: queue.Queue[Recognizer] = queue.Queue()
        for _ in range(config.pool_size):
            self._free.put(Recognizer(inner=_build(config)))

    def checkout(self) -> Recognizer:
        return self._free.get()

    def release(self, recognizer: Recognizer) -> None:
        self._free.put(recognizer)
