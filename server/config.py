"""Environment-driven configuration for the SenseVoice ASR service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 16000


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "")
    return value.strip() or default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # fail fast: a typo here is a deployment bug
        raise SystemExit(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Config:
    """Resolved settings. Built once at import from the container env."""

    model_dir: Path
    model_file: Path
    tokens_file: Path
    language: str
    num_threads: int
    pool_size: int
    max_upload_bytes: int
    segment_ms: int
    noise_db: int
    host: str
    port: int

    @classmethod
    def from_env(cls) -> "Config":
        model_dir = Path(_env("ASR_MODEL_DIR", "/models/sense-voice"))
        return cls(
            model_dir=model_dir,
            model_file=model_dir / _env("ASR_MODEL_FILE", "model.int8.onnx"),
            tokens_file=model_dir / _env("ASR_TOKENS_FILE", "tokens.txt"),
            # SenseVoice reads its language as a constructor argument, not per
            # request, so it is fixed for the container's lifetime. "auto"
            # covers zh/en/ja/ko/yue detection.
            language=_env("ASR_LANGUAGE", "auto"),
            num_threads=_env_int("ASR_NUM_THREADS", 2),
            pool_size=max(1, _env_int("ASR_POOL_SIZE", 1)),
            max_upload_bytes=_env_int("ASR_MAX_UPLOAD_MB", 512) * 1024 * 1024,
            # Chunks are cut at this boundary so one recognizer call never holds
            # more than half a minute of audio; long videos are stitched.
            segment_ms=_env_int("ASR_SEGMENT_MS", 30000),
            # ffmpeg silencedetect floor: audio under this level counts as
            # silence and is never sent to the recognizer.
            noise_db=_env_int("ASR_NOISE_DB", -35),
            host=_env("ASR_HOST", "0.0.0.0"),
            port=_env_int("ASR_PORT", 8080),
        )


config = Config.from_env()
