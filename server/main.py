"""OpenAI-compatible transcription endpoint backed by local SenseVoice.

POST /v1/audio/transcriptions  (multipart: file, model, optional language)
  -> {"text", "segments": [{start, end, text}], "duration", "language"}

The shape is the one OpenAI defined and compatible servers copied, so WeKnora
reaches it through its generic OpenAI-compatible ASR client with no Go changes.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .audio import AudioError, decode, plan_clips, probe_duration
from .config import config
from .engine import Pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sensevoice-asr")

app = FastAPI(title="sensevoice-asr", version="0.1.0")
pool = Pool(config)

# The pristine copy of the upload is kept under this suffix; ffmpeg identifies
# the container from it.
_SUFFIX_FALLBACK = ".bin"


def _suffix(file_name: str) -> str:
    suffix = Path(file_name or "").suffix
    return suffix if suffix else _SUFFIX_FALLBACK


def _transcribe(path: str) -> tuple[str, list[dict], str]:
    """Decode, cut into clips, recognize each, and stitch the result."""
    samples = decode(path)
    if samples.size == 0:
        return "", [], ""

    clips = plan_clips(path, samples, config.segment_ms, config.noise_db)
    recognizer = pool.checkout()
    try:
        pieces: list[str] = []
        segments: list[dict] = []
        language = ""
        for i, clip in enumerate(clips, start=1):
            result = recognizer.recognize(clip.samples)
            if result.language:
                language = language or result.language
            if not result.text:
                continue
            pieces.append(result.text)
            segments.append({"start": clip.start, "end": clip.end, "text": result.text})
            if i % 25 == 0:
                logger.info("transcribed %d/%d clips", i, len(clips))
    finally:
        pool.release(recognizer)

    return "".join(pieces), segments, language


@app.post("/v1/audio/transcriptions")
def transcribe(
    file: UploadFile = File(...),
    model: str = Form("sensevoice"),
    language: str = Form(""),
    response_format: str = Form("json"),
) -> JSONResponse:
    # language and response_format are accepted for wire compatibility and
    # ignored: SenseVoice's language is fixed at container start, and segment
    # timestamps are always returned (a superset of json's {"text"}).
    _ = language, response_format, model
    if config.max_upload_bytes and file.size and file.size > config.max_upload_bytes:
        raise HTTPException(status_code=413, detail="audio exceeds the configured upload limit")

    limited = config.max_upload_bytes or 0
    written = 0
    tmp = tempfile.NamedTemporaryFile(suffix=_suffix(file.filename), delete=False)
    try:
        with tmp:
            while chunk := file.file.read(1 << 20):
                written += len(chunk)
                if limited and written > limited:
                    raise HTTPException(status_code=413, detail="audio exceeds the configured upload limit")
                tmp.write(chunk)
        logger.info("transcribing %s (%d bytes)", file.filename, written)

        try:
            text, segments, lang = _transcribe(tmp.name)
        except AudioError as exc:
            logger.warning("decode failed for %s: %s", file.filename, exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        duration = probe_duration(tmp.name)
        logger.info("done %s: %d chars, %d segments", file.filename, len(text), len(segments))
        return JSONResponse(
            {"text": text, "segments": segments, "duration": duration, "language": lang}
        )
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "model": str(config.model_file.name), "threads": config.num_threads}


@app.get("/")
def root() -> dict:
    return {"service": "sensevoice-asr", "endpoint": "/v1/audio/transcriptions"}
