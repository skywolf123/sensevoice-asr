"""HTTP-layer tests with a stubbed recognizer — no model needed.

Covers the parts of the endpoint that a wrong wire format would break for
WeKnora: multipart parsing, the JSON response shape, and error mapping.
The engine pool is swapped for a stub so these run in seconds on any machine.
"""

from __future__ import annotations

import io
import wave

import pytest
from fastapi.testclient import TestClient

from server import main
from server.engine import ClipResult, Recognizer


class StubRecognizer(Recognizer):
    """Returns one fixed transcript regardless of the audio."""

    def __init__(self, text: str = "你好世界"):
        super().__init__(inner=None)  # type: ignore[arg-type]
        self._text = text

    def recognize(self, samples) -> ClipResult:  # noqa: ANN001 - stub signature
        seconds = len(samples) / 16000
        return ClipResult(text=self._text, language="zh", emotion="NEUTRAL", event="Speech")


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(main, "_pool", _FakePool(stub=StubRecognizer()))
    return TestClient(main.app)


class _FakePool:
    def __init__(self, stub: StubRecognizer):
        self._stub = stub

    def checkout(self) -> StubRecognizer:
        return self._stub

    def release(self, recognizer) -> None:  # noqa: ANN001
        pass


def _wav(seconds: float = 1.0, frequency: int = 440) -> bytes:
    """A real WAV in memory: loud enough that silencedetect keeps it."""
    import math
    import struct

    rate = 16000
    frames = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * frequency * i / rate)))
        for i in range(int(rate * seconds))
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return buffer.getvalue()


def test_transcribe_response_shape(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _wav(2.0), "audio/wav")},
        data={"model": "sensevoice"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "你好世界"
    assert payload["language"] == "zh"
    assert payload["duration"] == pytest.approx(2.0, abs=0.1)
    assert payload["segments"] == [
        {"start": payload["segments"][0]["start"], "end": payload["segments"][0]["end"], "text": "你好世界"}
    ]
    assert payload["segments"][0]["start"] == pytest.approx(0.0, abs=0.5)


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_invalid_audio_maps_to_400(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", b"not audio at all", "audio/wav")},
        data={"model": "sensevoice"},
    )
    assert response.status_code == 400


def test_file_required(client):
    response = client.post("/v1/audio/transcriptions", data={"model": "sensevoice"})
    assert response.status_code == 422
