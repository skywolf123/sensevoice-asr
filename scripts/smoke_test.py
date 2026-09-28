"""End-to-end smoke test.

In-process mode drives decode → segment → recognize and reports the RTF
(transcribe seconds / audio seconds) — the number that decides whether a
given CPU is usable. --http mode instead POSTs the file to a running server,
exercising the FastAPI layer with a hand-built multipart body (no extra deps).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _multipart(path: str) -> tuple[bytes, str]:
    boundary = "----sensevoice" + uuid.uuid4().hex
    parts = bytearray()
    parts += f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\nsensevoice\r\n'.encode()
    parts += (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{Path(path).name}"\r\nContent-Type: audio/wav\r\n\r\n'
    ).encode()
    parts += Path(path).read_bytes()
    parts += f"\r\n--{boundary}--\r\n".encode()
    return bytes(parts), f"multipart/form-data; boundary={boundary}"


def in_process(wav: str) -> None:
    from server.audio import decode, plan_clips
    from server.config import config
    from server.engine import Pool

    samples = decode(wav)
    duration = samples.shape[0] / 16000
    clips = plan_clips(wav, samples, config.segment_ms, config.noise_db)
    print(f"audio {duration:.1f}s -> {len(clips)} clip(s)")

    pool = Pool(config)
    recognizer = pool.checkout()
    started = time.time()
    texts: list[str] = []
    try:
        for clip in clips:
            result = recognizer.recognize(clip.samples)
            if result.text:
                texts.append(result.text)
    finally:
        pool.release(recognizer)
    took = time.time() - started

    rtf = took / duration if duration else 0.0
    print(f"transcribed in {took:.1f}s (RTF {rtf:.2f})")
    print("text:", "".join(texts)[:300])


def http(base: str, wav: str) -> None:
    body, content_type = _multipart(wav)
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": content_type},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        payload = json.load(response)
    print(json.dumps(payload, ensure_ascii=False, indent=2)[:1000])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", help="audio file to transcribe")
    parser.add_argument("--http", help="POST to this base URL instead of running in-process")
    args = parser.parse_args()
    if args.http:
        http(args.http, args.wav)
    else:
        in_process(args.wav)
