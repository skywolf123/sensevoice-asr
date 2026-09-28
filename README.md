# SenseVoice ASR — OpenAI-compatible speech-to-text for WeKnora

A small HTTP service that wraps [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)'s
SenseVoice (int8, CPU) into the OpenAI `/v1/audio/transcriptions` shape, so any
OpenAI-compatible client — WeKnora in particular — can use it as a local ASR
model with no code changes.

Fully local: no network calls at inference time, no API keys, no rate limits.

## Why

WeKnora's ASR wire layer speaks two protocols (openai-transcriptions,
openai-chat-audio). SenseVoice is a Python/C++ library, not an HTTP service.
This container is the bridge: it accepts the OpenAI multipart form and returns
`{"text", "segments", "duration", "language"}` with per-segment timestamps.

Built for a 4-core Celeron J4125 NAS: int8 inference, ~400 MB resident, chunked
decode so a 1-hour video never holds more than ~30 s of audio in the
recognizer. Official sherpa-onnx benchmarks put the int8 model around RTF 0.2
on comparable low-power 4-core CPUs (silence detection skips non-speech, so
talking-head videos transcribe faster than their wall length).

## API

- `POST /v1/audio/transcriptions` — multipart `file` (+ optional `model`,
  `language`, `response_format`, accepted and ignored: language is fixed at
  container start via `ASR_LANGUAGE`, segments are always returned).
- `GET /healthz` — liveness for Unraid/compose health checks.
- `GET /` — service banner.

No auth: bind it to a container network only. If you publish the port, front it
with your own guard.

## Configuration (env)

| Variable | Default | Meaning |
| --- | --- | --- |
| `ASR_MODEL_DIR` | `/models/sense-voice` | Model directory |
| `ASR_MODEL_FILE` | `model.int8.onnx` | Recognizer ONNX file |
| `ASR_TOKENS_FILE` | `tokens.txt` | Tokenizer file |
| `ASR_LANGUAGE` | `auto` | `auto`/`zh`/`en`/`ja`/`ko`/`yue` (fixed per container) |
| `ASR_NUM_THREADS` | `2` | ONNX intra-op threads per recognizer |
| `ASR_POOL_SIZE` | `1` | Concurrent recognizers (each ~300 MB; 4-core boxes: keep at 1-2) |
| `ASR_SEGMENT_MS` | `30000` | Max clip length fed to the recognizer |
| `ASR_NOISE_DB` | `-35` | ffmpeg silencedetect floor; quieter audio is skipped |
| `ASR_MAX_UPLOAD_MB` | `512` | Upload cap |
| `ASR_HOST` / `ASR_PORT` | `0.0.0.0` / `8080` | Listen address |

## Running

```bash
docker run --rm -p 8080:8080 ghcr.io/skywolf123/sensevoice-asr:latest
curl -F file=@sample.mp3 -F model=sensevoice http://127.0.0.1:8080/v1/audio/transcriptions
```

Models are baked into the image at build time (downloaded in CI from the
sherpa-onnx releases), so the container starts with zero downloads.

## WeKnora wiring

1. Model settings → add an ASR model:
   - provider `generic`（自定义 OpenAI 兼容接口）, type `ASR`
   - base URL `http://sensevoice-asr:8080/v1`（compose 网络内的服务名）
2. Model `spec.compat` JSON:

   ```json
   { "request_timeout_seconds": 3600 }
   ```

   The protocol default is 300 s, which a weak CPU will blow through on long
   audio. `path` can stay `/audio/transcriptions` (the default) — combined with
   the `/v1` base URL it hits the endpoint above.
3. Add the service hostname to `SSRF_WHITELIST_EXTRA` in the deployment env,
   otherwise the backend's SSRF guard refuses the private address.

## Development

```bash
uv venv && uv pip install -r requirements.txt
ASR_MODEL_DIR=... uvicorn server.main:app --port 8080
```

Models land in `models/` via `scripts/fetch-models.sh` (same files CI bakes in).
