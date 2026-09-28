# SenseVoice (int8) on CPU. ffmpeg decodes whatever container the upload
# arrives in (mp3/m4a/aac/webm/...); bzip2 unpacks the model tarball. The
# sherpa-onnx wheels bundle their own onnxruntime and need no extra system
# libraries (libgomp is not required).
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg bzip2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Models are baked in at build time so the NAS starts with zero downloads.
# Swap the ARG to pin a different snapshot; this is the int8-only package
# (zh/en/ja/ko/yue, ~163 MB compressed, ~228 MB unpacked).
ARG SENSEVOICE_URL=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
RUN mkdir -p /models/sense-voice \
    && python -c "import urllib.request;urllib.request.urlretrieve('${SENSEVOICE_URL}', '/tmp/sv.tar.bz2')" \
    && tar -xjf /tmp/sv.tar.bz2 -C /models/sense-voice --strip-components=1 \
    && rm /tmp/sv.tar.bz2 \
    && rm -rf /models/sense-voice/test_wavs

COPY server/ server/

ENV ASR_MODEL_DIR=/models/sense-voice \
    ASR_NUM_THREADS=2 \
    ASR_POOL_SIZE=1
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3)"
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8080"]
