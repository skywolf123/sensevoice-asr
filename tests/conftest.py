"""Point ASR_MODEL_DIR at stub files before anything imports server.main.

main.py calls config.ensure_model_files() at import (the container's fail-fast
on a missing model), so the test environment needs plausible paths — the real
recognizer is never built here because tests replace the pool wholesale.
"""

import os
import tempfile
from pathlib import Path

_stub_dir = tempfile.mkdtemp(prefix="sensevoice-test-models-")
Path(_stub_dir, "model.int8.onnx").write_bytes(b"stub")
Path(_stub_dir, "tokens.txt").write_bytes(b"stub")
os.environ["ASR_MODEL_DIR"] = _stub_dir
