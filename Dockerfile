# Dockerfile (root of repo)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libglib2.0-0 \
      libgl1 \
      libstdc++6 \
      ca-certificates \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# 1) Install Python deps (pinned in requirements.txt)
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# 2) Copy source
COPY . .


# ensure the package dir exists
RUN mkdir -p agguard/proto && touch agguard/proto/__init__.py

# generate stubs INTO the same package (no nested mirroring)
RUN python -m grpc_tools.protoc \
      -I agguard/proto \
      --python_out=agguard/proto \
      --grpc_python_out=agguard/proto \
      ingest.proto

# patch generated imports to be package-relative (ingest_pb2_grpc -> .ingest_pb2)
RUN python - <<'PY'
from pathlib import Path
import re
p = Path("agguard/proto/ingest_pb2_grpc.py")
s = p.read_text(encoding="utf-8")
s2 = re.sub(r'(?m)^import (\w+_pb2)\b', r'from . import \1', s)
if s2 != s:
    p.write_text(s2, encoding="utf-8")
    print("patched", p)
else:
    print("no patch needed", p)
PY

ENV AGGUARD_CFG=/app/configs/default.yaml
EXPOSE 50051
CMD ["python", "-m", "agguard.app.grpc_server"]
