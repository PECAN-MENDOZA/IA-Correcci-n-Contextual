FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# 1. PyTorch con CUDA 12.1 (2.4.1 estable, sin bug de distributed)
RUN pip install --no-cache-dir torch==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Dependencias (incluye peft para LoRA)
RUN pip install --no-cache-dir -r requirements.txt

# 3. Hornear BETO en la imagen durante el build
ENV HF_HOME=/app/hf_cache
ENV HF_HUB_DISABLE_IMPLICIT_TOKEN=1
ENV HF_HUB_DISABLE_TELEMETRY=1

COPY scripts/prefetch_beto.py ./scripts/prefetch_beto.py
RUN python scripts/prefetch_beto.py

COPY . .

ENV PORT=8080

# 4. Arrancar servidor
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 120 --preload "main:build_app()"