FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# 1. PyTorch con CUDA 12.8 (soporta Blackwell/RTX 50-series, sm_120)
RUN pip install --no-cache-dir torch==2.6.0  \
    --index-url https://download.pytorch.org/whl/cpu


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
# Logs sin buffer: si no, los print() de arranque/errores quedan atrapados en
# el buffer de stdout y nunca aparecen en `docker logs` (nos dejó ciegos ante
# los fallos silenciosos de T5). Con esto los [OK]/[WARN] salen al instante.
ENV PYTHONUNBUFFERED=1

# 4. Arrancar servidor
#
# SIN --preload a propósito: con --preload el modelo se carga en el proceso
# maestro y CUDA se inicializa ANTES del fork del worker; entonces cualquier
# .generate()/.forward() en el worker lanza
#   "RuntimeError: Cannot re-initialize CUDA in forked subprocess"
# y la inferencia (T5 y BETO) cae siempre al fallback. Al quitar --preload la
# app se construye dentro del worker (post-fork) y CUDA se inicializa ahí, que
# es lo que necesita la desambiguación con BETO. Con --workers 1 no hay coste
# de memoria extra. Se usa main:app (la app a nivel de módulo) para cargar el
# modelo una sola vez y no dos.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 120 "main:app"