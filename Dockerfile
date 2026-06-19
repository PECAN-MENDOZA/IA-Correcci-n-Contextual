FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# 1. Instalar PyTorch versión CPU explícitamente para reducir el tamaño de >2GB a ~200MB
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# 2. Instalar el resto de dependencias
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080

# 3. Ejecutar gunicorn usando 'exec' para procesar la variable $PORT y manejar threads
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 120 "main:build_app()"
