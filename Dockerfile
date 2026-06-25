FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# 1. Instalar PyTorch CON SOPORTE PARA GPU (CUDA 12.1)
# Cambiamos /cpu por /cu121 para aprovechar la NVIDIA T4
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# 2. Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# 3. HORNEAR EL MODELO EN LA IMAGEN
# Configura dónde se guardará y descarga el modelo BETO durante la construcción
ENV HF_HOME=/app/hf_cache
RUN python -c "from transformers import BertTokenizerFast, BertForMaskedLM; model='dccuchile/bert-base-spanish-wwm-cased'; BertTokenizerFast.from_pretrained(model); BertForMaskedLM.from_pretrained(model)"

COPY . .

ENV PORT=8080

# 4. Iniciar con preload
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 120 --preload "main:build_app()"