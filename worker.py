"""
worker.py — entry point del proceso de entrenamiento (ROLE=worker).

Corre SEPARADO de los procesos de API. No sirve HTTP, no carga el
modelo de inferencia: solo consume la cola Redis compartida y entrena
los adaptadores LoRA, uno a la vez.

Uso:
    REDIS_URL=redis://localhost:6379/0 python worker.py

En Docker/systemd, así arrancarías los distintos roles:
    # 1 worker
    REDIS_URL=redis://redis:6379/0 ROLE=worker python worker.py

    # N réplicas de api (detrás de un load balancer)
    REDIS_URL=redis://redis:6379/0 ROLE=api PORT=8080 python main.py
    REDIS_URL=redis://redis:6379/0 ROLE=api PORT=8081 python main.py
    ...
"""
import os

os.environ.setdefault("ROLE", "worker")

from main import run_worker

if __name__ == "__main__":
    run_worker()
