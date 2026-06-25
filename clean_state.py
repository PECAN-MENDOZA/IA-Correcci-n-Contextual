"""
clean_state.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Limpia el estado corrupto dejado por sesiones anteriores.

Ejecutar UNA VEZ antes de reiniciar el servidor:
    python clean_state.py

Qué hace:
  - Borra modelos de usuario corruptos (models/users/)
  - NO toca el modelo base (models/t5_correction/)
  - NO toca los datos de usuario (data/users/*.csv)
  - Crea los directorios necesarios si no existen
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import shutil
from pathlib import Path

USER_MODELS_DIR = Path("./models/users")
DATA_USERS_DIR  = Path("./data/users")

print("=== LIMPIEZA DE ESTADO ===\n")

# 1. Borrar modelos de usuario (pueden estar corruptos)
if USER_MODELS_DIR.exists():
    count = sum(1 for _ in USER_MODELS_DIR.iterdir() if _.is_dir())
    shutil.rmtree(USER_MODELS_DIR)
    print(f"[OK] Borrados {count} modelo(s) de usuario en {USER_MODELS_DIR}/")
    print("     (se regenerarán con el primer fine-tuning exitoso)")
else:
    print(f"[OK] {USER_MODELS_DIR}/ no existía — nada que limpiar")

USER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 2. Asegurar que data/users existe (los CSV de historial se conservan)
DATA_USERS_DIR.mkdir(parents=True, exist_ok=True)
csv_count = len(list(DATA_USERS_DIR.glob("*.csv")))
print(f"[OK] {DATA_USERS_DIR}/ conservado con {csv_count} archivo(s) de historial")

# 3. Asegurar que data/ existe para otros posibles subdirectorios
Path("./data").mkdir(exist_ok=True)

print("\n[LISTO] Ahora ejecuta: python main.py\n")
