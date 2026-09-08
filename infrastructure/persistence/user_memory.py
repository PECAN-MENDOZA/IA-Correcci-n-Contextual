"""
infrastructure/persistence/user_memory.py
Memoria de correcciones validadas por el usuario, POR-USUARIO y COMPARTIDA entre
réplicas vía Redis. Reemplaza al diccionario en RAM del pipeline, que (1) no se
compartía entre las réplicas api1/api2/api3 y (2) no estaba indexado por usuario
(una corrección de un alumno se filtraba a otros).

Clave Redis: hash `usermem:{student_id}`  →  { texto_normalizado: corrección }
  - lookup exacto: HGET  (O(1))
  - lookup difuso: HGETALL del usuario (conjunto pequeño) + similitud

Si Redis no está disponible (p. ej. ROLE=all en local), se cae a una
implementación en memoria con la misma interfaz.
"""
import os
import re
import unicodedata
from difflib import SequenceMatcher

try:
    import redis
except ImportError:
    redis = None

# Umbral de similitud para aceptar un match difuso (igual que el pipeline).
_SIMILARITY_THRESHOLD = 0.82


def _normalize(text: str) -> str:
    """Minúsculas, sin tildes, sin puntuación, espacios colapsados."""
    text = text.lower().strip()
    text = "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


class RedisUserMemory:
    """Memoria por-usuario respaldada por Redis (compartida entre réplicas)."""

    def __init__(self, url: str = None):
        if redis is None:
            raise ImportError("Falta el paquete 'redis'.")
        url = url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._client = redis.Redis.from_url(url)
        self._client.ping()

    @staticmethod
    def _key(student_id: str) -> str:
        return f"usermem:{student_id}"

    def remember(self, student_id: str, original: str, correction: str) -> None:
        self._client.hset(self._key(student_id), _normalize(original), correction.strip())

    def lookup(self, student_id: str, text: str) -> str | None:
        key = self._key(student_id)
        norm = _normalize(text)

        exact = self._client.hget(key, norm)     # caso común, O(1)
        if exact is not None:
            return exact.decode("utf-8")

        # Difuso: solo sobre las entradas de ESTE usuario (conjunto pequeño).
        best, best_score = None, 0.0
        for raw_key, raw_val in self._client.hgetall(key).items():
            score = _similarity(norm, raw_key.decode("utf-8"))
            if score > best_score:
                best_score, best = score, raw_val.decode("utf-8")
        return best if best_score >= _SIMILARITY_THRESHOLD else None


class InMemoryUserMemory:
    """Fallback en memoria (un solo proceso), misma interfaz que RedisUserMemory."""

    def __init__(self):
        self._store: dict[str, dict[str, str]] = {}

    def remember(self, student_id: str, original: str, correction: str) -> None:
        self._store.setdefault(student_id, {})[_normalize(original)] = correction.strip()

    def lookup(self, student_id: str, text: str) -> str | None:
        entries = self._store.get(student_id)
        if not entries:
            return None
        norm = _normalize(text)
        if norm in entries:
            return entries[norm]
        best, best_score = None, 0.0
        for k, v in entries.items():
            score = _similarity(norm, k)
            if score > best_score:
                best_score, best = score, v
        return best if best_score >= _SIMILARITY_THRESHOLD else None


def build_user_memory():
    """Redis si está disponible; si no, en memoria. Nunca lanza."""
    try:
        mem = RedisUserMemory()
        print("[INFO] Memoria de usuario: Redis (compartida entre réplicas).")
        return mem
    except Exception as exc:
        print(f"[WARN] Redis no disponible para memoria de usuario ({exc}). "
              f"Usando memoria en proceso.")
        return InMemoryUserMemory()
