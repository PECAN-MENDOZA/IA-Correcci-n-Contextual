"""
infrastructure/persistence/csv_user_repository.py

CORRECCIÓN: save_feedback() crea el directorio en el momento de escribir,
no solo en __init__. Esto evita FileNotFoundError si el proceso cambia de
directorio de trabajo después de la inicialización, o si la carpeta es
eliminada mientras el servidor está corriendo.
"""
import csv
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from domain.entities.user_feedback import UserFeedback
from domain.repositories.interfaces import IUserHistoryRepository

_WORD_RE = re.compile(r'^([^\wáéíóúüñÁÉÍÓÚÜÑ]*)(.*?)([^\wáéíóúüñÁÉÍÓÚÜÑ]*)$')


class CsvUserHistoryRepository(IUserHistoryRepository):

    def __init__(self, data_dir: str = "./data/users"):
        self._data_dir_str = data_dir          # guardamos el string original
        self._dir = Path(data_dir).resolve()   # ruta absoluta desde el inicio
        self._dir.mkdir(parents=True, exist_ok=True)

    def save_feedback(self, feedback: UserFeedback) -> None:
        # Crear directorio si fue borrado en caliente o si cwd cambió
        self._dir.mkdir(parents=True, exist_ok=True)

        path         = self._dir / f"{feedback.user_id}_history.csv"
        write_header = not path.exists()
        with open(path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["erronea", "corregida"])
            writer.writerow([
                feedback.original_text.lower(),
                feedback.corrected_text.lower(),
            ])

    def get_user_pairs(self, user_id: str) -> List[Tuple[str, str]]:
        path = self._dir / f"{user_id}_history.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path, encoding="utf-8", dtype=str)
        df = df.dropna(subset=["erronea", "corregida"])
        df["erronea"]   = df["erronea"].str.strip()
        df["corregida"] = df["corregida"].str.strip()
        df = df[(df["erronea"] != "") & (df["corregida"] != "")]
        return list(zip(df["erronea"], df["corregida"]))

    def get_user_vocabulary(self, user_id: str) -> dict:
        path     = self._dir / f"{user_id}_history.csv"
        word_map = {}
        if not path.exists():
            return word_map
        with open(path, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ew = row.get("erronea",   "").split()
                cw = row.get("corregida", "").split()
                if len(ew) != len(cw):
                    continue
                for e, c in zip(ew, cw):
                    m_e = _WORD_RE.match(e)
                    m_c = _WORD_RE.match(c)
                    core_e = m_e.group(2) if m_e else e
                    core_c = m_c.group(2) if m_c else c
                    if core_e and core_e.lower() != core_c.lower():
                        word_map[core_e.lower()] = core_c.lower()
        return word_map
