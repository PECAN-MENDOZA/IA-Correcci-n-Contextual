"""
infrastructure/persistence/csv_user_repository.py
Repositorio CSV: implementa IUserHistoryRepository usando archivos CSV locales.
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
    """Persiste el historial de correcciones de cada usuario en un CSV individual."""

    def __init__(self, data_dir: str = "./data/users"):
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── IUserHistoryRepository ─────────────────────────────────────────────────

    def save_feedback(self, feedback: UserFeedback) -> None:
        path        = self._dir / f"{feedback.user_id}_history.csv"
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
        df = pd.read_csv(path, encoding="utf-8")
        return list(zip(df["erronea"], df["corregida"]))

    def get_user_vocabulary(self, user_id: str) -> dict:
        path     = self._dir / f"{user_id}_history.csv"
        word_map = {}
        if not path.exists():
            return word_map

        with open(path, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ew = row.get("erronea",  "").split()
                cw = row.get("corregida","").split()
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
