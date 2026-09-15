"""
test_versioning.py

Tests de `infrastructure/versioning.py`: el `modelVersion` compartido por el
servidor (`main.py`), el caso de uso (`correct_text.py`) y el evaluador
(`evaluate.py`). Solo librería estándar: no carga torch ni modelos.

Uso:
    .venv\\Scripts\\python.exe -m unittest -v test_versioning
"""
import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

from infrastructure.versioning import (
    DEFAULT_MODEL_VERSION,
    REPO_DIR,
    manifest_candidates,
    resolve_model_version,
)


def _repo_with_manifest(tmp: str, name: str, payload) -> Path:
    repo = Path(tmp)
    lora = repo / "models" / "grammar_lora"
    lora.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    (lora / name).write_text(text, encoding="utf-8")
    return repo


class ResolveModelVersionTests(unittest.TestCase):
    def test_default_is_the_unversioned_global_lora(self):
        self.assertEqual(DEFAULT_MODEL_VERSION, "global-lora-unversioned")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(resolve_model_version(env={}, repo_dir=Path(tmp)), DEFAULT_MODEL_VERSION)

    def test_explicit_value_wins_over_env_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo_with_manifest(tmp, "manifest.json", {"modelVersion": "base@1+lora@2"})
            self.assertEqual(
                resolve_model_version(" t5@abc ", env={"MODEL_VERSION": "env-v"}, repo_dir=repo), "t5@abc")

    def test_env_beats_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo_with_manifest(tmp, "manifest.json", {"modelVersion": "base@1+lora@2"})
            self.assertEqual(resolve_model_version(env={"MODEL_VERSION": " env-v "}, repo_dir=repo), "env-v")
            self.assertEqual(resolve_model_version(env={"MODEL_VERSION": "   "}, repo_dir=repo), "base@1+lora@2")

    def test_training_manifest_beats_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo_with_manifest(tmp, "manifest.json", {"modelVersion": "viejo"})
            _repo_with_manifest(tmp, "training-manifest.json", {"modelVersion": "base@a1+lora@b2"})
            self.assertEqual(resolve_model_version(env={}, repo_dir=repo), "base@a1+lora@b2")

    def test_manifest_without_version_falls_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo_with_manifest(tmp, "training-manifest.json", {"otro": 1})
            _repo_with_manifest(tmp, "manifest.json", {"modelVersion": "de-manifest"})
            self.assertEqual(resolve_model_version(env={}, repo_dir=repo), "de-manifest")

    def test_unreadable_manifest_warns_and_uses_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo_with_manifest(tmp, "manifest.json", "{ no es json")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertEqual(resolve_model_version(env={}, repo_dir=repo), DEFAULT_MODEL_VERSION)
            self.assertTrue(any("manifest.json" in str(w.message) for w in caught))

    def test_manifest_candidates_live_under_models_grammar_lora(self):
        candidates = manifest_candidates(Path("repo"))
        self.assertEqual([c.name for c in candidates], ["training-manifest.json", "manifest.json"])
        for c in candidates:
            self.assertEqual(c.parent, Path("repo") / "models" / "grammar_lora")

    def test_repo_dir_is_the_repository_root(self):
        self.assertTrue((REPO_DIR / "infrastructure" / "versioning.py").exists())
        self.assertTrue((REPO_DIR / "evaluate.py").exists())


class SharedDefaultTests(unittest.TestCase):
    def test_use_case_and_evaluator_reexport_the_same_default(self):
        from application.use_cases.correct_text import DEFAULT_MODEL_VERSION as use_case_default
        import evaluate
        self.assertIs(use_case_default, DEFAULT_MODEL_VERSION)
        self.assertIs(evaluate.DEFAULT_MODEL_VERSION, DEFAULT_MODEL_VERSION)
        self.assertIs(evaluate.resolve_model_version, resolve_model_version)

    def test_importing_versioning_does_not_load_torch(self):
        code = ("import sys; import infrastructure.versioning; "
                "import application.use_cases.correct_text; "
                "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_DIR),
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
