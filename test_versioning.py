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
    BASE_TAG,
    DEFAULT_MODEL_VERSION,
    LORA_TAG,
    MODEL_MANIFEST_NAME,
    REPO_DIR,
    compose_model_version,
    manifest_candidates,
    model_manifest_path,
    resolve_model_version,
    served_model_version,
)


def _repo_with_manifest(tmp: str, name: str, payload) -> Path:
    repo = Path(tmp)
    lora = repo / "models" / "grammar_lora"
    lora.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    (lora / name).write_text(text, encoding="utf-8")
    return repo


def _repo_with_model_manifest(tmp: str, payload) -> Path:
    repo = Path(tmp)
    base = repo / "models" / "t5_correction"
    base.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    (base / MODEL_MANIFEST_NAME).write_text(text, encoding="utf-8")
    return repo


BASE_HASH = "0123456789abcdef" * 4
LORA_HASH = "fedcba9876543210" * 4


class ComposeModelVersionTests(unittest.TestCase):
    def test_composes_base_and_lora_tags_with_eight_hex(self):
        base = {"model_sha256": BASE_HASH}
        lora = {"adapter_sha256": LORA_HASH}
        self.assertEqual(compose_model_version(base, lora), f"{BASE_TAG}@01234567+{LORA_TAG}@fedcba98")
        self.assertEqual(BASE_TAG, "beto-t5-base")
        self.assertEqual(LORA_TAG, "global-lora-v1")

    def test_tags_come_from_the_manifests_or_explicit_arguments(self):
        base = {"model_sha256": BASE_HASH, "base_tag": "t5-x"}
        lora = {"adapter_sha256": LORA_HASH, "lora_tag": "lora-y"}
        self.assertEqual(compose_model_version(base, lora), "t5-x@01234567+lora-y@fedcba98")
        self.assertEqual(compose_model_version(base, lora, base_tag="b", lora_tag="l"), "b@01234567+l@fedcba98")

    def test_rejects_missing_short_or_non_hex_hashes(self):
        for base, lora in (
            ({}, {"adapter_sha256": LORA_HASH}),
            ({"model_sha256": BASE_HASH}, {}),
            ({"model_sha256": "abc"}, {"adapter_sha256": LORA_HASH}),
            ({"model_sha256": BASE_HASH}, {"adapter_sha256": "g" * 64}),
            ("no-dict", {"adapter_sha256": LORA_HASH}),
        ):
            with self.assertRaises(ValueError):
                compose_model_version(base, lora)

    def test_hashes_are_normalised_to_lowercase(self):
        version = compose_model_version({"model_sha256": BASE_HASH.upper()}, {"adapter_sha256": LORA_HASH.upper()})
        self.assertEqual(version, f"{BASE_TAG}@01234567+{LORA_TAG}@fedcba98")


class ComposedResolutionTests(unittest.TestCase):
    def test_both_manifests_compose_the_version_from_the_merged_one(self):
        # Ambos manifiestos describen la misma generación: la versión sale del
        # manifiesto del modelo fusionado (que embebe su propio adapter_sha256).
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo_with_model_manifest(tmp, {"model_sha256": BASE_HASH, "adapter_sha256": LORA_HASH,
                                                   "modelVersion": "ignorada"})
            _repo_with_manifest(tmp, "training-manifest.json",
                                {"adapter_sha256": LORA_HASH, "modelVersion": f"{LORA_TAG}@fedcba98"})
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertEqual(resolve_model_version(env={}, repo_dir=repo),
                                 f"{BASE_TAG}@01234567+{LORA_TAG}@fedcba98")
            self.assertEqual(caught, [])

    def test_newer_training_manifest_never_mixes_with_the_merged_model(self):
        # LoRA nuevo entrenado (training-manifest 2222…) pero todavía no
        # fusionado: los pesos servidos siguen siendo el modelo fusionado con el
        # adaptador 1111…, y la versión debe describir ESE adaptador, con aviso.
        old_adapter = "1" * 64
        new_adapter = "2" * 64
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo_with_model_manifest(tmp, {"model_sha256": BASE_HASH, "adapter_sha256": old_adapter})
            _repo_with_manifest(tmp, "training-manifest.json",
                                {"adapter_sha256": new_adapter, "modelVersion": f"{LORA_TAG}@22222222"})
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                version = resolve_model_version(env={}, repo_dir=repo)
            self.assertEqual(version, f"{BASE_TAG}@01234567+{LORA_TAG}@11111111")
            self.assertNotIn("22222222", version)
            self.assertTrue(any("adaptador entrenado" in str(w.message) and "adaptador fusionado" in str(w.message)
                                for w in caught), [str(w.message) for w in caught])

    def test_model_manifest_alone_composes_from_its_embedded_adapter_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo_with_model_manifest(tmp, {"model_sha256": BASE_HASH, "adapter_sha256": LORA_HASH})
            self.assertEqual(resolve_model_version(env={}, repo_dir=repo), f"{BASE_TAG}@01234567+{LORA_TAG}@fedcba98")

    def test_model_manifest_without_adapter_hash_is_not_completed_with_the_training_manifest(self):
        # Sin adapter_sha256 propio no se compone (nunca se toma el del LoRA
        # actual): aviso y se cae al modelVersion del manifiesto del LoRA.
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo_with_model_manifest(tmp, {"model_sha256": BASE_HASH})
            _repo_with_manifest(tmp, "training-manifest.json",
                                {"adapter_sha256": LORA_HASH, "modelVersion": f"{LORA_TAG}@fedcba98"})
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertEqual(resolve_model_version(env={}, repo_dir=repo), f"{LORA_TAG}@fedcba98")
            self.assertTrue(any(MODEL_MANIFEST_NAME in str(w.message) for w in caught))

    def test_only_lora_manifest_uses_its_model_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo_with_manifest(tmp, "training-manifest.json",
                                       {"adapter_sha256": LORA_HASH, "modelVersion": f"{LORA_TAG}@fedcba98"})
            self.assertEqual(resolve_model_version(env={}, repo_dir=repo), f"{LORA_TAG}@fedcba98")

    def test_env_and_explicit_still_beat_the_composed_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo_with_model_manifest(tmp, {"model_sha256": BASE_HASH})
            _repo_with_manifest(tmp, "training-manifest.json", {"adapter_sha256": LORA_HASH})
            self.assertEqual(resolve_model_version(env={"MODEL_VERSION": "env-v"}, repo_dir=repo), "env-v")
            self.assertEqual(resolve_model_version("expl", env={"MODEL_VERSION": "env-v"}, repo_dir=repo), "expl")

    def test_model_manifest_without_hash_warns_and_falls_back_to_lora_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo_with_model_manifest(tmp, {"otro": 1})
            _repo_with_manifest(tmp, "training-manifest.json",
                                {"adapter_sha256": LORA_HASH, "modelVersion": f"{LORA_TAG}@fedcba98"})
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertEqual(resolve_model_version(env={}, repo_dir=repo), f"{LORA_TAG}@fedcba98")
            self.assertTrue(any(MODEL_MANIFEST_NAME in str(w.message) for w in caught))

    def test_unreadable_model_manifest_warns_and_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _repo_with_model_manifest(tmp, "{ roto")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertEqual(resolve_model_version(env={}, repo_dir=repo), DEFAULT_MODEL_VERSION)
            self.assertTrue(any(MODEL_MANIFEST_NAME in str(w.message) for w in caught))

    def test_model_manifest_path_lives_under_models_t5_correction(self):
        self.assertEqual(model_manifest_path(Path("repo")), Path("repo") / "models" / "t5_correction" / MODEL_MANIFEST_NAME)


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


class ServedModelVersionTests(unittest.TestCase):
    """La versión que sirve `main.py` solo puede ser la compuesta si el T5 está cargado."""

    COMPOSED = f"{BASE_TAG}@01234567+{LORA_TAG}@fedcba98"

    def test_composed_version_is_served_when_t5_is_loaded(self):
        self.assertEqual(served_model_version(True, None, self.COMPOSED), self.COMPOSED)
        self.assertEqual(served_model_version(True, "", self.COMPOSED), self.COMPOSED)

    def test_explicit_version_is_served_with_or_without_t5(self):
        self.assertEqual(served_model_version(True, " v-explicita ", self.COMPOSED), "v-explicita")
        self.assertEqual(served_model_version(False, "rules-beto-only", self.COMPOSED), "rules-beto-only")

    def test_refuses_to_serve_without_t5_and_without_explicit_version(self):
        # El pipeline servido sería reglas+BETO: la versión compuesta (o la del
        # LoRA, o el default) describiría un modelo que no corrige.
        for composed in (self.COMPOSED, f"{LORA_TAG}@fedcba98", DEFAULT_MODEL_VERSION):
            with self.assertRaises(ValueError) as ctx:
                served_model_version(False, None, composed)
            self.assertIn("MODEL_VERSION", str(ctx.exception))
        with self.assertRaises(ValueError):
            served_model_version(False, "   ", self.COMPOSED)


class SharedDefaultTests(unittest.TestCase):
    def test_use_case_and_evaluator_reexport_the_same_default(self):
        from application.use_cases.correct_text import DEFAULT_MODEL_VERSION as use_case_default
        import evaluate
        self.assertIs(use_case_default, DEFAULT_MODEL_VERSION)
        self.assertIs(evaluate.DEFAULT_MODEL_VERSION, DEFAULT_MODEL_VERSION)
        self.assertIs(evaluate.resolve_model_version, resolve_model_version)

    def test_main_uses_the_shared_resolver(self):
        """`main.py` no puede importarse sin torch: se inspecciona el código fuente."""
        source = (REPO_DIR / "main.py").read_text(encoding="utf-8")
        self.assertIn("from infrastructure.versioning import", source)
        self.assertIn("resolve_model_version", source)
        self.assertNotIn("def resolve_model_version", source)
        self.assertNotIn("GRAMMAR_LORA_MANIFEST", source)

    def test_main_refuses_to_start_without_t5_unless_the_version_is_explicit(self):
        """`build_app` pasa por `served_model_version` y convierte su error en una salida clara."""
        source = (REPO_DIR / "main.py").read_text(encoding="utf-8")
        self.assertIn("served_model_version(", source)
        self.assertIn("t5_loaded=t5_model is not None", source)
        self.assertIn('os.environ.get("MODEL_VERSION")', source)
        self.assertRegex(source, r"except ValueError as exc:\s*\n\s*sys\.exit\(f?\"\[ERROR\]")

    def test_importing_versioning_does_not_load_torch(self):
        code = ("import sys; import infrastructure.versioning; "
                "import application.use_cases.correct_text; "
                "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_DIR),
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
