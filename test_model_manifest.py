"""
test_model_manifest.py

Tests de `scripts/model_manifest.py`: manifiestos reproducibles del LoRA
gramatical global (`training-manifest.json`) y del modelo fusionado
(`model-manifest.json`), y la composición del `modelVersion`
`<base-tag>@<hash8>+<lora-tag>@<hash8>`.

Puros: solo librería estándar, directorios temporales y archivos falsos; no
cargan torch ni ningún modelo ni tocan `models/`.

Uso:
    .venv\\Scripts\\python.exe -m unittest -v test_model_manifest
"""
import hashlib
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.model_manifest import (
    BASE_MODEL,
    MODEL_MANIFEST_NAME,
    TRAINING_DEFAULTS,
    TRAINING_MANIFEST_NAME,
    REPO_DIR,
    adapter_fields,
    base_version,
    compose_model_version,
    load_pairs,
    merged_model_manifest,
    reconstructed_training_manifest,
    resolve_merge_base,
    sha256_file,
    split_indices,
    training_manifest,
    validate_merged_dir,
    write_current,
    write_json,
)

FIXED_NOW = "2026-09-15T00:00:00Z"


def _fake_adapter(directory: Path, weights: bytes = b"adapter-bytes", rank: int = 16) -> Path:
    """Crea un adaptador falso: `adapter_model.safetensors` + `adapter_config.json`."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_model.safetensors").write_bytes(weights)
    (directory / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": BASE_MODEL, "r": rank, "lora_alpha": 2 * rank,
        "lora_dropout": 0.05, "target_modules": ["q", "v"], "peft_version": "0.20.0",
    }), encoding="utf-8")
    return directory


def _fake_merged(directory: Path, weights: bytes = b"merged-bytes", vocab: str = "spiece.model") -> Path:
    """Crea un modelo fusionado falso con los archivos que el pipeline exige (incluido el vocabulario)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.safetensors").write_bytes(weights)
    for name in ("config.json", "tokenizer_config.json", "generation_config.json"):
        (directory / name).write_text("{}", encoding="utf-8")
    if vocab:
        (directory / vocab).write_bytes(b"vocab")
    return directory


def _dataset(directory: Path, rows=(("iva", "iba"), ("ola", "hola"))) -> Path:
    path = directory / "train.csv"
    lines = ["bad,good"] + [f"{a},{b}" for a, b in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class ModelManifestTests(unittest.TestCase):
    def test_manifest_records_exact_data_and_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            path.write_text("bad,good\niva,iba\n", encoding="utf-8")
            manifest = training_manifest(path, {"seed": 42, "epochs": 3})
            self.assertEqual(manifest["dataset_sha256"], sha256_file(path))
            self.assertEqual(manifest["parameters"]["seed"], 42)

    def test_sha256_file_hashes_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.bin"
            path.write_bytes(b"\x00hola\n")
            self.assertEqual(sha256_file(path), hashlib.sha256(b"\x00hola\n").hexdigest())
            self.assertRegex(sha256_file(path), r"^[0-9a-f]{64}$")

    def test_manifest_records_rows_split_base_and_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _dataset(Path(directory), rows=[(f"e{i}", f"c{i}") for i in range(10)])
            manifest = training_manifest(path, {"seed": 7, "validation_size": 3}, now=FIXED_NOW)
            self.assertEqual(manifest["base_model"], BASE_MODEL)
            self.assertEqual(manifest["provenance"], "trained")
            self.assertEqual(manifest["created_at_utc"], FIXED_NOW)
            self.assertEqual(manifest["dataset"]["rows"], 10)
            self.assertEqual(manifest["dataset"]["validation_rows"], 3)
            self.assertEqual(manifest["dataset"]["train_rows"], 7)
            self.assertEqual(manifest["dataset"]["sha256"], manifest["dataset_sha256"])
            train_idx, val_idx = split_indices(10, seed=7, validation_size=3)
            self.assertEqual(manifest["split"]["validation_indices"], val_idx)
            self.assertEqual(manifest["split"]["train_indices"], train_idx)
            self.assertEqual(manifest["split"]["seed"], 7)
            self.assertIn("python", manifest["libraries"])
            # Los parámetros no indicados toman los valores por defecto del entrenamiento.
            for key, value in TRAINING_DEFAULTS.items():
                if key not in ("seed", "validation_size"):
                    self.assertEqual(manifest["parameters"][key], value)
            self.assertEqual(manifest["parameters"]["seed"], 7)
            self.assertNotIn("modelVersion", manifest)  # sin adaptador todavía

    def test_manifest_is_json_serializable_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _dataset(Path(directory))
            manifest = training_manifest(path, {"seed": 42}, now=FIXED_NOW)
            out = Path(directory) / "m.json"
            write_json(out, manifest)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8")), manifest)


class SplitIndicesTests(unittest.TestCase):
    def test_split_matches_the_training_script_shuffle(self):
        """`random.seed(seed); random.shuffle(pairs); pairs[:val]` da la misma partición."""
        pairs = [(f"e{i}", f"c{i}") for i in range(50)]
        shuffled = list(pairs)
        random.seed(42)
        random.shuffle(shuffled)
        val_pairs, train_pairs = shuffled[:5], shuffled[5:]
        train_idx, val_idx = split_indices(len(pairs), seed=42, validation_size=5)
        self.assertEqual([pairs[i] for i in val_idx], val_pairs)
        self.assertEqual([pairs[i] for i in train_idx], train_pairs)

    def test_split_is_deterministic_and_covers_every_row(self):
        a = split_indices(20, seed=42, validation_size=4)
        b = split_indices(20, seed=42, validation_size=4)
        self.assertEqual(a, b)
        self.assertEqual(sorted(a[0] + a[1]), list(range(20)))
        self.assertNotEqual(a, split_indices(20, seed=43, validation_size=4))

    def test_load_pairs_skips_header_and_incomplete_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.csv"
            path.write_text("erronea,corregida\niva,iba\nsolo\n,vacia\nola,hola\n", encoding="utf-8")
            self.assertEqual(load_pairs(path), [("iva", "iba"), ("ola", "hola")])


class AdapterAndComposeTests(unittest.TestCase):
    def test_adapter_fields_hash_the_adapter_and_give_a_partial_version(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = _fake_adapter(Path(directory) / "grammar_lora", b"pesos")
            fields = adapter_fields(adapter, lora_tag="global-lora-v1")
            expected = hashlib.sha256(b"pesos").hexdigest()
            self.assertEqual(fields["adapter_sha256"], expected)
            self.assertEqual(fields["lora_tag"], "global-lora-v1")
            self.assertEqual(fields["modelVersion"], f"global-lora-v1@{expected[:8]}")
            self.assertEqual(fields["adapter"]["config"]["r"], 16)
            self.assertIn("adapter_config.json", fields["adapter"]["files"])

    def test_compose_model_version_uses_eight_hex_of_each_hash(self):
        base = {"model_sha256": "a" * 64, "base_tag": "beto-t5-base"}
        lora = {"adapter_sha256": "b" * 64, "lora_tag": "global-lora-v1"}
        self.assertEqual(compose_model_version(base, lora), "beto-t5-base@aaaaaaaa+global-lora-v1@bbbbbbbb")

    def test_compose_model_version_rejects_missing_or_bad_hashes(self):
        with self.assertRaises(ValueError):
            compose_model_version({}, {"adapter_sha256": "b" * 64})
        with self.assertRaises(ValueError):
            compose_model_version({"model_sha256": "zz"}, {"adapter_sha256": "b" * 64})

    def test_merged_manifest_extends_training_manifest_and_composes_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root)
            adapter = _fake_adapter(root / "grammar_lora", b"pesos-lora")
            merged = _fake_merged(root / "t5_new", b"pesos-fusionados")
            training = training_manifest(dataset, {"seed": 42}, now=FIXED_NOW)
            training.update(adapter_fields(adapter))
            manifest = merged_model_manifest(training, merged, adapter, now="2026-09-15T01:02:03Z")
            model_hash = hashlib.sha256(b"pesos-fusionados").hexdigest()
            adapter_hash = hashlib.sha256(b"pesos-lora").hexdigest()
            self.assertEqual(manifest["model_sha256"], model_hash)
            self.assertEqual(manifest["adapter_sha256"], adapter_hash)
            self.assertEqual(manifest["merged_at_utc"], "2026-09-15T01:02:03Z")
            self.assertEqual(manifest["dataset_sha256"], training["dataset_sha256"])
            self.assertEqual(manifest["parameters"], training["parameters"])
            self.assertEqual(manifest["modelVersion"],
                             f"beto-t5-base@{model_hash[:8]}+global-lora-v1@{adapter_hash[:8]}")
            self.assertEqual(manifest["modelVersion"], compose_model_version(manifest, training))
            # El manifiesto de entrenamiento no se muta: conserva su versión parcial.
            self.assertEqual(training["modelVersion"], f"global-lora-v1@{adapter_hash[:8]}")
            self.assertNotIn("merged_at_utc", training)
            self.assertNotIn("model_sha256", training)

    def test_merged_manifest_refuses_an_adapter_that_is_not_the_trained_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = training_manifest(_dataset(root), {"seed": 42}, now=FIXED_NOW)
            training.update(adapter_fields(_fake_adapter(root / "a1", b"uno")))
            merged = _fake_merged(root / "t5_new")
            with self.assertRaises(ValueError):
                merged_model_manifest(training, merged, _fake_adapter(root / "a2", b"dos"))


class ValidateMergedDirTests(unittest.TestCase):
    def test_complete_directory_has_no_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = _fake_adapter(root / "grammar_lora")
            merged = _fake_merged(root / "t5_new")
            training = training_manifest(_dataset(root), {"seed": 42}, now=FIXED_NOW)
            training.update(adapter_fields(adapter))
            write_json(merged / MODEL_MANIFEST_NAME, merged_model_manifest(training, merged, adapter))
            self.assertEqual(validate_merged_dir(merged), [])

    def test_missing_files_and_bad_manifest_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            merged = Path(directory) / "t5_new"
            merged.mkdir()
            (merged / "config.json").write_text("{}", encoding="utf-8")
            errors = validate_merged_dir(merged)
            self.assertTrue(any("model.safetensors" in e for e in errors))
            self.assertTrue(any("tokenizer_config.json" in e for e in errors))
            self.assertTrue(any(MODEL_MANIFEST_NAME in e for e in errors))
            _fake_merged(merged)
            (merged / MODEL_MANIFEST_NAME).write_text("{ roto", encoding="utf-8")
            errors = validate_merged_dir(merged)
            self.assertEqual(len(errors), 1)
            self.assertIn(MODEL_MANIFEST_NAME, errors[0])
            (merged / MODEL_MANIFEST_NAME).write_text(json.dumps({"model_sha256": "0" * 64}), encoding="utf-8")
            errors = validate_merged_dir(merged)
            self.assertTrue(any("model_sha256" in e for e in errors))  # hash no coincide con el archivo

    def test_tokenizer_vocabulary_is_required(self):
        # tokenizer_config.json solo no basta: sin spiece.model ni tokenizer.json
        # el pipeline no puede tokenizar. Cualquiera de los dos vale.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = _fake_adapter(root / "grammar_lora")
            training = training_manifest(_dataset(root), {"seed": 42}, now=FIXED_NOW)
            training.update(adapter_fields(adapter))
            merged = _fake_merged(root / "sin-vocab", vocab=None)
            write_json(merged / MODEL_MANIFEST_NAME, merged_model_manifest(training, merged, adapter))
            errors = validate_merged_dir(merged)
            self.assertEqual(len(errors), 1)
            self.assertIn("spiece.model", errors[0])
            self.assertIn("tokenizer.json", errors[0])
            for vocab in ("spiece.model", "tokenizer.json"):
                merged = _fake_merged(root / vocab.replace(".", "-"), vocab=vocab)
                write_json(merged / MODEL_MANIFEST_NAME, merged_model_manifest(training, merged, adapter))
                self.assertEqual(validate_merged_dir(merged), [])


class ResolveMergeBaseTests(unittest.TestCase):
    """El merge carga el base que registró el entrenamiento (nombre + revisión), nunca otro."""

    def test_base_and_revision_come_from_the_training_manifest(self):
        training = {"provenance": "trained", "base_model": BASE_MODEL, "base_model_revision": "abc123"}
        self.assertEqual(resolve_merge_base(training), (BASE_MODEL, "abc123"))
        self.assertEqual(resolve_merge_base(training, env_base=BASE_MODEL), (BASE_MODEL, "abc123"))
        self.assertEqual(resolve_merge_base(training, env_base="  "), (BASE_MODEL, "abc123"))

    def test_env_override_must_match(self):
        training = {"provenance": "trained", "base_model": BASE_MODEL, "base_model_revision": "abc123"}
        with self.assertRaises(ValueError) as ctx:
            resolve_merge_base(training, env_base="otro/modelo")
        self.assertIn("otro/modelo", str(ctx.exception))
        self.assertIn(BASE_MODEL, str(ctx.exception))

    def test_trained_manifest_without_revision_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_merge_base({"provenance": "trained", "base_model": BASE_MODEL, "base_model_revision": None})
        self.assertIn("base_model_revision", str(ctx.exception))
        with self.assertRaises(ValueError):
            resolve_merge_base({"provenance": "trained", "base_model": ""})

    def test_reconstructed_manifest_may_lack_the_revision(self):
        training = {"provenance": "reconstructed", "base_model": BASE_MODEL, "base_model_revision": None}
        self.assertEqual(resolve_merge_base(training), (BASE_MODEL, None))
        self.assertEqual(resolve_merge_base({"provenance": "reconstructed"}), (BASE_MODEL, None))


class TrainingAndMergeScriptTests(unittest.TestCase):
    """Los scripts con torch no se importan: se inspecciona el código fuente (como test_versioning con main.py)."""

    def test_training_seeds_everything_before_loading_the_base_model_and_peft(self):
        source = (REPO_DIR / "train_grammar_lora.py").read_text(encoding="utf-8")
        seeds_def = source.index("def configure_seeds(")
        seeds_call = source.index("= configure_seeds(SEED)", seeds_def)     # la llamada real, no la del docstring
        self.assertLess(seeds_call, source.index("from_pretrained(", seeds_def))
        self.assertLess(seeds_call, source.index("get_peft_model(", seeds_def))
        body = source[seeds_def:seeds_call]
        for call in ("random.seed(", "torch.manual_seed(", "torch.cuda.manual_seed_all("):
            self.assertIn(call, body)
        self.assertIn("torch.Generator", source)
        self.assertRegex(source, r"DataLoader\(PairDS\(train_pairs\)[^)]*generator=")
        self.assertIn('manifest["deterministic"]', source)

    def test_merge_uses_the_recorded_base_revision_and_fails_uniformly(self):
        source = (REPO_DIR / "scripts" / "merge_grammar_lora.py").read_text(encoding="utf-8")
        self.assertIn("resolve_merge_base(", source)
        self.assertRegex(source, r"from_pretrained\(BASE[^)]*revision=REVISION")
        self.assertNotIn('os.environ.get("BASE", "vgaraujov/t5-base-spanish")', source)
        self.assertRegex(source, r"except \(FileNotFoundError, ValueError, OSError\) as exc:\s*\n\s*sys\.exit\(f?\"\[ERROR\]")
        # El manifiesto/validación se hace sobre el temporal, antes de tocar OUT.
        self.assertLess(source.index("validate_merged_dir(TMP)"), source.index("shutil.move(TMP, OUT)"))


class WriteCurrentTests(unittest.TestCase):
    """`--write-current` reconstruye los manifiestos de los artefactos existentes sin reentrenar."""

    def _repo(self, root: Path, with_dataset=True) -> Path:
        _fake_adapter(root / "models" / "grammar_lora", b"adaptador-actual")
        _fake_merged(root / "models" / "t5_correction", b"modelo-actual")
        if with_dataset:
            (root / "data").mkdir()
            _dataset(root / "data", rows=[(f"e{i}", f"c{i}") for i in range(8)]).rename(
                root / "data" / "training_pairs_clean.csv")
        return root

    def test_reconstructed_manifest_reads_adapter_config_and_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._repo(Path(directory))
            manifest = reconstructed_training_manifest(repo, now=FIXED_NOW)
            self.assertEqual(manifest["provenance"], "reconstructed")
            self.assertEqual(manifest["parameters"]["rank"], 16)
            self.assertEqual(manifest["parameters"]["lora_alpha"], 32)
            self.assertEqual(manifest["parameters"]["seed"], 42)
            self.assertEqual(manifest["parameters"]["epochs"], TRAINING_DEFAULTS["epochs"])
            self.assertEqual(manifest["dataset"]["rows"], 8)
            self.assertEqual(manifest["adapter_sha256"], hashlib.sha256(b"adaptador-actual").hexdigest())
            self.assertEqual(manifest["libraries"]["peft"], "0.20.0")

    def test_reconstructed_manifest_without_dataset_marks_it_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._repo(Path(directory), with_dataset=False)
            manifest = reconstructed_training_manifest(repo, now=FIXED_NOW)
            self.assertIsNone(manifest["dataset_sha256"])
            self.assertIsNone(manifest["split"])

    def test_write_current_writes_both_manifests_and_returns_the_composite_version(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._repo(Path(directory))
            result = write_current(repo, now=FIXED_NOW)
            lora_path = repo / "models" / "grammar_lora" / TRAINING_MANIFEST_NAME
            base_path = repo / "models" / "t5_correction" / MODEL_MANIFEST_NAME
            self.assertTrue(lora_path.exists())
            self.assertTrue(base_path.exists())
            model_hash = hashlib.sha256(b"modelo-actual").hexdigest()
            adapter_hash = hashlib.sha256(b"adaptador-actual").hexdigest()
            expected = f"beto-t5-base@{model_hash[:8]}+global-lora-v1@{adapter_hash[:8]}"
            self.assertEqual(result["modelVersion"], expected)
            base = json.loads(base_path.read_text(encoding="utf-8"))
            lora = json.loads(lora_path.read_text(encoding="utf-8"))
            self.assertEqual(base["modelVersion"], expected)
            self.assertEqual(base["provenance"], "reconstructed")
            self.assertEqual(lora["modelVersion"], f"global-lora-v1@{adapter_hash[:8]}")
            self.assertEqual(compose_model_version(base, lora), expected)
            # Y `infrastructure.versioning` resuelve exactamente esa cadena desde el repo.
            from infrastructure.versioning import resolve_model_version
            self.assertEqual(resolve_model_version(env={}, repo_dir=repo), expected)

    def test_write_current_keeps_an_existing_training_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._repo(Path(directory))
            lora_path = repo / "models" / "grammar_lora" / TRAINING_MANIFEST_NAME
            original = {"provenance": "trained", "adapter_sha256": hashlib.sha256(b"adaptador-actual").hexdigest(),
                        "modelVersion": "global-lora-v1@deadbeef", "parameters": {"seed": 42}}
            write_json(lora_path, original)
            result = write_current(repo, now=FIXED_NOW)
            self.assertEqual(json.loads(lora_path.read_text(encoding="utf-8")), original)
            base = json.loads((repo / "models" / "t5_correction" / MODEL_MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(base["provenance"], "trained")
            self.assertTrue(result["modelVersion"].endswith(f"+global-lora-v1@{original['adapter_sha256'][:8]}"))

    def test_write_current_refuses_a_stale_training_manifest_unless_forced(self):
        """Si el adaptador cambió respecto al manifiesto guardado, no se compone una versión falsa."""
        with tempfile.TemporaryDirectory() as directory:
            repo = self._repo(Path(directory))
            lora_path = repo / "models" / "grammar_lora" / TRAINING_MANIFEST_NAME
            write_json(lora_path, {"provenance": "trained", "adapter_sha256": "1" * 64})
            with self.assertRaises(ValueError):
                write_current(repo, now=FIXED_NOW)
            result = write_current(repo, now=FIXED_NOW, force=True)
            self.assertEqual(json.loads(lora_path.read_text(encoding="utf-8"))["provenance"], "reconstructed")
            self.assertIn("modelVersion", result)

    def test_write_current_fails_clearly_without_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                write_current(Path(directory), now=FIXED_NOW)


class NoHeavyImportsTests(unittest.TestCase):
    def test_importing_model_manifest_does_not_load_torch(self):
        code = ("import sys; import scripts.model_manifest; "
                "assert 'torch' not in sys.modules and 'transformers' not in sys.modules, "
                "sorted(m for m in sys.modules if m in ('torch', 'transformers'))")
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_DIR),
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_cli_help_runs_from_the_scripts_folder(self):
        proc = subprocess.run([sys.executable, str(REPO_DIR / "scripts" / "model_manifest.py"), "--help"],
                              cwd=str(REPO_DIR), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--write-current", proc.stdout)
        self.assertIn("--base-version", proc.stdout)


class BaseVersionTests(unittest.TestCase):
    """`--base-version DIR`: versión hasheada de un T5 sin LoRA (el baseline del holdout)."""

    def test_base_version_hashes_the_weights_with_the_given_tag(self):
        with tempfile.TemporaryDirectory() as directory:
            merged = _fake_merged(Path(directory) / "t5_base", b"pesos-base")
            expected = hashlib.sha256(b"pesos-base").hexdigest()[:8]
            self.assertEqual(base_version(merged), f"t5-base@{expected}")
            self.assertEqual(base_version(merged, tag="otro"), f"otro@{expected}")
            with self.assertRaises(FileNotFoundError):
                base_version(Path(directory) / "no-existe")
            proc = subprocess.run([sys.executable, str(REPO_DIR / "scripts" / "model_manifest.py"),
                                   "--base-version", str(merged)], cwd=str(REPO_DIR), capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout.strip(), f"t5-base@{expected}")


class ShowVersionTests(unittest.TestCase):
    """`--show` falla cerrado (código 1) si el manifiesto fusionado existe pero es inválido."""

    def test_show_reports_an_invalid_merged_manifest_instead_of_a_partial_version(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            base_dir = repo / "models" / "t5_correction"
            base_dir.mkdir(parents=True)
            (base_dir / MODEL_MANIFEST_NAME).write_text(json.dumps({"model_sha256": "0" * 64}), encoding="utf-8")
            lora_dir = repo / "models" / "grammar_lora"
            lora_dir.mkdir(parents=True)
            (lora_dir / TRAINING_MANIFEST_NAME).write_text(
                json.dumps({"adapter_sha256": "b" * 64, "modelVersion": "global-lora-v1@bbbbbbbb"}), encoding="utf-8")
            proc = subprocess.run([sys.executable, str(REPO_DIR / "scripts" / "model_manifest.py"),
                                   "--repo-dir", str(repo), "--show"], cwd=str(REPO_DIR), capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("[ERROR]", proc.stdout)
            self.assertNotIn("global-lora-v1@bbbbbbbb", proc.stdout)


if __name__ == "__main__":
    unittest.main()
