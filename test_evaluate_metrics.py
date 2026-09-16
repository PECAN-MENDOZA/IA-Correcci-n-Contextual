"""
test_evaluate_metrics.py

Tests puros del evaluador (evaluate.py): scorer de ediciones exactas
`exact_token_edits_v1` (Precisión / Recall / F0.5 a nivel de edición con
alineación por token), lectura del dataset con varias referencias, informe
versionado (las claves del contrato en el nivel superior, tal como las valida
el panel Vue y las acepta el backend), errores del modelo visibles y la guarda
de setup_model.py contra sobrescribir el checkpoint global.

Corren SIN torch/transformers/GPU ni modelos: `evaluate` y `setup_model` solo
importan la librería estándar a nivel de módulo (los modelos se cargan dentro
de `main()` / `build_predictor`).

Uso:
    .venv\\Scripts\\python.exe -m unittest test_evaluate_metrics -v
"""
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluate import (
    DEVELOPMENT_DATASETS,
    MAX_CATEGORIES,
    MAX_CATEGORY_NAME,
    MAX_MODEL_VERSION,
    MODEL_MANIFEST_NAME,
    REPO_DIR,
    SCORER_VERSION,
    align_tokens,
    build_report,
    dataset_sha256,
    edit_scores,
    ensure_offline,
    evaluation_model_version,
    exit_status,
    extract_edits,
    git_commit,
    is_development,
    load_dataset,
    pipeline_info,
    prf,
    report_pipeline_blocks,
    require_local_t5_dir,
    resolve_model_version,
    score_sentence,
    validate_model_version,
)
import setup_model


# ───────────────────────────── réplica del validador del panel ─────────────────────────────
# Reimplementación pura de `technicalEvaluationErrors` de
# frontend-web/src/features/research/utils/results.js (que a su vez repite las
# reglas del backend). Se aplica al INFORME COMPLETO, como hace el panel con el
# archivo subido: el contrato vive en el nivel superior del JSON.

PANEL_SCORER_VERSION = "exact_token_edits_v1"
PANEL_RATE_TOLERANCE = 1e-6
PANEL_MODEL_VERSION_MAX_LENGTH = 160
PANEL_CATEGORY_NAME_MAX_LENGTH = 80
PANEL_CATEGORIES_MAX = 50
PANEL_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _is_text(value, max_length) -> bool:
    return isinstance(value, str) and 0 < len(value.strip()) <= max_length


def _is_rate(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value) and 0 <= value <= 1


def _is_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _matches_ratio(reported, numerator, denominator) -> bool:
    if denominator == 0:
        return reported == 0
    return abs(numerator / denominator - reported) <= PANEL_RATE_TOLERANCE


def _f_zero_five(precision, recall) -> float:
    denominator = 0.25 * precision + recall
    return 0 if denominator == 0 else 1.25 * precision * recall / denominator


def _vector_errors(prefix, precision, recall, f05, tp, fp, fn, names) -> list:
    errors = []
    for key, value in (("precision", precision), ("recall", recall), ("f05", f05)):
        if not _is_rate(value):
            errors.append(f"{prefix}{names[key]} debe ser un número entre 0 y 1.")
    for key, value in (("tp", tp), ("fp", fp), ("fn", fn)):
        if not _is_count(value):
            errors.append(f"{prefix}{names[key]} debe ser un entero mayor o igual a 0.")
    if errors:
        return errors
    if not _matches_ratio(precision, tp, tp + fp):
        errors.append(f"{prefix}{names['precision']} no coincide con TP / (TP + FP).")
    if not _matches_ratio(recall, tp, tp + fn):
        errors.append(f"{prefix}{names['recall']} no coincide con TP / (TP + FN).")
    expected = _f_zero_five(precision, recall)
    if abs(expected - f05) > PANEL_RATE_TOLERANCE:
        errors.append(f"{prefix}{names['f05']} no coincide con {names['precision']} y {names['recall']}.")
    return errors


def _categories_errors(categories) -> list:
    if not isinstance(categories, list) or len(categories) > PANEL_CATEGORIES_MAX:
        return [f"categories debe ser una lista de hasta {PANEL_CATEGORIES_MAX} categorías."]
    errors, seen = [], set()
    for index, category in enumerate(categories):
        if not isinstance(category, dict):
            errors.append(f"Categoría #{index + 1}: debe ser un objeto.")
            continue
        name = category.get("category").strip() if isinstance(category.get("category"), str) else ""
        prefix = f"Categoría '{name}': " if name else f"Categoría #{index + 1}: "
        if not _is_text(category.get("category"), PANEL_CATEGORY_NAME_MAX_LENGTH):
            errors.append(f"{prefix}category es obligatorio (texto de hasta "
                          f"{PANEL_CATEGORY_NAME_MAX_LENGTH} caracteres).")
        elif name in seen:
            errors.append(f"Categoría '{name}' repetida.")
        else:
            seen.add(name)
        errors.extend(_vector_errors(
            prefix, category.get("precision"), category.get("recall"), category.get("f05"),
            category.get("tp"), category.get("fp"), category.get("fn"),
            {"precision": "precision", "recall": "recall", "f05": "f05", "tp": "tp", "fp": "fp", "fn": "fn"},
        ))
    return errors


def panel_errors(report) -> list:
    """Réplica de `technicalEvaluationErrors(report)` sobre el informe completo."""
    if not isinstance(report, dict):
        return ["El archivo debe contener un objeto JSON con el informe de la evaluación."]
    errors = []
    if not _is_text(report.get("modelVersion"), PANEL_MODEL_VERSION_MAX_LENGTH):
        errors.append("modelVersion es obligatorio (texto de hasta 160 caracteres).")
    sha = report.get("datasetSha256")
    if not isinstance(sha, str) or not PANEL_SHA256_PATTERN.match(sha):
        errors.append("datasetSha256 debe ser un SHA-256 en hexadecimal minúsculas (64 caracteres).")
    if report.get("scorerVersion") != PANEL_SCORER_VERSION:
        errors.append(f"scorerVersion debe ser {PANEL_SCORER_VERSION}.")
    errors.extend(_vector_errors(
        "", report.get("precision"), report.get("recall"), report.get("fZeroFive"),
        report.get("truePositives"), report.get("falsePositives"), report.get("falseNegatives"),
        {"precision": "precision", "recall": "recall", "f05": "fZeroFive",
         "tp": "truePositives", "fp": "falsePositives", "fn": "falseNegatives"},
    ))
    if report.get("categories") is not None:
        errors.extend(_categories_errors(report["categories"]))
    return errors


class PanelReplicaSelfTests(unittest.TestCase):
    """La réplica detecta lo mismo que el panel real detectó en la revisión (H1)."""

    def test_nested_only_report_is_rejected_with_six_errors(self):
        nested = {"modelVersion": "v", "datasetSha256": "a" * 64, "scorerVersion": PANEL_SCORER_VERSION,
                  "technicalEvaluation": {"precision": 1.0, "recall": 1.0, "fZeroFive": 1.0,
                                          "truePositives": 1, "falsePositives": 0, "falseNegatives": 0,
                                          "categories": []}}
        self.assertEqual(len(panel_errors(nested)), 6)

    def test_inconsistent_rates_and_bad_categories_are_rejected(self):
        bad = {"modelVersion": " ", "datasetSha256": "A" * 64, "scorerVersion": "otro",
               "precision": 0.5, "recall": 1.0, "fZeroFive": 1.0,
               "truePositives": 1, "falsePositives": 0, "falseNegatives": 0.5,
               "categories": [{"category": "x", "tp": 1, "fp": 0, "fn": 0,
                               "precision": 1.0, "recall": 1.0, "f05": 1.0},
                              {"category": " x ", "tp": 1, "fp": 0, "fn": 0,
                               "precision": 1.0, "recall": 1.0, "f05": 1.0}]}
        errors = panel_errors(bad)
        self.assertTrue(any("modelVersion" in e for e in errors))
        self.assertTrue(any("datasetSha256" in e for e in errors))
        self.assertTrue(any("scorerVersion" in e for e in errors))
        self.assertTrue(any("falseNegatives" in e for e in errors))
        self.assertTrue(any("repetida" in e for e in errors))


# ───────────────────────────── scorer ─────────────────────────────

class EditScoresTests(unittest.TestCase):
    def test_exact_gold_is_perfect(self):
        result = edit_scores(["el niño iva"], [["el niño iba"]], ["el niño iba"])
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["fn"], 0)
        self.assertEqual(result["f0_5"], 1.0)

    def test_unnecessary_change_is_false_positive(self):
        result = edit_scores(["el niño iba"], [["el niño iba"]], ["el niño fue"])
        self.assertEqual(result["tp"], 0)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 0)
        # Convención del backend: sin verdaderos positivos P = R = F0.5 = 0.
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["f0_5"], 0.0)

    def test_best_valid_reference_is_used(self):
        result = edit_scores(["hola"], [["Hola.", "Hola"]], ["Hola"])
        self.assertEqual(result["f0_5"], 1.0)

    def test_missed_correction_is_false_negative(self):
        result = edit_scores(["el niño iva"], [["el niño iba"]], ["el niño iva"])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 0, 1))
        # Denominador de la precisión = 0 → 0.0 (no 1.0), como valida el backend.
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["f0_5"], 0.0)

    def test_untouched_correct_sentence_has_no_counts(self):
        result = edit_scores(["la casa es grande"], [["la casa es grande"]], ["la casa es grande"])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 0, 0))
        self.assertEqual((result["precision"], result["recall"], result["f0_5"]), (0.0, 0.0, 0.0))

    def test_counts_aggregate_over_the_corpus(self):
        inputs = ["el niño iva", "boy a la escuela manana", "la casa es grande"]
        golds = [["el niño iba"], ["voy a la escuela mañana"], ["la casa es grande"]]
        preds = ["el niño iba", "voy a la escuela manana", "la casa es linda"]
        result = edit_scores(inputs, golds, preds)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (2, 1, 1))
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["f0_5"], 2 / 3)
        self.assertEqual(result["n"], 3)

    def test_partial_span_is_not_a_true_positive(self):
        # La edición debe coincidir exactamente en posición y reemplazo.
        result = edit_scores(["fui ala tienda"], [["fui a la tienda"]], ["fui a tienda"])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))

    def test_scorer_is_identified(self):
        self.assertEqual(SCORER_VERSION, "exact_token_edits_v1")
        result = edit_scores([], [], [])
        self.assertEqual(result["scorer"], "exact_token_edits_v1")
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 0, 0))

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            edit_scores(["a"], [["a"]], [])
        with self.assertRaises(ValueError):
            edit_scores(["a"], [[]], ["a"])

    def test_empty_prediction_is_one_deletion_false_positive(self):
        result = edit_scores(["el niño iva"], [["el niño iba"]], [""])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))
        self.assertEqual(extract_edits("el niño iva", ""), {(0, 3, ())})


class ScoreSentenceReferenceChoiceTests(unittest.TestCase):
    # 20 tokens; la predicción pone en mayúscula los 10 primeros (10 ediciones 1:1).
    SOURCE = " ".join("abcdefghijklmnopqrst")
    PREDICTION = " ".join("ABCDEFGHIJklmnopqrst")
    REF_ALL = " ".join("ABCDEFGHIJKLMNOPQRST")          # tp=10 fp=0 fn=10 → P=1, R=0.5
    REF_NINE = " ".join("ABCDEFGHIjKLMNOpqrst")         # tp=9 fp=1 fn=5 → P=0.9, R=9/14

    def test_f05_tie_is_broken_by_more_true_positives(self):
        # Ambas referencias dan exactamente el mismo F0.5 (5/6) con TP distinto.
        self.assertEqual(prf(10, 0, 10)[2], prf(9, 1, 5)[2])
        for references in ([self.REF_NINE, self.REF_ALL], [self.REF_ALL, self.REF_NINE]):
            chosen = score_sentence(self.SOURCE, references, self.PREDICTION)
            self.assertEqual(chosen["gold"], self.REF_ALL)
            self.assertEqual((chosen["tp"], chosen["fp"], chosen["fn"]), (10, 0, 10))

    def test_full_tie_keeps_the_first_reference(self):
        # Referencias con los mismos conteos: se conserva la primera.
        chosen = score_sentence("el niño iva", ["el niño iba", "el niño ibа"], "el niño iva")
        self.assertEqual(chosen["gold"], "el niño iba")
        self.assertEqual((chosen["tp"], chosen["fp"], chosen["fn"]), (0, 0, 1))
        # Con F0.5 y TP iguales (0), gana la referencia con menos FP+FN.
        chosen = score_sentence("hola", ["Hola.", "Hola"], "hola")
        self.assertEqual(chosen["gold"], "Hola")

    def test_reference_is_required(self):
        with self.assertRaises(ValueError):
            score_sentence("a", [], "a")


class PrfTests(unittest.TestCase):
    def test_f05_weighs_precision_over_recall(self):
        precision, recall, f0_5 = prf(3, 1, 3)
        self.assertAlmostEqual(precision, 0.75)
        self.assertAlmostEqual(recall, 0.5)
        self.assertAlmostEqual(f0_5, 1.25 * 0.75 * 0.5 / (0.25 * 0.75 + 0.5))

    def test_zero_denominators_give_zero(self):
        self.assertEqual(prf(0, 0, 0), (0.0, 0.0, 0.0))
        self.assertEqual(prf(0, 2, 0), (0.0, 0.0, 0.0))
        self.assertEqual(prf(0, 0, 2), (0.0, 0.0, 0.0))

    def test_values_are_reproducible_from_counts_within_backend_tolerance(self):
        precision, recall, f0_5 = prf(7, 3, 5)
        self.assertLessEqual(abs(precision - 7 / 10), 1e-6)
        self.assertLessEqual(abs(recall - 7 / 12), 1e-6)
        expected = 1.25 * precision * recall / (0.25 * precision + recall)
        self.assertLessEqual(abs(f0_5 - expected), 1e-6)


class ExtractEditsTests(unittest.TestCase):
    def test_identical_text_has_no_edits(self):
        self.assertEqual(extract_edits("el niño iba", "el niño iba"), set())

    def test_punctuation_is_its_own_token_and_its_own_edit(self):
        # Mayúscula inicial y punto final son dos correcciones distintas.
        self.assertEqual(extract_edits("hola", "Hola."), {(0, 1, ("Hola",)), (1, 1, (".",))})

    def test_split_word_is_one_replacement_span(self):
        self.assertEqual(extract_edits("fui ala tienda", "fui a la tienda"), {(1, 2, ("a", "la"))})

    def test_joined_words_are_one_edit(self):
        self.assertEqual(extract_edits("fui a la tienda", "fui ala tienda"), {(1, 3, ("ala",))})

    def test_deletion_has_empty_replacement(self):
        edits = extract_edits("el niño niño come", "el niño come")
        self.assertEqual(len(edits), 1)
        (start, end, replacement), = edits
        self.assertEqual(end - start, 1)
        self.assertEqual(replacement, ())

    def test_accent_change_is_a_replacement(self):
        self.assertEqual(extract_edits("el sabado voy", "el sábado voy"), {(1, 2, ("sábado",))})

    def test_adjacent_one_to_one_replacements_are_separate_edits(self):
        self.assertEqual(extract_edits("el arbol esta lleno", "el árbol está lleno"),
                         {(1, 2, ("árbol",)), (2, 3, ("está",))})

    def test_unequal_length_replacement_stays_one_span(self):
        self.assertEqual(extract_edits("los uillos juegan", "los niños juegan"), {(1, 2, ("niños",))})
        self.assertEqual(extract_edits("delas frutas", "de las frutas"), {(0, 1, ("de", "las"))})

    def test_insertion_next_to_corrections_is_its_own_edit(self):
        # M1: una inserción vecina no absorbe las tildes corregidas.
        self.assertEqual(extract_edits("el arbol esta lleno", "el árbol está muy lleno"),
                         {(1, 2, ("árbol",)), (2, 3, ("está",)), (3, 3, ("muy",))})

    def test_substitution_next_to_split_is_its_own_edit(self):
        self.assertEqual(extract_edits("yo iva ala tienda", "yo iba a la tienda"),
                         {(1, 2, ("iba",)), (2, 3, ("a", "la"))})

    def test_transposition_like_swap_is_two_substitutions(self):
        self.assertEqual(extract_edits("la a", "a la"), {(0, 1, ("a",)), (1, 2, ("la",))})

    def test_consecutive_insertions_or_deletions_merge_into_one_edit(self):
        self.assertEqual(extract_edits("casa", "casa muy grande"), {(1, 1, ("muy", "grande"))})
        self.assertEqual(extract_edits("casa muy grande", "casa"), {(1, 3, ())})

    def test_unrelated_one_to_one_replacement_is_a_substitution(self):
        self.assertEqual(extract_edits("el niño iba", "el niño fue"), {(2, 3, ("fue",))})

    def test_two_word_split_next_to_insertion_stays_apart(self):
        # `estámuy` no es la unión de `esta`: la división 1:m solo se alinea
        # cuando las letras coinciden (sin tildes ni mayúsculas).
        edits = extract_edits("esta lleno", "está muy lleno")
        self.assertEqual(edits, {(0, 1, ("está",)), (1, 1, ("muy",))})

    def test_alignment_is_deterministic(self):
        pairs = [("la a", "a la"), ("el niño niño come", "el niño come"), ("xyz", "foo bar")]
        for source, target in pairs:
            first = extract_edits(source, target)
            for _ in range(5):
                self.assertEqual(extract_edits(source, target), first)


class AlignTokensTests(unittest.TestCase):
    def test_costs_follow_the_documented_rules(self):
        ops = align_tokens(["el", "arbol", "esta", "lleno"], ["el", "árbol", "está", "muy", "lleno"])
        self.assertEqual([op[0] for op in ops], ["match", "sub", "sub", "ins", "match"])

    def test_split_and_join_are_single_ops(self):
        self.assertEqual([op[0] for op in align_tokens(["ala"], ["a", "la"])], ["split"])
        self.assertEqual([op[0] for op in align_tokens(["a", "la"], ["ala"])], ["join"])
        self.assertEqual([op[0] for op in align_tokens(["alavez"], ["a", "la", "vez"])], ["split"])

    def test_far_tokens_do_not_split(self):
        self.assertEqual([op[0] for op in align_tokens(["esta"], ["está", "muy"])], ["sub", "ins"])


class AdjacentEditScoringTests(unittest.TestCase):
    def test_partial_fix_next_to_missed_word_is_tp_plus_fn(self):
        result = edit_scores(["el arbol esta lleno de hojas"], [["el árbol está lleno de hojas"]],
                             ["el árbol esta lleno de hojas"])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 1))

    def test_extra_insertion_does_not_erase_true_positives(self):
        result = edit_scores(["el arbol esta lleno"], [["el árbol está lleno"]], ["el árbol está muy lleno"])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (2, 1, 0))

    def test_split_next_to_substitution_full_and_partial(self):
        full = edit_scores(["yo iva ala tienda"], [["yo iba a la tienda"]], ["yo iba a la tienda"])
        self.assertEqual((full["tp"], full["fp"], full["fn"]), (2, 0, 0))
        partial = edit_scores(["yo iva ala tienda"], [["yo iba a la tienda"]], ["yo iba ala tienda"])
        self.assertEqual((partial["tp"], partial["fp"], partial["fn"]), (1, 0, 1))

    def test_scoring_is_monotone_when_more_is_corrected(self):
        source, gold = "la niña ba ala escuela", ["la niña va a la escuela"]
        less = edit_scores([source], [gold], ["la niña va ala escuela"])
        more = edit_scores([source], [gold], ["la niña va a la escuela"])
        self.assertEqual((less["tp"], less["fp"], less["fn"]), (1, 0, 1))
        self.assertEqual((more["tp"], more["fp"], more["fn"]), (2, 0, 0))
        self.assertGreater(more["f0_5"], less["f0_5"])


# ───────────────────────────── dataset ─────────────────────────────

class LoadDatasetTests(unittest.TestCase):
    def _write(self, text: str) -> str:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
        tmp.write(text)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return tmp.name

    def test_reads_category_input_and_multiple_references(self):
        path = self._write(
            "# comentario\n"
            "categoria|entrada|esperado\n"
            "tilde|el sabado voy|el sábado voy\n"
            "\n"
            "punt|hola|Hola.|Hola\n"
        )
        rows = load_dataset(path)
        self.assertEqual(rows, [
            ("tilde", "el sabado voy", ["el sábado voy"]),
            ("punt", "hola", ["Hola.", "Hola"]),
        ])

    def test_rejects_fewer_than_three_fields(self):
        path = self._write("tilde|solo dos campos\n")
        with self.assertRaises(ValueError):
            load_dataset(path)

    def test_rejects_row_without_reference(self):
        path = self._write("tilde|entrada| \n")
        with self.assertRaises(ValueError):
            load_dataset(path)

    def test_sha256_is_over_exact_bytes(self):
        content = "categoria|entrada|esperado\ntilde|el sabado|el sábado\n"
        path = self._write(content)
        expected = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        digest = dataset_sha256(path)
        self.assertEqual(digest, expected)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")


class DevelopmentFlagTests(unittest.TestCase):
    def test_development_datasets_are_detected_by_name(self):
        self.assertEqual(DEVELOPMENT_DATASETS, {"eval_gold.csv", "pruebas.txt"})
        self.assertTrue(is_development("data/eval_gold.csv", False))
        self.assertTrue(is_development(r"C:\otro\pruebas.txt", False))
        self.assertFalse(is_development("data/holdout.csv", False))

    def test_flag_forces_development(self):
        self.assertTrue(is_development("data/holdout.csv", True))


# ───────────────────────────── versionado y guardas ─────────────────────────────

class ResolveModelVersionTests(unittest.TestCase):
    """`evaluate.resolve_model_version` es la de `infrastructure/versioning.py`."""

    def test_explicit_version_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(resolve_model_version("t5@abc", {"MODEL_VERSION": "env"}, Path(tmp)), "t5@abc")

    def test_env_beats_manifest_and_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora = Path(tmp) / "models" / "grammar_lora"
            lora.mkdir(parents=True)
            (lora / "manifest.json").write_text(json.dumps({"modelVersion": "base@1+lora@2"}), encoding="utf-8")
            self.assertEqual(resolve_model_version(None, {"MODEL_VERSION": "env-v"}, Path(tmp)), "env-v")
            self.assertEqual(resolve_model_version(None, {}, Path(tmp)), "base@1+lora@2")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(resolve_model_version(None, {}, Path(tmp)), "global-lora-unversioned")


class EvaluationModelVersionTests(unittest.TestCase):
    """La versión del informe describe el pipeline evaluado, nunca los manifiestos de `models/` por defecto."""

    COMPOSED = "beto-t5-base@01234567+global-lora-v1@fedcba98"

    def _t5_dir(self, tmp, manifest=None, name="t5"):
        t5_dir = Path(tmp) / name
        t5_dir.mkdir()
        if manifest is not None:
            (t5_dir / MODEL_MANIFEST_NAME).write_text(
                manifest if isinstance(manifest, str) else json.dumps(manifest), encoding="utf-8")
        return t5_dir

    def test_explicit_flag_and_env_win_in_that_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            t5_dir = self._t5_dir(tmp, {"model_sha256": "0123456789abcdef" * 4, "adapter_sha256": "fedcba9876543210" * 4})
            self.assertEqual(evaluation_model_version(" flag ", source="pipeline", t5_dir=t5_dir, no_beto=False,
                                                      env={"MODEL_VERSION": "env-v"}), "flag")
            self.assertEqual(evaluation_model_version(None, source="pipeline", t5_dir=t5_dir, no_beto=False,
                                                      env={"MODEL_VERSION": "env-v"}), "env-v")
            self.assertEqual(evaluation_model_version("flag", source="pipeline", t5_dir=None, no_beto=True, env={}), "flag")
            self.assertEqual(evaluation_model_version("flag", source="http", t5_dir=None, no_beto=False, env={},
                                                      server_version="srv"), "flag")

    def test_full_pipeline_composes_from_the_evaluated_t5_dir_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            t5_dir = self._t5_dir(tmp, {"model_sha256": "0123456789abcdef" * 4, "adapter_sha256": "fedcba9876543210" * 4})
            self.assertEqual(evaluation_model_version(None, source="pipeline", t5_dir=t5_dir, no_beto=False, env={}),
                             self.COMPOSED)
            # Un `--t5-dir` distinto de models/t5_correction usa SU manifiesto, no el del repo.
            other = self._t5_dir(tmp, {"model_sha256": "a" * 64, "adapter_sha256": "b" * 64, "base_tag": "t5-base",
                                       "lora_tag": "sin-lora"}, name="otro")
            self.assertEqual(evaluation_model_version(None, source="pipeline", t5_dir=other, no_beto=False, env={}),
                             "t5-base@aaaaaaaa+sin-lora@bbbbbbbb")

    def test_t5_dir_without_readable_manifest_requires_an_explicit_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            for index, manifest in enumerate((None, "{ roto", {"model_sha256": "a" * 64})):
                t5_dir = self._t5_dir(tmp, manifest, name=f"t5-{index}")
                with self.assertRaises(ValueError) as ctx:
                    evaluation_model_version(None, source="pipeline", t5_dir=t5_dir, no_beto=False, env={})
                self.assertIn("--model-version", str(ctx.exception))
                self.assertIn(MODEL_MANIFEST_NAME, str(ctx.exception))

    def test_rules_only_and_no_beto_require_an_explicit_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            t5_dir = self._t5_dir(tmp, {"model_sha256": "0123456789abcdef" * 4, "adapter_sha256": "fedcba9876543210" * 4})
            for kwargs in (dict(t5_dir=None, no_beto=False), dict(t5_dir=None, no_beto=True),
                           dict(t5_dir=t5_dir, no_beto=True)):
                with self.assertRaises(ValueError) as ctx:
                    evaluation_model_version(None, source="pipeline", env={}, **kwargs)
                self.assertIn("--model-version", str(ctx.exception))

    def test_http_takes_the_server_version(self):
        self.assertEqual(evaluation_model_version(None, source="http", t5_dir=None, no_beto=False, env={},
                                                  server_version=" srv@1 "), "srv@1")
        with self.assertRaises(ValueError) as ctx:
            evaluation_model_version(None, source="http", t5_dir=None, no_beto=False, env={}, server_version=None)
        self.assertIn("modelVersion", str(ctx.exception))

    def test_cli_exits_2_without_version_for_rules_only_before_loading_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "mini.csv"
            dataset.write_text("tilde|el arbol|el árbol\n", encoding="utf-8")
            proc = subprocess.run([sys.executable, str(REPO_DIR / "evaluate.py"), "--no-beto", "--dataset", str(dataset)],
                                  cwd=str(REPO_DIR), capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("--model-version", proc.stdout + proc.stderr)


class ValidateModelVersionTests(unittest.TestCase):
    def test_accepts_trimmed_text_up_to_160_chars(self):
        self.assertEqual(validate_model_version("  t5@abcd1234+grammar@ef567890 "), "t5@abcd1234+grammar@ef567890")
        self.assertEqual(len(validate_model_version("x" * MAX_MODEL_VERSION)), 160)

    def test_rejects_blank_or_too_long(self):
        for bad in ("", "   ", None, "x" * (MAX_MODEL_VERSION + 1)):
            with self.assertRaises(ValueError):
                validate_model_version(bad)


class RequireLocalT5DirTests(unittest.TestCase):
    def test_accepts_directory_with_config_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text("{}", encoding="utf-8")
            (Path(tmp) / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(require_local_t5_dir(tmp), Path(tmp))

    def test_rejects_missing_files_before_any_heavy_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                require_local_t5_dir(tmp)
            (Path(tmp) / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit):
                require_local_t5_dir(tmp)
        with self.assertRaises(SystemExit):
            require_local_t5_dir(str(Path(tmp) / "no-existe"))


class OfflineAndExitTests(unittest.TestCase):
    def test_ensure_offline_sets_hf_hub_offline_without_overriding(self):
        env = {}
        ensure_offline(env)
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        env = {"HF_HUB_OFFLINE": "0"}
        ensure_offline(env)
        self.assertEqual(env["HF_HUB_OFFLINE"], "0")

    def test_exit_status_is_two_only_for_final_reports_with_errors(self):
        self.assertEqual(exit_status(errors=0, development=False), 0)
        self.assertEqual(exit_status(errors=0, development=True), 0)
        self.assertEqual(exit_status(errors=3, development=True), 0)
        self.assertEqual(exit_status(errors=1, development=False), 2)


# ───────────────────────────── informe ─────────────────────────────

def _case(cat, inp, golds, pred, error=None):
    case = {"cat": cat, "input": inp, "golds": golds, "pred": pred}
    if error is not None:
        case["error"] = error
    return case


class BuildReportTests(unittest.TestCase):
    SHA = "a" * 64
    CONTRACT_KEYS = {
        "modelVersion", "datasetSha256", "scorerVersion", "precision", "recall", "fZeroFive",
        "truePositives", "falsePositives", "falseNegatives", "categories",
    }

    def _report(self, cases=None, **kw):
        cases = cases if cases is not None else [
            _case("tilde", "el sabado voy", ["el sábado voy"], "el sábado voy"),
            _case("tilde", "el arbol", ["el árbol"], "el arbol"),
            _case("correcto", "la casa es grande", ["la casa es grande"], "la casa es linda"),
        ]
        params = dict(model_version="t5@abc", dataset_path="data/x.csv", dataset_hash=self.SHA,
                      model_dir=None, development=True, source="reglas", latencies_ms=[1, 2, 3])
        params.update(kw)
        return build_report(cases, **params)

    def test_contract_keys_are_at_the_top_level(self):
        report = self._report()
        self.assertEqual(report["modelVersion"], "t5@abc")
        self.assertEqual(report["datasetSha256"], self.SHA)
        self.assertEqual(report["scorerVersion"], "exact_token_edits_v1")
        self.assertEqual((report["truePositives"], report["falsePositives"], report["falseNegatives"]), (1, 1, 1))
        self.assertAlmostEqual(report["precision"], 0.5)
        self.assertAlmostEqual(report["recall"], 0.5)
        self.assertAlmostEqual(report["fZeroFive"], 0.5)
        self.assertTrue(self.CONTRACT_KEYS <= set(report))

    def test_nested_block_mirrors_the_top_level(self):
        report = self._report()
        te = report["technicalEvaluation"]
        self.assertEqual(set(te), self.CONTRACT_KEYS)
        for key in self.CONTRACT_KEYS:
            self.assertEqual(te[key], report[key], key)

    def test_whole_report_passes_the_panel_validator(self):
        self.assertEqual(panel_errors(self._report()), [])
        # También tras pasar por JSON, como el archivo que sube el panel.
        roundtrip = json.loads(json.dumps(self._report(), ensure_ascii=False))
        self.assertEqual(panel_errors(roundtrip), [])
        # Y con muchas categorías, todas con conteos propios.
        many = self._report([_case(f"c{i}", "a b", ["a c"], "a c" if i % 2 else "a b") for i in range(50)])
        self.assertEqual(panel_errors(many), [])

    def test_categories_follow_dataset_order_with_own_counts(self):
        cats = self._report()["categories"]
        self.assertEqual([c["category"] for c in cats], ["tilde", "correcto"])
        tilde, correcto = cats
        self.assertEqual((tilde["tp"], tilde["fp"], tilde["fn"]), (1, 0, 1))
        self.assertEqual((tilde["precision"], tilde["recall"]), (1.0, 0.5))
        self.assertAlmostEqual(tilde["f05"], 1.25 * 1.0 * 0.5 / (0.25 + 0.5))
        self.assertEqual((correcto["tp"], correcto["fp"], correcto["fn"]), (0, 1, 0))
        self.assertEqual((correcto["precision"], correcto["recall"], correcto["f05"]), (0.0, 0.0, 0.0))
        for c in cats:
            self.assertEqual(set(c), {"category", "tp", "fp", "fn", "precision", "recall", "f05"})

    def test_category_names_are_stripped_truncated_and_merged(self):
        self.assertEqual(MAX_CATEGORY_NAME, 80)
        # ` c1 ` y `c1` son la misma categoría; el recorte a 80 no deja un
        # espacio final (`x`*79 + ` y` → `x`*79), como harán panel y backend.
        cases = [_case(" c1 ", "a b", ["a c"], "a c"), _case("c1", "b", ["b"], "b"),
                 _case(" " + "x" * 79 + " y", "a", ["a"], "a")]
        cats = self._report(cases)["categories"]
        self.assertEqual([c["category"] for c in cats], ["c1", "x" * 79])
        self.assertEqual((cats[0]["tp"], cats[0]["fp"], cats[0]["fn"]), (1, 0, 0))
        self.assertTrue(all(c["category"] == c["category"].strip() for c in cats))

    def test_more_than_fifty_categories_is_an_error(self):
        self.assertEqual(MAX_CATEGORIES, 50)
        cases = [_case(f"c{i}", "a", ["a"], "a") for i in range(51)]
        with self.assertRaises(ValueError):
            self._report(cases)
        self.assertEqual(len(self._report(cases[:50])["categories"]), 50)

    def test_model_version_is_validated(self):
        with self.assertRaises(ValueError):
            self._report(model_version="  ")
        with self.assertRaises(ValueError):
            self._report(model_version="v" * 161)
        self.assertEqual(self._report(model_version=" v1 ")["modelVersion"], "v1")

    def test_report_is_versioned_and_labelled(self):
        report = self._report(model_dir="models/t5_correction")
        self.assertTrue(report["development"])
        self.assertEqual(report["scorerVersion"], "exact_token_edits_v1")
        self.assertEqual(report["modelVersion"], "t5@abc")
        self.assertEqual(report["modelDir"], "models/t5_correction")
        self.assertEqual(report["datasetSha256"], self.SHA)
        self.assertEqual(report["dataset"], "data/x.csv")
        self.assertIn("timestamp", report)
        self.assertEqual(report["latency_ms_avg"], 2)
        self.assertEqual(report["global"]["n"], 3)
        self.assertIn("wer", report["global"])
        self.assertIn("exact", report["global"])
        self.assertIn("f0_5", report["global"])
        self.assertEqual(len(report["casos"]), 3)
        self.assertEqual(report["casos"][0]["gold"], "el sábado voy")
        self.assertEqual(report["casos"][0]["exact"], True)
        self.assertEqual((report["casos"][1]["tp"], report["casos"][1]["fp"], report["casos"][1]["fn"]), (0, 0, 1))

    def test_exact_accepts_any_reference(self):
        report = self._report([_case("punt", "hola", ["Hola.", "Hola"], "Hola")])
        self.assertEqual(report["global"]["exact"], 1.0)
        self.assertEqual(report["casos"][0]["gold"], "Hola")

    def test_model_errors_are_visible_and_still_scored(self):
        cases = [
            _case("tilde", "el sabado voy", ["el sábado voy"], "el sabado voy", error="RuntimeError: CUDA"),
            _case("tilde", "el arbol", ["el árbol"], "el árbol"),
        ]
        report = self._report(cases)
        self.assertEqual(report["errors"], 1)
        self.assertEqual(report["errorList"], [{"cat": "tilde", "input": "el sabado voy", "error": "RuntimeError: CUDA"}])
        self.assertEqual(report["casos"][0]["error"], "RuntimeError: CUDA")
        self.assertNotIn("error", report["casos"][1])
        # El caso fallido se puntúa como "sin cambios": un FN.
        self.assertEqual((report["truePositives"], report["falsePositives"], report["falseNegatives"]), (1, 0, 1))
        self.assertEqual(panel_errors(report), [])

    def test_report_without_errors_says_zero(self):
        report = self._report()
        self.assertEqual(report["errors"], 0)
        self.assertEqual(report["errorList"], [])

    def test_report_is_json_serializable(self):
        json.dumps(self._report(), ensure_ascii=False)

    def test_report_records_pipeline_thresholds_and_commit(self):
        # `modelVersion` identifica pesos, no código: el informe registra los
        # umbrales de la capa 5 y el commit para que dos informes con la misma
        # versión y reglas distintas se distingan.
        from infrastructure.nlp.alternatives import THRESHOLDS
        report = self._report(pipeline=pipeline_info(commit="abc1234"))
        self.assertEqual(report["pipeline"]["thresholds"], THRESHOLDS)
        self.assertEqual(report["pipeline"]["commit"], "abc1234")
        self.assertEqual(panel_errors(report), [])
        info = pipeline_info()
        self.assertTrue(info["commit"] is None or re.fullmatch(r"[0-9a-f]{7,40}(-dirty|-unknown)?", info["commit"]),
                        info["commit"])
        self.assertIsNone(self._report()["pipeline"])
        self.assertNotIn("clientPipeline", self._report(pipeline=pipeline_info(commit="abc1234")))

    def test_http_reports_label_thresholds_and_commit_as_client_metadata(self):
        # En --source http el servidor solo devuelve modelVersion: los umbrales
        # y el commit son los del checkout que EVALÚA, no los del servidor. Se
        # escriben como `clientPipeline` con nota y `pipeline` queda en null.
        from infrastructure.nlp.alternatives import THRESHOLDS
        info = pipeline_info(commit="abc1234")
        self.assertEqual(report_pipeline_blocks("pipeline", info), (info, None))
        pipeline, client = report_pipeline_blocks("http", info)
        self.assertIsNone(pipeline)
        self.assertEqual(client["thresholds"], THRESHOLDS)
        self.assertEqual(client["commit"], "abc1234")
        self.assertIn("cliente", client["note"])
        report = self._report(source="HTTP http://servidor", pipeline=pipeline, client_pipeline=client)
        self.assertIsNone(report["pipeline"])
        self.assertEqual(report["clientPipeline"], client)
        self.assertEqual(panel_errors(report), [])
        json.dumps(report, ensure_ascii=False)

    def test_git_commit_marks_dirty_trees_and_failed_status_checks(self):
        # `git status` que falla no puede reportarse como árbol limpio: sufijo
        # `-unknown`; con cambios en archivos versionados, `-dirty`.
        head = "8bfca498417b0a23582b66577b65ae098d7cf724"

        def fake_run_factory(status_code, status_out):
            def fake_run(cmd, **kwargs):
                if cmd[:2] == ["git", "rev-parse"]:
                    return subprocess.CompletedProcess(cmd, 0, stdout=head + "\n", stderr="")
                if cmd[:2] == ["git", "status"]:
                    return subprocess.CompletedProcess(cmd, status_code, stdout=status_out, stderr="")
                raise AssertionError(cmd)
            return fake_run

        with patch("evaluate.subprocess.run", fake_run_factory(0, "")):
            self.assertEqual(git_commit(), head)
        with patch("evaluate.subprocess.run", fake_run_factory(0, " M evaluate.py\n")):
            self.assertEqual(git_commit(), head + "-dirty")
        with patch("evaluate.subprocess.run", fake_run_factory(128, "")):
            self.assertEqual(git_commit(), head + "-unknown")
        with patch("evaluate.subprocess.run", fake_run_factory(1, " M evaluate.py\n")):
            self.assertEqual(git_commit(), head + "-unknown")


class SetupModelGuardTests(unittest.TestCase):
    def test_default_save_dir_is_not_the_global_checkpoint(self):
        save_dir = setup_model.resolve_save_dir({})
        self.assertEqual(save_dir.resolve(), (setup_model.REPO_DIR / "models" / "t5_base").resolve())

    def test_out_env_selects_the_directory(self):
        self.assertEqual(setup_model.resolve_save_dir({"OUT": "./models/otro"}).resolve(),
                         (setup_model.REPO_DIR / "models" / "otro").resolve())

    def test_refuses_to_overwrite_global_checkpoint(self):
        for out in ("./models/t5_correction", "models/t5_correction/", str(setup_model.PROTECTED_DIR)):
            with self.assertRaises(SystemExit):
                setup_model.resolve_save_dir({"OUT": out})
            with self.assertRaises(SystemExit):
                setup_model.resolve_save_dir({"OUT": out, "ALLOW_MODEL_OVERWRITE": "false"})

    def test_explicit_allow_flag_permits_overwrite(self):
        save_dir = setup_model.resolve_save_dir({"OUT": "./models/t5_correction", "ALLOW_MODEL_OVERWRITE": "true"})
        self.assertEqual(save_dir.resolve(), setup_model.PROTECTED_DIR.resolve())

    def test_revision_comes_from_the_environment(self):
        self.assertIsNone(setup_model.resolve_revision({}))
        self.assertIsNone(setup_model.resolve_revision({"MODEL_REVISION": "  "}))
        self.assertEqual(setup_model.resolve_revision({"MODEL_REVISION": " abc123 "}), "abc123")


class NoHeavyImportsTests(unittest.TestCase):
    def test_importing_evaluate_and_setup_model_does_not_load_models(self):
        # En un subproceso limpio: no depende de lo que otros tests hayan importado.
        code = ("import sys; import evaluate, setup_model; "
                "bad = [m for m in ('torch', 'transformers', 'main', "
                "'infrastructure.nlp.correction_pipeline', 'infrastructure.ml.t5_model') "
                "if m in sys.modules]; "
                "assert not bad, bad")
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_DIR), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
