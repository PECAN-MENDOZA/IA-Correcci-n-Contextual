"""
test_evaluate_metrics.py

Tests puros del evaluador (evaluate.py): scorer de ediciones exactas
`exact_token_edits_v1` (Precisión / Recall / F0.5 a nivel de edición), lectura
del dataset con varias referencias, informe versionado (bloque
`technicalEvaluation` que consume el backend) y la guarda de setup_model.py
contra sobrescribir el checkpoint global.

Corren SIN torch/transformers/GPU ni modelos: `evaluate` y `setup_model` solo
importan la librería estándar a nivel de módulo (los modelos se cargan dentro
de `main()` / `build_predictor`).

Uso:
    .venv\\Scripts\\python.exe -m unittest test_evaluate_metrics -v
"""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from evaluate import (
    SCORER_VERSION,
    build_report,
    dataset_sha256,
    edit_scores,
    extract_edits,
    load_dataset,
    prf,
    resolve_model_version,
)
import setup_model


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

    def test_punctuation_is_its_own_token(self):
        self.assertEqual(extract_edits("hola", "Hola."), {(0, 1, ("Hola", "."))})

    def test_split_word_is_one_replacement_span(self):
        self.assertEqual(extract_edits("fui ala tienda", "fui a la tienda"), {(1, 2, ("a", "la"))})

    def test_deletion_has_empty_replacement(self):
        edits = extract_edits("el niño niño come", "el niño come")
        self.assertEqual(len(edits), 1)
        (start, end, replacement), = edits
        self.assertEqual(end - start, 1)
        self.assertEqual(replacement, ())

    def test_accent_change_is_a_replacement(self):
        self.assertEqual(extract_edits("el sabado voy", "el sábado voy"), {(1, 2, ("sábado",))})

    def test_adjacent_one_to_one_replacements_are_separate_edits(self):
        # difflib funde `arbol esta` → `árbol está` en un solo bloque; cada
        # palabra corregida debe contar por separado para no penalizar una
        # corrección parcial como FP + FN.
        self.assertEqual(extract_edits("el arbol esta lleno", "el árbol está lleno"),
                         {(1, 2, ("árbol",)), (2, 3, ("está",))})

    def test_unequal_length_replacement_stays_one_span(self):
        self.assertEqual(extract_edits("los uillos juegan", "los niños juegan"), {(1, 2, ("niños",))})
        self.assertEqual(extract_edits("delas frutas", "de las frutas"), {(0, 1, ("de", "las"))})


class AdjacentEditScoringTests(unittest.TestCase):
    def test_partial_fix_next_to_missed_word_is_tp_plus_fn(self):
        result = edit_scores(["el arbol esta lleno de hojas"], [["el árbol está lleno de hojas"]],
                             ["el árbol esta lleno de hojas"])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 1))


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


class ResolveModelVersionTests(unittest.TestCase):
    def test_explicit_version_wins(self):
        self.assertEqual(resolve_model_version("t5@abc", {"MODEL_VERSION": "env"}, Path("no/existe")), "t5@abc")

    def test_env_beats_manifest_and_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(json.dumps({"modelVersion": "base@1+lora@2"}), encoding="utf-8")
            self.assertEqual(resolve_model_version(None, {"MODEL_VERSION": "env-v"}, manifest), "env-v")
            self.assertEqual(resolve_model_version(None, {}, manifest), "base@1+lora@2")
        self.assertEqual(resolve_model_version(None, {}, Path(tmp) / "manifest.json"), "global-lora-unversioned")


def _case(cat, inp, golds, pred):
    return {"cat": cat, "input": inp, "golds": golds, "pred": pred}


class BuildReportTests(unittest.TestCase):
    SHA = "a" * 64

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

    def test_technical_evaluation_block_matches_backend_contract(self):
        te = self._report()["technicalEvaluation"]
        self.assertEqual(te["modelVersion"], "t5@abc")
        self.assertEqual(te["datasetSha256"], self.SHA)
        self.assertEqual(te["scorerVersion"], "exact_token_edits_v1")
        self.assertEqual((te["truePositives"], te["falsePositives"], te["falseNegatives"]), (1, 1, 1))
        self.assertAlmostEqual(te["precision"], 0.5)
        self.assertAlmostEqual(te["recall"], 0.5)
        self.assertAlmostEqual(te["fZeroFive"], 0.5)
        self.assertEqual(set(te), {
            "modelVersion", "datasetSha256", "scorerVersion", "precision", "recall", "fZeroFive",
            "truePositives", "falsePositives", "falseNegatives", "categories",
        })

    def test_categories_follow_dataset_order_with_own_counts(self):
        cats = self._report()["technicalEvaluation"]["categories"]
        self.assertEqual([c["category"] for c in cats], ["tilde", "correcto"])
        tilde, correcto = cats
        self.assertEqual((tilde["tp"], tilde["fp"], tilde["fn"]), (1, 0, 1))
        self.assertEqual((tilde["precision"], tilde["recall"]), (1.0, 0.5))
        self.assertAlmostEqual(tilde["f05"], 1.25 * 1.0 * 0.5 / (0.25 + 0.5))
        self.assertEqual((correcto["tp"], correcto["fp"], correcto["fn"]), (0, 1, 0))
        self.assertEqual((correcto["precision"], correcto["recall"], correcto["f05"]), (0.0, 0.0, 0.0))
        for c in cats:
            self.assertEqual(set(c), {"category", "tp", "fp", "fn", "precision", "recall", "f05"})

    def test_category_names_are_stripped_unique_and_capped_at_fifty(self):
        cases = [_case(f" c{i} ", "a", ["a"], "a") for i in range(55)]
        cases.append(_case("c1", "b", ["b"], "b"))
        cats = self._report(cases)["technicalEvaluation"]["categories"]
        names = [c["category"] for c in cats]
        self.assertEqual(len(names), 50)
        self.assertEqual(len(set(names)), 50)
        self.assertEqual(names[:2], ["c0", "c1"])

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

    def test_report_is_json_serializable(self):
        json.dumps(self._report(), ensure_ascii=False)


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


class NoHeavyImportsTests(unittest.TestCase):
    def test_importing_evaluate_and_setup_model_does_not_load_models(self):
        for mod in ("torch", "transformers", "main", "infrastructure.nlp.correction_pipeline"):
            self.assertNotIn(mod, sys.modules, f"{mod} no debe cargarse al importar evaluate/setup_model")


if __name__ == "__main__":
    unittest.main()
