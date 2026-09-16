"""
test_user_experiment_analysis.py

Tests de `scripts/analyze_user_experiment.py`: reproducción offline de PEO,
PPM y TAS a partir del `analysis.csv` del backend (19 columnas reales) con
las mismas fórmulas y reglas de muestra que `StudyMetricsService`.

Puros: solo librería estándar (csv/json/math/statistics), sin modelo, sin red
y sin torch. Incluyen la reconciliación contra los datos reales del estudio
PILOTO-02 (`tests/fixtures/analysis-piloto02-real.csv` frente a
`tests/fixtures/results-piloto02-real.json`).

Uso:
    .venv\\Scripts\\python.exe -m unittest -v test_user_experiment_analysis
"""
import csv
import hashlib
import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_user_experiment import (
    FORMULA_VERSION,
    AnalysisValidationError,
    analyze,
    compare_results,
    load_rows,
    main,
    paired_summary,
    render_markdown,
    row_metrics,
    semantic_rates,
    t_cdf,
    t_quantile,
    wilson,
    word_count,
)

REPO_DIR = Path(__file__).resolve().parent
FIXTURE = REPO_DIR / "tests" / "fixtures" / "experiment-analysis.csv"
REAL_CSV = REPO_DIR / "tests" / "fixtures" / "analysis-piloto02-real.csv"
REAL_RESULTS = REPO_DIR / "tests" / "fixtures" / "results-piloto02-real.json"

HEADER = (
    "pseudonym,condition,task,protocol_version,included,excluded,run_id,duration_ms,"
    "incident_reasons,word_count,orthography_errors,orthography_batch_id,final_text,"
    "suggestion_index,original_text,suggestion,semantic_score,accepted,semantic_batch_id"
)


def _write_csv(lines):
    """Escribe un CSV temporal con la cabecera real y devuelve su ruta."""
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    handle.write(HEADER + "\n" + "\n".join(lines) + "\n")
    handle.close()
    return Path(handle.name)


class FormulaTests(unittest.TestCase):
    """Fórmulas por ejecución y tasas agregadas (mismas que el backend)."""

    def test_word_count_follows_backend_tokenizer(self):
        self.assertEqual(word_count("uno dos tres cuatro"), 4)
        self.assertEqual(word_count("l'amour re-hacer, ¡hola!  año2026 ..."), 4)
        self.assertEqual(word_count(""), 0)
        self.assertEqual(word_count("   ...  --- "), 0)
        self.assertEqual(word_count("_sub_ palabra_"), 2)

    def test_word_count_normalises_combining_marks_like_java(self):
        # "cafés" en NFD (e + tilde combinante): Java (\w con UNICODE_CHARACTER_CLASS)
        # cuenta 1 palabra; sin NFC Python contaría 2. La aproximación se documenta.
        self.assertEqual(word_count("cafés bien"), 2)
        self.assertEqual(word_count("niñós"), 1)

    def test_paired_summary_without_pairs_is_all_null(self):
        summary = paired_summary([])
        self.assertEqual(summary["n"], 0)
        self.assertTrue(all(summary[key] is None for key in summary if key != "n"))

    def test_row_metrics(self):
        row = {"final_text": "uno dos tres cuatro", "duration_ms": "120000", "orthography_errors": "1"}
        self.assertEqual(row_metrics(row), {"words": 4, "peo": 25.0, "ppm": 2.0})

    def test_row_metrics_uses_word_count_column_when_present(self):
        row = {"final_text": "uno dos", "duration_ms": "60000", "orthography_errors": "1", "word_count": "4"}
        self.assertEqual(row_metrics(row), {"words": 4, "peo": 25.0, "ppm": 4.0})

    def test_row_metrics_without_words_or_scores(self):
        self.assertEqual(
            row_metrics({"final_text": "...", "duration_ms": "60000", "orthography_errors": "0"}),
            {"words": 0, "peo": None, "ppm": 0.0},
        )
        self.assertEqual(
            row_metrics({"final_text": "uno dos", "duration_ms": "60000", "orthography_errors": ""}),
            {"words": 2, "peo": None, "ppm": 2.0},
        )

    def test_row_metrics_rejects_nonpositive_duration(self):
        with self.assertRaises(AnalysisValidationError):
            row_metrics({"final_text": "uno", "duration_ms": "0", "orthography_errors": "0"})

    def test_semantic_rates(self):
        rows = [
            {"semantic_score": "0", "accepted": "true"},
            {"semantic_score": "2", "accepted": "false"},
        ]
        self.assertEqual(semantic_rates(rows), {"tas": 50.0, "tas_accepted": 100.0})

    def test_semantic_rates_without_denominator(self):
        self.assertEqual(semantic_rates([]), {"tas": None, "tas_accepted": None})
        rows = [{"semantic_score": "1", "accepted": "false"}, {"semantic_score": "", "accepted": ""}]
        self.assertEqual(semantic_rates(rows), {"tas": 0.0, "tas_accepted": None})


class StatisticsTests(unittest.TestCase):
    """Distribución t (fallback puro), IC t, prueba t emparejada e intervalo de Wilson."""

    def test_t_quantile_matches_reference_values(self):
        for df, expected in [(1, 12.7062047362), (2, 4.30265272991), (10, 2.22813885196), (30, 2.04227245630)]:
            self.assertAlmostEqual(t_quantile(df, 0.975), expected, places=8)
        self.assertAlmostEqual(t_quantile(5, 0.5), 0.0, places=10)
        self.assertAlmostEqual(t_quantile(5, 0.025), -t_quantile(5, 0.975), places=10)

    def test_t_cdf_matches_reference_values(self):
        self.assertAlmostEqual(t_cdf(0.0, 7), 0.5, places=12)
        self.assertAlmostEqual(t_cdf(2.22813885196, 10), 0.975, places=9)
        self.assertAlmostEqual(t_cdf(-12.7062047362, 1), 0.025, places=9)

    def test_paired_summary_matches_textbook_formulas(self):
        pairs = [(10.0, 30.0), (25.0 / 6.0, 25.0), (10.0, 20.0)]
        deltas = [a - u for a, u in pairs]
        summary = paired_summary(pairs)
        mean = statistics.fmean(deltas)
        sd = statistics.stdev(deltas)
        half = t_quantile(2, 0.975) * sd / math.sqrt(3)
        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["meanDelta"], mean, places=12)
        self.assertAlmostEqual(summary["sdDelta"], sd, places=12)
        self.assertAlmostEqual(summary["ci95Lower"], mean - half, places=12)
        self.assertAlmostEqual(summary["ci95Upper"], mean + half, places=12)
        self.assertAlmostEqual(summary["tStatistic"], mean / (sd / math.sqrt(3)), places=12)
        self.assertAlmostEqual(summary["cohenDz"], mean / sd, places=12)
        self.assertAlmostEqual(summary["pValue"], 2 * t_cdf(-abs(summary["tStatistic"]), 2), places=12)
        self.assertAlmostEqual(summary["assistedMean"], statistics.fmean(a for a, _ in pairs), places=12)
        self.assertAlmostEqual(summary["unassistedSd"], statistics.stdev(u for _, u in pairs), places=12)

    def test_paired_summary_with_one_pair_reports_nulls(self):
        summary = paired_summary([(12.5, 16.666666666666668)])
        self.assertEqual(summary["n"], 1)
        self.assertAlmostEqual(summary["meanDelta"], -4.166666666666668, places=12)
        for key in ("assistedSd", "unassistedSd", "sdDelta", "ci95Lower", "ci95Upper", "tStatistic", "pValue", "cohenDz"):
            self.assertIsNone(summary[key], key)

    def test_paired_summary_with_zero_variance_reports_no_inference(self):
        summary = paired_summary([(5.0, 3.0), (7.0, 5.0)])
        self.assertEqual(summary["sdDelta"], 0.0)
        self.assertIsNone(summary["ci95Lower"])
        self.assertIsNone(summary["tStatistic"])
        self.assertIsNone(summary["cohenDz"])

    def test_wilson_interval(self):
        lower, upper = wilson(0, 1)
        self.assertEqual(lower, 0.0)
        self.assertAlmostEqual(upper, 79.34506882081973, places=12)
        self.assertEqual(wilson(0, 0), (None, None))
        lower, upper = wilson(3, 6)
        self.assertAlmostEqual(lower, 18.76, delta=0.01)
        self.assertAlmostEqual(upper, 81.24, delta=0.01)


class FixtureAnalysisTests(unittest.TestCase):
    """Fixture sintético con 5 participantes (3 incluidos, 1 par incompleto, 1 sin ejecución elegible)."""

    @classmethod
    def setUpClass(cls):
        cls.rows, cls.sha = load_rows(FIXTURE)
        cls.report = analyze(cls.rows, input_sha256=cls.sha, input_name=FIXTURE.name, ppm_margin=2.0)

    def test_versioning_and_hash(self):
        self.assertEqual(FORMULA_VERSION, "user_study_v1")
        self.assertEqual(self.report["formulaVersion"], "user_study_v1")
        self.assertEqual(self.report["inputSha256"], hashlib.sha256(FIXTURE.read_bytes()).hexdigest())
        self.assertEqual(len(self.report["inputSha256"]), 64)

    def test_sample_partition(self):
        self.assertEqual(
            self.report["sample"],
            {
                "participantsTotal": 5,
                "participantsIncluded": 3,
                "participantsWithIncompletePair": 1,
                "participantsWithoutEligibleRun": 1,
                "runsCompleted": 10,
                "runsIncluded": 8,
                "runsExcluded": 1,
                "runsInIncompletePairs": 1,
                "runsWithoutCountableWords": 1,
            },
        )
        self.assertEqual(self.report["excluded"]["participantsWithIncompletePair"], ["P-004"])
        self.assertEqual(self.report["excluded"]["participantsWithoutEligibleRun"], ["P-005"])

    def test_peo_delta_is_negative(self):
        peo = self.report["peo"]
        self.assertEqual(peo["paired"]["n"], 3)
        self.assertEqual(peo["participantsAnalyzed"], 3)
        self.assertEqual(peo["runsAnalyzed"], 7)
        self.assertAlmostEqual(peo["paired"]["assistedMean"], (10.0 + 25.0 / 6.0 + 10.0) / 3, places=12)
        self.assertAlmostEqual(peo["paired"]["unassistedMean"], 25.0, places=12)
        self.assertAlmostEqual(peo["paired"]["meanDelta"], (-20.0 - (25.0 - 25.0 / 6.0) - 10.0) / 3, places=12)
        self.assertLess(peo["paired"]["meanDelta"], 0)
        self.assertLess(peo["paired"]["ci95Lower"], peo["paired"]["meanDelta"])
        self.assertGreater(peo["paired"]["ci95Upper"], peo["paired"]["meanDelta"])
        self.assertLess(peo["paired"]["tStatistic"], 0)
        self.assertLess(peo["paired"]["cohenDz"], 0)
        self.assertEqual(peo["relativeReductionN"], 3)
        self.assertEqual(peo["relativeReductionSkipped"], 0)
        self.assertAlmostEqual(peo["relativeReductionMean"], (200.0 / 3 + 250.0 / 3 + 50.0) / 3, places=12)
        self.assertEqual(peo["upperCiBelowZero"], peo["paired"]["ci95Upper"] < 0)

    def test_ppm_direction_and_non_inferiority(self):
        ppm = self.report["ppm"]
        self.assertEqual(ppm["paired"]["n"], 3)
        self.assertEqual(ppm["runsAnalyzed"], 8)
        self.assertAlmostEqual(ppm["paired"]["assistedMean"], (20.0 / 3 + 12.0 + 10.0) / 3, places=12)
        self.assertAlmostEqual(ppm["paired"]["unassistedMean"], (5.0 + 8.0 + 2.5) / 3, places=12)
        self.assertGreater(ppm["paired"]["meanDelta"], 0)
        self.assertFalse(ppm["descriptive"])
        self.assertEqual(ppm["nonInferiorityMargin"], 2.0)
        self.assertEqual(ppm["nonInferior"], ppm["paired"]["ci95Lower"] > -2.0)

    def test_ppm_is_descriptive_without_margin(self):
        report = analyze(self.rows, input_sha256=self.sha, input_name=FIXTURE.name)
        self.assertTrue(report["ppm"]["descriptive"])
        self.assertIsNone(report["ppm"]["nonInferiorityMargin"])
        self.assertIsNone(report["ppm"]["nonInferior"])
        self.assertTrue(report["tas"]["descriptive"])
        self.assertIsNone(report["tas"]["upperCiBelowLimit"])

    def test_tas_pooled_and_per_participant(self):
        tas = self.report["tas"]
        self.assertEqual(tas["suggestionsEvaluated"], 6)
        self.assertEqual(tas["harmfulSuggestions"], 3)
        self.assertEqual(tas["runsAnalyzed"], 3)
        self.assertEqual(tas["participantsEvaluated"], 3)
        self.assertEqual(tas["participantsWithoutDenominator"], 0)
        self.assertAlmostEqual(tas["pooledRate"], 50.0, places=12)
        self.assertAlmostEqual(tas["participantMean"], (50.0 + 100.0 + 100.0 / 3) / 3, places=12)
        self.assertGreaterEqual(tas["participantCi95Lower"], 0.0)
        self.assertLessEqual(tas["participantCi95Upper"], 100.0)
        lower, upper = wilson(3, 6)
        self.assertAlmostEqual(tas["pooledCi95Lower"], lower, places=12)
        self.assertAlmostEqual(tas["pooledCi95Upper"], upper, places=12)

    def test_tas_accepted(self):
        accepted = self.report["tasAccepted"]
        self.assertEqual(accepted["suggestionsEvaluated"], 3)
        self.assertEqual(accepted["harmfulSuggestions"], 1)
        self.assertEqual(accepted["runsAnalyzed"], 3)
        self.assertAlmostEqual(accepted["pooledRate"], 100.0 / 3, places=12)
        self.assertAlmostEqual(accepted["participantMean"], 100.0 / 3, places=12)

    def test_tas_limit_criterion(self):
        report = analyze(self.rows, input_sha256=self.sha, input_name=FIXTURE.name, tas_limit=90.0)
        self.assertFalse(report["tas"]["descriptive"])
        self.assertEqual(report["tas"]["limit"], 90.0)
        self.assertEqual(report["tas"]["upperCiBelowLimit"], report["tas"]["pooledCi95Upper"] < 90.0)

    def test_participants_rows(self):
        rows = {row["pseudonym"]: row for row in self.report["participants"]}
        self.assertEqual(sorted(rows), ["P-001", "P-002", "P-003"])
        self.assertAlmostEqual(rows["P-002"]["peoAssisted"], 25.0 / 6.0, places=12)
        self.assertAlmostEqual(rows["P-002"]["ppmAssisted"], 12.0, places=12)
        self.assertAlmostEqual(rows["P-002"]["tas"], 100.0, places=12)
        self.assertAlmostEqual(rows["P-002"]["tasAccepted"], 100.0, places=12)
        self.assertAlmostEqual(rows["P-003"]["ppmUnassisted"], 2.5, places=12)
        self.assertAlmostEqual(rows["P-003"]["tasAccepted"], 0.0, places=12)
        self.assertAlmostEqual(rows["P-001"]["peoRelativeReduction"], 200.0 / 3, places=12)

    def test_annotation_status(self):
        self.assertEqual(self.report["orthographyAnnotation"]["status"], "ADJUDICATED")
        self.assertEqual(self.report["semanticAnnotation"]["status"], "ADJUDICATED")
        self.assertEqual(self.report["provenance"]["protocolVersions"], [1])

    def test_markdown_report_mentions_everything(self):
        text = render_markdown(self.report)
        for needle in ("PEO", "PPM", "TAS", "user_study_v1", self.report["inputSha256"], "n = 3", "P-004", "P-005", "IC 95"):
            self.assertIn(needle, text)

    def test_json_is_serialisable_and_stable(self):
        text = json.dumps(self.report, ensure_ascii=False, sort_keys=True)
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)


class ValidationTests(unittest.TestCase):
    """Filas inválidas se rechazan con un mensaje que identifica la fila."""

    BASE = (
        "P-001,{cond},TASK_A,1,{inc},{exc},{run},{dur},,{wc},{err},b1,{text},{idx},{orig},{sug},{score},{acc},{sb}"
    )

    def _row(self, **kw):
        values = dict(cond="ASSISTED", inc="true", exc="false", run="r1", dur="60000", wc="", err="0",
                      text="uno dos", idx="", orig="", sug="", score="", acc="", sb="")
        values.update(kw)
        return self.BASE.format(**values)

    def _assert_rejected(self, lines, fragment):
        path = _write_csv(lines)
        try:
            with self.assertRaises(AnalysisValidationError) as ctx:
                load_rows(path)
            self.assertIn(fragment, str(ctx.exception))
        finally:
            path.unlink()

    def test_unknown_condition(self):
        self._assert_rejected([self._row(cond="MIXED")], "condition")

    def test_nonpositive_duration(self):
        self._assert_rejected([self._row(dur="0")], "duration_ms")

    def test_negative_or_non_integer_errors(self):
        self._assert_rejected([self._row(err="-1")], "orthography_errors")
        self._assert_rejected([self._row(err="1.5")], "orthography_errors")

    def test_errors_above_word_count(self):
        self._assert_rejected([self._row(err="3")], "orthography_errors")

    def test_duplicate_suggestion_row(self):
        row = self._row(idx="0", orig="uno dos", sug="uno dos.", score="2", acc="true", sb="s1")
        self._assert_rejected([row, row], "duplicate")

    def test_run_rows_must_agree(self):
        first = self._row(idx="0", orig="a", sug="b", score="2", acc="true", sb="s1")
        second = self._row(idx="1", orig="a", sug="c", score="2", acc="true", sb="s1", dur="61000")
        self._assert_rejected([first, second], "run_id")

    def test_run_cannot_be_shared_by_participants(self):
        other = self._row().replace("P-001", "P-002", 1)
        self._assert_rejected([self._row(), other], "run_id")

    def test_missing_column(self):
        path = _write_csv([])
        try:
            path.write_text("pseudonym,condition\nP-001,ASSISTED\n", encoding="utf-8")
            with self.assertRaises(AnalysisValidationError) as ctx:
                load_rows(path)
            self.assertIn("column", str(ctx.exception))
        finally:
            path.unlink()

    def test_included_flag_must_match_pair_rule(self):
        self._assert_rejected([self._row(inc="true")], "included")

    def test_missing_adjudication_yields_null_metric_not_error(self):
        lines = [
            self._row(run="r1", cond="ASSISTED", inc="true", err="", idx="0", orig="uno dos", sug="uno dos.", score="", acc="true", sb=""),
            self._row(run="r2", cond="UNASSISTED", inc="true", err=""),
        ]
        path = _write_csv(lines)
        try:
            rows, sha = load_rows(path)
            report = analyze(rows, input_sha256=sha, input_name="x.csv")
            self.assertIsNone(report["peo"])
            self.assertIsNone(report["tas"])
            self.assertIsNotNone(report["ppm"])
            self.assertEqual(report["orthographyAnnotation"]["status"], "NO_BATCH")
            self.assertEqual(report["semanticAnnotation"]["status"], "NO_BATCH")
            self.assertEqual(report["sample"]["participantsIncluded"], 1)
        finally:
            path.unlink()

    def _analyze(self, lines):
        path = _write_csv(lines)
        try:
            rows, sha = load_rows(path)
            return analyze(rows, input_sha256=sha, input_name="x.csv")
        finally:
            path.unlink()

    def test_incomplete_coverage_status(self):
        # Dos ejecuciones incluidas, solo una adjudicada: INCOMPLETE_COVERAGE (nunca un PEO parcial).
        report = self._analyze([
            self._row(run="r1", cond="ASSISTED", inc="true", err="1", idx="0", orig="a", sug="b", score="2", acc="true", sb="s1"),
            self._row(run="r2", cond="UNASSISTED", inc="true", err=""),
            self._row(run="r3", cond="ASSISTED", inc="true", err="0", idx="0", orig="a", sug="c", score="", acc="false", sb=""),
        ])
        self.assertEqual(report["orthographyAnnotation"]["status"], "INCOMPLETE_COVERAGE")
        self.assertEqual(report["semanticAnnotation"]["status"], "INCOMPLETE_COVERAGE")
        self.assertIsNone(report["peo"])
        self.assertIsNone(report["tas"])

    def test_not_applicable_status_without_countable_words_or_suggestions(self):
        report = self._analyze([
            self._row(run="r1", cond="ASSISTED", inc="true", text="...", wc="0", err="0"),
            self._row(run="r2", cond="UNASSISTED", inc="true", text="---", wc="0", err="0"),
        ])
        self.assertEqual(report["orthographyAnnotation"]["status"], "NOT_APPLICABLE")
        self.assertEqual(report["semanticAnnotation"]["status"], "NOT_APPLICABLE")
        self.assertEqual(report["sample"]["runsWithoutCountableWords"], 2)
        self.assertEqual(report["ppm"]["paired"]["assistedMean"], 0.0)

    def test_no_sample_status_without_complete_pairs(self):
        report = self._analyze([self._row(run="r1", cond="ASSISTED", inc="false", err="0")])
        self.assertEqual(report["orthographyAnnotation"]["status"], "NO_SAMPLE")
        self.assertEqual(report["semanticAnnotation"]["status"], "NO_SAMPLE")
        self.assertIsNone(report["ppm"])
        self.assertEqual(report["sample"]["participantsWithIncompletePair"], 1)

    def test_word_count_mismatch_is_a_warning_and_the_csv_value_wins(self):
        report = self._analyze([
            self._row(run="r1", cond="ASSISTED", inc="true", wc="5", err="1", text="uno dos"),
            self._row(run="r2", cond="UNASSISTED", inc="true", wc="2", err="0", text="uno dos"),
        ])
        self.assertEqual(len(report["warnings"]), 1)
        self.assertIn("word_count=5", report["warnings"][0])
        self.assertIn("counts 2", report["warnings"][0])
        self.assertAlmostEqual(report["participants"][0]["peoAssisted"], 20.0, places=12)

    def test_rows_with_more_fields_than_the_header_are_rejected(self):
        self._assert_rejected([self._row() + ",extra"], "too many fields")


class ReconciliationTests(unittest.TestCase):
    """El análisis offline reproduce `GET …/results` del backend para el mismo CSV real (PILOTO-02)."""

    @classmethod
    def setUpClass(cls):
        cls.rows, cls.sha = load_rows(REAL_CSV)
        cls.report = analyze(cls.rows, input_sha256=cls.sha, input_name=REAL_CSV.name)
        cls.backend = json.loads(REAL_RESULTS.read_text(encoding="utf-8"))

    def test_real_numbers_match_backend(self):
        peo = self.report["peo"]["paired"]
        self.assertEqual(peo["n"], 1)
        self.assertAlmostEqual(peo["assistedMean"], 12.5, places=12)
        self.assertAlmostEqual(peo["unassistedMean"], 16.666666666666668, places=12)
        self.assertAlmostEqual(peo["meanDelta"], -4.166666666666668, places=12)
        ppm = self.report["ppm"]["paired"]
        self.assertAlmostEqual(ppm["assistedMean"], 17.116570980280283, places=12)
        self.assertAlmostEqual(ppm["unassistedMean"], 5.869979944235191, places=12)
        self.assertAlmostEqual(ppm["meanDelta"], 11.246591036045093, places=12)
        self.assertAlmostEqual(self.report["tas"]["pooledCi95Upper"], 79.34506882081973, places=12)
        self.assertEqual(self.report["tas"]["pooledRate"], 0.0)
        self.assertIsNone(self.report["tasAccepted"]["pooledRate"])
        self.assertEqual(self.report["tasAccepted"]["participantsWithoutDenominator"], 1)

    def test_inferential_fields_are_null_with_one_pair(self):
        for block in ("peo", "ppm"):
            paired = self.report[block]["paired"]
            for key in ("assistedSd", "unassistedSd", "sdDelta", "ci95Lower", "ci95Upper", "tStatistic", "pValue", "cohenDz"):
                self.assertIsNone(paired[key], f"{block}.{key}")
        self.assertIsNone(self.report["peo"]["upperCiBelowZero"])
        self.assertIsNone(self.report["tas"]["participantSd"])
        self.assertIsNone(self.report["tas"]["participantCi95Lower"])

    def test_compare_results_passes(self):
        discrepancies, notes = compare_results(self.report, self.backend)
        self.assertEqual(discrepancies, [])
        # El CSV solo exporta participantes con ejecuciones completadas: el backend cuenta uno más sin ninguna.
        self.assertTrue(any("participantsTotal" in note for note in notes))

    def test_compare_results_detects_a_discrepancy(self):
        tampered = json.loads(json.dumps(self.backend))
        tampered["peo"]["paired"]["meanDelta"] += 1e-3
        tampered["tas"]["pooledCi95Upper"] = None
        tampered["participants"][0]["ppmAssisted"] = 0.0
        discrepancies, _ = compare_results(self.report, tampered)
        joined = "\n".join(discrepancies)
        self.assertIn("peo.paired.meanDelta", joined)
        self.assertIn("tas.pooledCi95Upper", joined)
        self.assertIn("participants[P-002].ppmAssisted", joined)
        self.assertEqual(len(discrepancies), 3)

    def test_participants_total_gap_must_match_without_eligible_gap(self):
        tampered = json.loads(json.dumps(self.backend))
        tampered["sample"]["participantsTotal"] += 1            # gap 2 vs withoutEligible gap 1
        self.assertTrue(any("participantsTotal" in d for d in compare_results(self.report, tampered)[0]))
        tampered = json.loads(json.dumps(self.backend))
        tampered["sample"]["participantsTotal"] = 0             # gap negativo
        self.assertTrue(any("participantsTotal" in d for d in compare_results(self.report, tampered)[0]))
        tampered = json.loads(json.dumps(self.backend))
        del tampered["sample"]["participantsTotal"]             # clave ausente
        self.assertTrue(any("participantsTotal" in d for d in compare_results(self.report, tampered)[0]))

    def test_not_adjudicated_backend_status_is_accepted_with_a_note(self):
        # El backend puede estar en NOT_ADJUDICATED (lote nuevo sin adjudicación
        # vigente); el CSV solo puede ver NO_BATCH / INCOMPLETE_COVERAGE. Si los
        # números coinciden, no es discrepancia: se anota.
        with REAL_CSV.open(encoding="utf-8-sig", newline="") as handle:
            table = list(csv.reader(handle))
        err_i = table[0].index("orthography_errors")
        for cells in table[1:]:
            cells[err_i] = ""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blank.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerows(table)
            rows, sha = load_rows(path)
            report = analyze(rows, input_sha256=sha, input_name="blank.csv")
        self.assertEqual(report["orthographyAnnotation"]["status"], "NO_BATCH")
        backend = json.loads(json.dumps(self.backend))
        backend["orthographyAnnotation"]["status"] = "NOT_ADJUDICATED"
        backend["peo"] = None
        for row in backend["participants"]:
            for key in ("peoAssisted", "peoUnassisted", "peoDelta", "peoRelativeReduction"):
                row[key] = None
        discrepancies, notes = compare_results(report, backend)
        self.assertEqual(discrepancies, [])
        self.assertTrue(any("NOT_ADJUDICATED" in note and "NO_BATCH" in note for note in notes))
        # ADJUDICATED en el backend con NO_BATCH offline sí es una discrepancia.
        backend["orthographyAnnotation"]["status"] = "ADJUDICATED"
        discrepancies, _ = compare_results(report, backend)
        self.assertTrue(any("orthographyAnnotation.status" in d for d in discrepancies))
        # NO_SAMPLE / NOT_APPLICABLE se exigen literales.
        backend = json.loads(json.dumps(self.backend))
        backend["semanticAnnotation"]["status"] = "NOT_APPLICABLE"
        discrepancies, _ = compare_results(self.report, backend)
        self.assertTrue(any("semanticAnnotation.status" in d for d in discrepancies))


class CliTests(unittest.TestCase):
    """Punto de entrada: escribe JSON y Markdown y devuelve un código de salida claro."""

    def test_main_writes_reports_and_compares(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "out.json"
            md_path = Path(tmp) / "out.md"
            code = main([
                str(REAL_CSV), "--json", str(json_path), "--markdown", str(md_path),
                "--compare-results", str(REAL_RESULTS),
            ])
            self.assertEqual(code, 0)
            report = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(report["formulaVersion"], "user_study_v1")
            self.assertEqual(report["inputSha256"], hashlib.sha256(REAL_CSV.read_bytes()).hexdigest())
            markdown = md_path.read_text(encoding="utf-8")
            self.assertIn("## Reconciliación", markdown)
            self.assertIn("sin discrepancias", markdown)

    def test_main_fails_when_backend_differs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tampered = json.loads(REAL_RESULTS.read_text(encoding="utf-8"))
            tampered["ppm"]["paired"]["assistedMean"] = 1.0
            results = Path(tmp) / "results.json"
            results.write_text(json.dumps(tampered), encoding="utf-8")
            self.assertEqual(main([str(REAL_CSV), "--compare-results", str(results)]), 1)

    def test_main_rejects_invalid_thresholds(self):
        self.assertEqual(main([str(REAL_CSV), "--ppm-margin", "0"]), 2)
        self.assertEqual(main([str(REAL_CSV), "--tas-limit", "101"]), 2)

    def test_main_validates_tolerance(self):
        for bad in ("-1e-6", "nan", "inf"):
            self.assertEqual(main([str(REAL_CSV), "--compare-results", str(REAL_RESULTS), f"--tolerance={bad}"]), 2)
        self.assertEqual(main([str(REAL_CSV), "--compare-results", str(REAL_RESULTS), "--tolerance", "0"]), 0)

    def test_main_reports_unreadable_results_file_as_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main([str(REAL_CSV), "--compare-results", str(Path(tmp) / "no-existe.json")]), 2)
            broken = Path(tmp) / "broken.json"
            broken.write_text("{ roto", encoding="utf-8")
            self.assertEqual(main([str(REAL_CSV), "--compare-results", str(broken)]), 2)
            not_object = Path(tmp) / "list.json"
            not_object.write_text("[1, 2]", encoding="utf-8")
            self.assertEqual(main([str(REAL_CSV), "--compare-results", str(not_object)]), 2)

    def test_markdown_shows_the_tolerance_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            tampered = json.loads(REAL_RESULTS.read_text(encoding="utf-8"))
            tampered["ppm"]["paired"]["assistedMean"] += 1e-3
            results = Path(tmp) / "results.json"
            results.write_text(json.dumps(tampered), encoding="utf-8")
            md_path = Path(tmp) / "out.md"
            self.assertEqual(main([str(REAL_CSV), "--compare-results", str(results), "--tolerance", "1e-4",
                                   "--markdown", str(md_path)]), 1)
            markdown = md_path.read_text(encoding="utf-8")
            self.assertIn("tolerancia 0.0001", markdown)
            self.assertNotIn("1e-6", markdown)
            self.assertEqual(main([str(REAL_CSV), "--compare-results", str(results), "--tolerance", "1e-2"]), 0)


if __name__ == "__main__":
    unittest.main()
