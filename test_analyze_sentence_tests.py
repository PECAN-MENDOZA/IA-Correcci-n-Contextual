"""
test_analyze_sentence_tests.py

Tests de `scripts/analyze_sentence_tests.py`: reproducción offline de los
resultados de una prueba de oraciones (`GET /api/v1/research/tests/{id}/results`)
a partir del CSV de exportación de 21 columnas, con el mismo bootstrap
determinista (SplitMix64, semilla 42, 2000 remuestreos, cuantil R-7) que
`TestResultsService` del backend.

Puros: solo librería estándar, sin modelo, sin red y sin torch. La fixture
`test_fixtures/sentence_tests/responses-sample.csv` replica la cohorte
sintética de `TestResultsServiceTests` (Java) con un cuarto alumno excluido, de
modo que las expectativas calculadas a mano son las mismas en ambos lados.

Uso:
    .venv\\Scripts\\python.exe -m unittest -v test_analyze_sentence_tests
"""
import contextlib
import csv
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_sentence_tests import (
    COLUMNS,
    AnalysisValidationError,
    SplitMix64,
    analyze,
    bootstrap_mean_interval,
    compare,
    interval,
    load_rows,
    main,
    paired_summary,
    per_student_condition,
    quantile,
    wilson,
)

REPO_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = REPO_DIR / "test_fixtures" / "sentence_tests"
SAMPLE_CSV = FIXTURE_DIR / "responses-sample.csv"
SAMPLE_JSON = FIXTURE_DIR / "results-sample.json"

# Valores por alumno de la cohorte sintética (ver SyntheticCohort.java).
ERR_ASSISTED = [200.0 / 7, 0.0, 200.0 / 3]
ERR_UNASSISTED = [25.0, 25.0, 100.0 / 6]
PPM_ASSISTED = [7 / (9000 / 60000.0), 4 / (3000 / 60000.0), 7 / (11000 / 60000.0)]
PPM_UNASSISTED = [40.0, 120.0, 30.0]


def _write_csv(rows, header=COLUMNS):
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    with handle:
        writer = csv.writer(handle, lineterminator="\r\n")
        writer.writerow(header)
        writer.writerows(rows)
    return Path(handle.name)


def _sample_rows():
    with SAMPLE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        return header, [row for row in reader]


# --------------------------------------------------------------------------- valores dorados (Task 6)


class SplitMix64Tests(unittest.TestCase):
    def test_first_three_values_with_seed_42(self):
        rng = SplitMix64(42)
        self.assertEqual([rng.next_long() for _ in range(3)],
                         [-4767286540954276203, 2949826092126892291, 5139283748462763858])

    def test_next_long_stays_within_signed_64_bits(self):
        rng = SplitMix64(-1)
        for _ in range(1000):
            value = rng.next_long()
            self.assertTrue(-(1 << 63) <= value < (1 << 63))

    def test_next_index_uses_unsigned_remainder(self):
        # Primer valor negativo con semilla 42: el resto se toma sobre el entero de 64 bits sin signo
        # (13679457532755275413 mod 5 = 3), no sobre el negativo (Python daría 2 y Java -3).
        self.assertEqual(SplitMix64(42).next_index(5), 3)
        self.assertEqual(SplitMix64(42).next_index(1), 0)


class BootstrapTests(unittest.TestCase):
    def test_golden_interval_for_one_to_five(self):
        result = bootstrap_mean_interval([1, 2, 3, 4, 5])
        self.assertEqual(result["n"], 5)
        self.assertAlmostEqual(result["mean"], 3.0, delta=1e-9)
        self.assertAlmostEqual(result["lower"], 1.8, delta=1e-9)
        self.assertAlmostEqual(result["upper"], 4.2, delta=1e-9)

    def test_is_deterministic(self):
        values = [3.5, 1.25, 9.0, 4.75]
        self.assertEqual(bootstrap_mean_interval(values), bootstrap_mean_interval(values))

    def test_single_value_has_mean_but_no_bounds(self):
        self.assertEqual(bootstrap_mean_interval([7.5]), {"n": 1, "mean": 7.5, "lower": None, "upper": None})

    def test_empty_matches_metric_interval_empty(self):
        self.assertEqual(bootstrap_mean_interval([]), {"n": 0, "mean": None, "lower": None, "upper": None})

    def test_quantile_r7(self):
        sorted_values = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(quantile(sorted_values, 0.0), 1.0)
        self.assertAlmostEqual(quantile(sorted_values, 0.5), 2.5)
        self.assertAlmostEqual(quantile(sorted_values, 1.0), 4.0)
        self.assertAlmostEqual(quantile(sorted_values, 0.025), 1.075)


# --------------------------------------------------------------------------- estadística clásica


class StatsTests(unittest.TestCase):
    def test_wilson_matches_backend_constant(self):
        lower, upper = wilson(6, 9)
        self.assertLess(lower, 600.0 / 9)
        self.assertGreater(upper, 600.0 / 9)
        # Fórmula cerrada con Z_95 = 1.959964.
        z = 1.959964
        p = 6 / 9
        z2n = z * z / 9
        center = (p + z2n / 2) / (1 + z2n)
        half = z / (1 + z2n) * math.sqrt(p * (1 - p) / 9 + z * z / (4 * 81))
        self.assertAlmostEqual(lower, 100 * (center - half), delta=1e-9)
        self.assertAlmostEqual(upper, 100 * (center + half), delta=1e-9)
        self.assertEqual(wilson(0, 0), (None, None))
        self.assertAlmostEqual(wilson(0, 3)[0], 0.0, delta=1e-12)
        self.assertAlmostEqual(wilson(3, 3)[1], 100.0, delta=1e-12)

    def test_interval_uses_student_t(self):
        result = interval([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["n"], 4)
        self.assertAlmostEqual(result["mean"], 2.5)
        # t(0.975, 3) = 3.182446305; sd = 1.290994449; semiancho = t * sd / 2.
        half = 3.182446305 * 1.2909944487358056 / 2
        self.assertAlmostEqual(result["lower"], 2.5 - half, delta=1e-6)
        self.assertAlmostEqual(result["upper"], 2.5 + half, delta=1e-6)
        self.assertEqual(interval([2.0, 2.0])["lower"], None)
        self.assertEqual(interval([])["mean"], None)

    def test_paired_summary_keys_follow_paired_delta(self):
        summary = paired_summary([(3.0, 1.0), (4.0, 1.5), (5.0, 2.0)])
        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["meanAssisted"], 4.0)
        self.assertAlmostEqual(summary["meanUnassisted"], 1.5)
        self.assertAlmostEqual(summary["meanDelta"], 2.5)
        self.assertAlmostEqual(summary["t"], 2.5 / (0.5 / math.sqrt(3)), delta=1e-9)
        self.assertAlmostEqual(summary["dz"], 5.0, delta=1e-9)
        self.assertTrue(0.0 < summary["p"] < 1.0)
        self.assertLess(summary["tLower"], 2.5)
        self.assertGreater(summary["tUpper"], 2.5)
        empty = paired_summary([])
        self.assertEqual(empty["n"], 0)
        self.assertIsNone(empty["meanDelta"])


# --------------------------------------------------------------------------- carga y validación


class LoadRowsTests(unittest.TestCase):
    def test_loads_sample_with_sha256_of_bytes(self):
        rows, digest = load_rows(SAMPLE_CSV)
        self.assertEqual(len(rows), 16)
        self.assertEqual(digest, hashlib.sha256(SAMPLE_CSV.read_bytes()).hexdigest())
        quoted = rows[10]
        self.assertEqual(quoted["final_text"], 'Un texto, con "comillas"')
        self.assertIsNone(quoted["error_count"])
        self.assertEqual(quoted["error_source"], "PENDING")
        skipped = rows[7]
        self.assertTrue(skipped["skipped"])
        self.assertIsNone(skipped["duration_first_key_ms"])
        self.assertEqual(skipped["duration_start_ms"], 500)
        self.assertTrue(rows[12]["excluded"])
        self.assertIsNone(rows[12]["model_version"])

    def test_rejects_different_header(self):
        header, rows = _sample_rows()
        for bad in (header[:-1], header[1:] + header[:1], header + ["extra"], [c.upper() for c in header]):
            path = _write_csv(rows, header=bad)
            with self.assertRaises(AnalysisValidationError):
                load_rows(path)

    def test_rejects_unknown_error_source(self):
        header, rows = _sample_rows()
        rows[0][COLUMNS.index("error_source")] = "MANUAL"
        with self.assertRaisesRegex(AnalysisValidationError, "error_source"):
            load_rows(_write_csv(rows))

    def test_rejects_bad_types(self):
        header, rows = _sample_rows()
        cases = (
            ("position", "x"), ("kind", "OTHER"), ("assistance", "MAYBE"), ("skipped", "yes"),
            ("word_count", "-1"), ("error_count", "1.5"), ("duration_first_key_ms", "abc"),
            ("suggestions_offered", ""), ("excluded", ""),
        )
        for column, value in cases:
            broken = [list(row) for row in rows]
            broken[0][COLUMNS.index(column)] = value
            with self.subTest(column=column), self.assertRaisesRegex(AnalysisValidationError, column):
                load_rows(_write_csv(broken))

    def test_rejects_row_with_too_many_fields(self):
        header, rows = _sample_rows()
        rows[0].append("extra")
        with self.assertRaisesRegex(AnalysisValidationError, "line 2"):
            load_rows(_write_csv(rows))


# --------------------------------------------------------------------------- acumuladores por alumno


class PerStudentConditionTests(unittest.TestCase):
    def test_accumulators_match_synthetic_cohort(self):
        rows, _ = load_rows(SAMPLE_CSV)
        acc = per_student_condition(rows)
        self.assertEqual(set(acc), {(u, c) for u in ("alumno-a", "alumno-b", "alumno-c")
                                    for c in ("ASSISTED", "UNASSISTED")})
        # A con ayuda: 2 errores en 7 palabras, 9000 ms, 5 ofrecidas / 3 aceptadas.
        self.assertEqual(acc[("alumno-a", "ASSISTED")],
                         {"words": 7, "errors": 2, "ppm_words": 7, "duration_ms": 9000, "offered": 5, "accepted": 3,
                          "pending_free": 0})
        # C con ayuda: la libre sin anotar no cuenta en errores/palabras pero sí en PPM y en pendientes.
        self.assertEqual(acc[("alumno-c", "ASSISTED")],
                         {"words": 3, "errors": 2, "ppm_words": 7, "duration_ms": 11000, "offered": 3, "accepted": 3,
                          "pending_free": 1})
        # B sin ayuda: la omitida no aporta nada (4 palabras, 1 error, 2000 ms).
        self.assertEqual(acc[("alumno-b", "UNASSISTED")],
                         {"words": 4, "errors": 1, "ppm_words": 4, "duration_ms": 2000, "offered": 0, "accepted": 0,
                          "pending_free": 0})

    def test_skipped_rows_do_not_count_suggestions_nor_participation(self):
        header, rows = _sample_rows()
        skipped = rows[7]
        skipped[COLUMNS.index("suggestions_offered")] = "5"
        skipped[COLUMNS.index("suggestions_accepted")] = "3"
        loaded, _ = load_rows(_write_csv([rows[7]]))
        self.assertEqual(per_student_condition(loaded), {})


# --------------------------------------------------------------------------- análisis completo


class AnalyzeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.sha = load_rows(SAMPLE_CSV)
        cls.report = analyze(cls.rows, cls.sha, SAMPLE_CSV.name)

    def test_header_sample_and_flags(self):
        report = self.report
        self.assertEqual(report["code"], "PRUEBA-01")
        self.assertEqual(report["sample"]["completed"], 4)
        self.assertEqual(report["sample"]["excluded"], 1)
        self.assertEqual(report["sample"]["unannotatedFree"], 1)
        self.assertTrue(report["incomplete"])
        self.assertTrue(report["sampleInsufficient"])
        self.assertEqual(report["minSample"], 8)
        self.assertTrue(report["designManual"])
        self.assertEqual(report["datasetSha256"], self.sha)
        self.assertEqual(report["provenance"]["modelVersions"], ["beto-v3", "beto-v4"])
        self.assertEqual(report["provenance"]["appVersions"], ["app-1", "app-2"])

    def test_errors_per_100_words_by_condition(self):
        assisted = self.report["conditions"]["ASSISTED"]
        unassisted = self.report["conditions"]["UNASSISTED"]
        self.assertEqual(assisted["participants"], 3)
        self.assertEqual(unassisted["participants"], 3)
        errors = assisted["errorsPer100Words"]
        self.assertEqual(errors["n"], 3)
        self.assertAlmostEqual(errors["mean"], sum(ERR_ASSISTED) / 3, delta=1e-9)
        expected = bootstrap_mean_interval(ERR_ASSISTED)
        self.assertAlmostEqual(errors["lower"], expected["lower"], delta=1e-9)
        self.assertAlmostEqual(errors["upper"], expected["upper"], delta=1e-9)
        self.assertLessEqual(errors["lower"], errors["mean"])
        self.assertGreaterEqual(errors["upper"], errors["mean"])
        self.assertAlmostEqual(unassisted["errorsPer100Words"]["mean"], sum(ERR_UNASSISTED) / 3, delta=1e-9)

    def test_words_per_minute_and_acceptance_rate(self):
        assisted = self.report["conditions"]["ASSISTED"]
        unassisted = self.report["conditions"]["UNASSISTED"]
        self.assertEqual(assisted["wordsPerMinute"]["n"], 3)
        self.assertAlmostEqual(assisted["wordsPerMinute"]["mean"], sum(PPM_ASSISTED) / 3, delta=1e-9)
        self.assertAlmostEqual(unassisted["wordsPerMinute"]["mean"], sum(PPM_UNASSISTED) / 3, delta=1e-9)
        acceptance = assisted["acceptanceRate"]
        self.assertEqual(acceptance["accepted"], 6)
        self.assertEqual(acceptance["offered"], 9)
        self.assertAlmostEqual(acceptance["ratePct"], 600.0 / 9, delta=1e-9)
        self.assertEqual((acceptance["wilsonLower"], acceptance["wilsonUpper"]), wilson(6, 9))
        self.assertIsNone(unassisted["acceptanceRate"])

    def test_paired_deltas(self):
        errors = self.report["paired"]["errorsPer100Words"]
        self.assertEqual(errors["n"], 3)
        self.assertAlmostEqual(errors["meanAssisted"], sum(ERR_ASSISTED) / 3, delta=1e-9)
        self.assertAlmostEqual(errors["meanUnassisted"], sum(ERR_UNASSISTED) / 3, delta=1e-9)
        self.assertAlmostEqual(errors["meanDelta"], errors["meanAssisted"] - errors["meanUnassisted"], delta=1e-9)
        expected = bootstrap_mean_interval([a - u for a, u in zip(ERR_ASSISTED, ERR_UNASSISTED)])
        self.assertAlmostEqual(errors["bootstrapLower"], expected["lower"], delta=1e-9)
        self.assertAlmostEqual(errors["bootstrapUpper"], expected["upper"], delta=1e-9)
        self.assertLess(errors["tLower"], errors["meanDelta"])
        self.assertGreater(errors["tUpper"], errors["meanDelta"])
        self.assertIsNotNone(errors["t"])
        self.assertTrue(0.0 <= errors["p"] <= 1.0)
        self.assertIsNotNone(errors["dz"])
        ppm = self.report["paired"]["wordsPerMinute"]
        self.assertEqual(ppm["n"], 3)
        self.assertAlmostEqual(ppm["meanUnassisted"], sum(PPM_UNASSISTED) / 3, delta=1e-9)

    def test_per_sentence_stats(self):
        sentences = self.report["sentences"]
        self.assertEqual([s["position"] for s in sentences], [1, 2, 3, 4])
        first = sentences[0]
        self.assertEqual((first["kind"], first["assistance"], first["n"], first["skippedCount"]),
                         ("DICTATED", "ASSISTED", 3, 0))
        self.assertAlmostEqual(first["meanErrors"], 1.0)
        self.assertAlmostEqual(first["meanDurationFirstKeyMs"], (3000 + 2000 + 6000) / 3.0)
        self.assertAlmostEqual(sentences[2]["meanErrors"], 0.5)
        fourth = sentences[3]
        self.assertEqual((fourth["n"], fourth["skippedCount"]), (3, 1))
        self.assertAlmostEqual(fourth["meanErrors"], 0.5)
        self.assertAlmostEqual(fourth["meanDurationFirstKeyMs"], 6000.0)

    def test_excluded_attempt_is_counted_but_not_analyzed(self):
        analyzed = [row for row in self.rows if not row["excluded"]]
        without = analyze(analyzed, "0" * 64, "x.csv")
        self.assertEqual(without["sample"]["completed"], 3)
        self.assertEqual(without["sample"]["excluded"], 0)
        self.assertEqual(without["conditions"], self.report["conditions"])
        self.assertEqual(without["paired"], self.report["paired"])
        self.assertEqual(without["sentences"], self.report["sentences"])

    def test_no_rows_produce_empty_report(self):
        report = analyze([], "0" * 64, "empty.csv")
        self.assertEqual(report["sample"]["completed"], 0)
        self.assertFalse(report["incomplete"])
        self.assertTrue(report["sampleInsufficient"])
        self.assertEqual(report["conditions"]["ASSISTED"]["errorsPer100Words"],
                         {"n": 0, "mean": None, "lower": None, "upper": None})
        self.assertEqual(report["paired"]["errorsPer100Words"]["n"], 0)
        self.assertEqual(report["sentences"], [])

    def test_report_is_json_serializable_without_nan(self):
        text = json.dumps(self.report, allow_nan=False)
        self.assertIn('"ASSISTED"', text)


# --------------------------------------------------------------------------- reconciliación


class CompareTests(unittest.TestCase):
    def setUp(self):
        rows, sha = load_rows(SAMPLE_CSV)
        self.report = analyze(rows, sha, SAMPLE_CSV.name)
        self.backend = json.loads(SAMPLE_JSON.read_text(encoding="utf-8"))

    def _json_with(self, mutate):
        data = json.loads(json.dumps(self.backend))
        mutate(data)
        path = Path(tempfile.mkdtemp()) / "results.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_sample_fixture_matches(self):
        self.assertEqual(compare(self.report, SAMPLE_JSON), [])

    def test_detects_mean_difference(self):
        def mutate(data):
            data["conditions"]["ASSISTED"]["errorsPer100Words"]["mean"] += 1e-3
        diffs = compare(self.report, self._json_with(mutate))
        self.assertEqual(len(diffs), 1)
        self.assertIn("conditions.ASSISTED.errorsPer100Words.mean", diffs[0])

    def test_tolerates_small_float_noise(self):
        def mutate(data):
            data["paired"]["wordsPerMinute"]["bootstrapLower"] += 1e-8
        self.assertEqual(compare(self.report, self._json_with(mutate)), [])

    def test_detects_null_versus_number(self):
        def mutate(data):
            data["conditions"]["ASSISTED"]["acceptanceRate"]["wilsonLower"] = None
        diffs = compare(self.report, self._json_with(mutate))
        self.assertTrue(any("acceptanceRate.wilsonLower" in d for d in diffs))

    def test_detects_sentence_and_sha_differences(self):
        def mutate(data):
            data["sentences"][3]["skippedCount"] = 0
            data["datasetSha256"] = "0" * 64
            data["sample"]["excluded"] = 2
        diffs = compare(self.report, self._json_with(mutate))
        self.assertTrue(any("sentences[3].skippedCount" in d for d in diffs))
        self.assertTrue(any("datasetSha256" in d for d in diffs))
        self.assertTrue(any("sample.excluded" in d for d in diffs))
        self.assertEqual(len(diffs), 3)


class MainTests(unittest.TestCase):
    @staticmethod
    def _run(argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_main_writes_report_and_reconciles(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            code, stdout, _ = self._run([str(SAMPLE_CSV), "--compare-results", str(SAMPLE_JSON), "--out", str(out)])
            self.assertEqual(code, 0)
            self.assertIn("sin diferencias", stdout)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["conditions"]["ASSISTED"]["participants"], 3)
            self.assertEqual(report["datasetSha256"], hashlib.sha256(SAMPLE_CSV.read_bytes()).hexdigest())

    def test_main_returns_1_on_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = json.loads(SAMPLE_JSON.read_text(encoding="utf-8"))
            data["conditions"]["UNASSISTED"]["participants"] = 2
            bad = Path(tmp) / "results.json"
            bad.write_text(json.dumps(data), encoding="utf-8")
            code, stdout, _ = self._run([str(SAMPLE_CSV), "--compare-results", str(bad)])
            self.assertEqual(code, 1)
            self.assertIn("conditions.UNASSISTED.participants", stdout)

    def test_main_returns_2_on_invalid_csv(self):
        header, rows = _sample_rows()
        code, _, stderr = self._run([str(_write_csv(rows, header=header[:-1]))])
        self.assertEqual(code, 2)
        self.assertIn("header", stderr)


if __name__ == "__main__":
    unittest.main()
