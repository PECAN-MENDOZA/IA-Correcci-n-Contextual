"""
test_fetch_external_data.py

Tests de `scripts/fetch_external_data.py`: descarga verificada + alineación de
COWS-L2H (Task 0 del plan de mejora del modelo). Puros: solo librería
estándar, directorios temporales y un `urlopen` simulado; no hacen red.

Uso:
    .venv\\Scripts\\python.exe -m unittest test_fetch_external_data -v
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.fetch_external_data import (
    EDIT_RATIO_THRESHOLD,
    HOLDOUT_FRACTION,
    SPLIT_SEED,
    align_essay,
    build_manifest,
    build_pairs_by_essay,
    download_file,
    format_row,
    parse_cowsl2h_csv,
    sha256_bytes,
    sha256_file,
    split_by_essay,
    split_sentences,
    usable_essays,
    verify_raw_files,
    word_edit_ratio,
    write_pairs_csv,
)


class _FakeResponse:
    """Contexto que imita lo que devuelve `urlopen(...)`, sin red."""

    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _fake_opener(data: bytes):
    def opener(url, timeout=None):
        return _FakeResponse(data)
    return opener


class SplitSentencesTests(unittest.TestCase):
    def test_splits_on_period_before_uppercase(self):
        text = "Hola. Como estas. Bien."
        self.assertEqual(split_sentences(text), ["Hola.", "Como estas.", "Bien."])

    def test_splits_on_question_and_exclamation(self):
        text = "¿Como estas? Muy bien! Que bueno."
        self.assertEqual(split_sentences(text), ["¿Como estas?", "Muy bien!", "Que bueno."])

    def test_does_not_split_on_abbreviation_period_followed_by_lowercase(self):
        text = "Vive en la calle etc. cerca del parque."
        self.assertEqual(split_sentences(text), ["Vive en la calle etc. cerca del parque."])

    def test_empty_text_returns_empty_list(self):
        self.assertEqual(split_sentences(""), [])
        self.assertEqual(split_sentences("   "), [])

    def test_strips_whitespace_and_ignores_blank_fragments(self):
        text = "Uno.   Dos.  "
        self.assertEqual(split_sentences(text), ["Uno.", "Dos."])


class WordEditRatioTests(unittest.TestCase):
    def test_identical_sentences_have_zero_ratio(self):
        self.assertEqual(word_edit_ratio("el gato come", "el gato come"), 0.0)

    def test_small_correction_is_below_threshold(self):
        ratio = word_edit_ratio("el perro come rapido", "el perro come rápido")
        self.assertLess(ratio, EDIT_RATIO_THRESHOLD)

    def test_full_rewrite_is_above_threshold(self):
        ratio = word_edit_ratio(
            "el perro come mucho todos los dias",
            "mi hermana fue al mercado ayer por la tarde",
        )
        self.assertGreater(ratio, EDIT_RATIO_THRESHOLD)

    def test_empty_input_and_nonempty_output_is_full_ratio(self):
        self.assertEqual(word_edit_ratio("", "hola"), 1.0)

    def test_both_empty_is_zero(self):
        self.assertEqual(word_edit_ratio("", ""), 0.0)


class AlignEssayTests(unittest.TestCase):
    """Ensayo de ejemplo (sintético, con la forma de una fila de COWS-L2H) que
    ejercita identidad, corrección válida, reescritura descartada y '|'."""

    def test_example_essay_identity_correction_and_rewrite(self):
        essay = (
            "Hoy voy al parque. Mi hermano juega futbol. "
            "El perro corrio muy rapido por el jardin."
        )
        corrected1 = (
            "Hoy voy al parque. Mi hermano juega fútbol. "
            "Mi tía preparó una torta de chocolate para la fiesta del sábado pasado."
        )
        result = align_essay(essay, corrected1)
        self.assertTrue(result["aligned"])
        self.assertEqual(result["n_sentences"], 3)
        # "Hoy voy al parque." no cambia -> par de identidad, no en pairs.
        self.assertEqual(result["identity_pairs"], [("Hoy voy al parque.", "Hoy voy al parque.")])
        # "Mi hermano juega futbol." -> corrección puntual (tilde), se conserva.
        self.assertEqual(
            result["pairs"],
            [("Mi hermano juega futbol.", "Mi hermano juega fútbol.", None)],
        )
        # La tercera oración es una reescritura total -> descartada por el filtro 40 %.
        self.assertEqual(result["discarded_filter"], 1)
        self.assertEqual(result["discarded_pipe"], 0)

    def test_not_aligned_when_sentence_counts_differ(self):
        essay = "Una oracion. Otra oracion."
        corrected1 = "Una oracion corregida."
        result = align_essay(essay, corrected1)
        self.assertFalse(result["aligned"])
        self.assertEqual(result["pairs"], [])
        self.assertEqual(result["identity_pairs"], [])
        self.assertEqual(result["discarded_filter"], 0)

    def test_adds_esperado_2_when_corrected2_also_aligns(self):
        # Frase larga para que las 3 palabras con tilde no superen el 40 %.
        essay = "El nino corrio muy rapido ayer en el parque."
        corrected1 = "El niño corrió muy rápido ayer en el parque."
        corrected2 = "El niño corría muy rápido ayer en el parque."
        result = align_essay(essay, corrected1, corrected2)
        self.assertEqual(
            result["pairs"],
            [(essay, corrected1, corrected2)],
        )

    def test_ignores_corrected2_when_it_does_not_align(self):
        essay = "El nino corrio muy rapido ayer en el parque. Fue al parque bonito."
        corrected1 = "El niño corrió muy rápido ayer en el parque. Fue al parque bonito."
        corrected2 = "El niño corrió muy rápido ayer."  # una sola oración: no alinea
        result = align_essay(essay, corrected1, corrected2)
        self.assertEqual(
            result["pairs"],
            [("El nino corrio muy rapido ayer en el parque.",
              "El niño corrió muy rápido ayer en el parque.", None)],
        )
        self.assertEqual(
            result["identity_pairs"],
            [("Fue al parque bonito.", "Fue al parque bonito.")],
        )

    def test_pipe_in_sentence_is_discarded_not_rewritten(self):
        essay = "El nino comio | manzanas. Fue al parque."
        corrected1 = "El niño comió | manzanas. Fue al parque."
        result = align_essay(essay, corrected1)
        self.assertEqual(result["discarded_pipe"], 1)
        self.assertEqual(result["pairs"], [])
        self.assertEqual(result["identity_pairs"], [("Fue al parque.", "Fue al parque.")])


class CsvParsingTests(unittest.TestCase):
    SAMPLE_CSV = (
        "id,essay,corrected1,corrected2\n"
        '100,"Hoy voy al parque. Mi hermano juega futbol.",'
        '"Hoy voy al parque. Mi hermano juega fútbol.",""\n'
        '101,"Ensayo sin correccion del profesor.","",""\n'
        '102,"Otro ensayo con dato.","Otro ensayo con dato corregido.","Otro ensayo con dato revisado."\n'
    )

    def test_parse_returns_all_raw_rows(self):
        rows = parse_cowsl2h_csv(self.SAMPLE_CSV)
        self.assertEqual(len(rows), 3)

    def test_usable_essays_skips_empty_corrected1(self):
        rows = parse_cowsl2h_csv(self.SAMPLE_CSV)
        essays = usable_essays(rows)
        ids = [e["id"] for e in essays]
        self.assertEqual(ids, ["100", "102"])

    def test_usable_essays_keeps_corrected2(self):
        rows = parse_cowsl2h_csv(self.SAMPLE_CSV)
        essays = usable_essays(rows)
        essay_102 = next(e for e in essays if e["id"] == "102")
        self.assertEqual(essay_102["corrected2"], "Otro ensayo con dato revisado.")

    def test_usable_essays_collapses_embedded_newlines(self):
        # Regresión: varios ensayos reales de COWS-L2H traen '\n' dentro del
        # propio texto (párrafos); sin normalizar, una fila del CSV de salida
        # terminaría partida en más de una línea física y rompería el formato
        # de una-fila-por-línea que lee evaluate.load_dataset.
        rows = [{
            "id": "200",
            "essay": "Primera parte.\nSegunda parte del mismo ensayo.",
            "corrected1": "Primera parte.\r\nSegunda parte del mismo ensayo.",
            "corrected2": "",
        }]
        essays = usable_essays(rows)
        self.assertNotIn("\n", essays[0]["essay"])
        self.assertNotIn("\r", essays[0]["corrected1"])
        self.assertEqual(essays[0]["essay"], "Primera parte. Segunda parte del mismo ensayo.")


class BuildPairsByEssayTests(unittest.TestCase):
    def test_aggregates_counts_across_essays(self):
        essays = [
            {"id": "1", "essay": "Hoy voy al parque. Mi hermano juega futbol.",
             "corrected1": "Hoy voy al parque. Mi hermano juega fútbol.", "corrected2": ""},
            {"id": "2", "essay": "Una oracion. Otra oracion.",
             "corrected1": "Una oracion distinta y mucho mas larga que la original entera.",
             "corrected2": ""},
        ]
        pairs_by_essay, stats = build_pairs_by_essay(essays)
        self.assertEqual(stats["essays_con_corrected1"], 2)
        self.assertEqual(stats["essays_alineados"], 1)
        self.assertEqual(stats["pares_identidad"], 1)
        self.assertEqual(stats["pares_alineados"], 1)
        self.assertIn("1", pairs_by_essay)
        self.assertNotIn("2", pairs_by_essay)


class SplitByEssayTests(unittest.TestCase):
    def test_is_deterministic_for_a_fixed_seed(self):
        ids = [str(i) for i in range(100)]
        dev1, holdout1 = split_by_essay(ids, seed=SPLIT_SEED, holdout_fraction=HOLDOUT_FRACTION)
        dev2, holdout2 = split_by_essay(ids, seed=SPLIT_SEED, holdout_fraction=HOLDOUT_FRACTION)
        self.assertEqual(dev1, dev2)
        self.assertEqual(holdout1, holdout2)

    def test_partition_is_exhaustive_and_disjoint(self):
        ids = [str(i) for i in range(57)]
        dev, holdout = split_by_essay(ids, seed=SPLIT_SEED, holdout_fraction=HOLDOUT_FRACTION)
        self.assertEqual(dev | holdout, set(ids))
        self.assertEqual(dev & holdout, set())

    def test_holdout_is_about_30_percent(self):
        ids = [str(i) for i in range(1000)]
        _, holdout = split_by_essay(ids, seed=SPLIT_SEED, holdout_fraction=HOLDOUT_FRACTION)
        self.assertEqual(len(holdout), 300)


class WritePairsCsvTests(unittest.TestCase):
    def test_format_row_omits_esperado_2_when_absent(self):
        self.assertEqual(
            format_row("sin_anotar", "el nino corrio", "el niño corrió"),
            "sin_anotar|el nino corrio|el niño corrió",
        )

    def test_format_row_includes_esperado_2_when_present(self):
        self.assertEqual(
            format_row("sin_anotar", "a", "b", "c"),
            "sin_anotar|a|b|c",
        )

    def test_write_pairs_csv_is_readable_by_evaluate_load_dataset(self):
        pairs_by_essay = {
            "1": [("el nino corrio", "el niño corrió", None)],
            "2": [("ella juga futbol", "ella juega fútbol", "ella jugaba fútbol")],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs-dev.csv"
            n = write_pairs_csv(path, pairs_by_essay, ["1", "2"], header_comment="# comentario\n")
            self.assertEqual(n, 2)

            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from evaluate import load_dataset
            rows = load_dataset(str(path))
            self.assertEqual(len(rows), 2)
            cats = {r[0] for r in rows}
            self.assertEqual(cats, {"sin_anotar"})
            row2 = next(r for r in rows if r[1] == "ella juga futbol")
            self.assertEqual(row2[2], ["ella juega fútbol", "ella jugaba fútbol"])


class DownloadFileTests(unittest.TestCase):
    def test_downloads_and_returns_hash_matching_content(self):
        data = b"contenido de prueba"
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "archivo.csv"
            digest = download_file("http://ejemplo/archivo.csv", dest, opener=_fake_opener(data))
            self.assertEqual(digest, hashlib.sha256(data).hexdigest())
            self.assertEqual(dest.read_bytes(), data)

    def test_raises_on_hash_mismatch(self):
        data = b"contenido inesperado"
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "archivo.csv"
            with self.assertRaises(ValueError):
                download_file("http://ejemplo/archivo.csv", dest, expected_sha256="0" * 64,
                               opener=_fake_opener(data))

    def test_skips_download_when_cached_file_already_matches(self):
        data = b"contenido cacheado"
        expected = hashlib.sha256(data).hexdigest()
        calls = []

        def opener(url, timeout=None):
            calls.append(url)
            return _FakeResponse(data)

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "archivo.csv"
            dest.write_bytes(data)
            digest = download_file("http://ejemplo/archivo.csv", dest, expected_sha256=expected, opener=opener)
            self.assertEqual(digest, expected)
            self.assertEqual(calls, [])  # no debió llamar a la red


class VerifyRawFilesTests(unittest.TestCase):
    def test_reports_ok_missing_and_mismatched(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            good = b"bien"
            (raw_dir / "a.csv").write_bytes(good)
            (raw_dir / "b.csv").write_bytes(b"cambiado")
            hashes = {
                "a.csv": hashlib.sha256(good).hexdigest(),
                "b.csv": hashlib.sha256(b"original").hexdigest(),
                "c.csv": hashlib.sha256(b"nunca-descargado").hexdigest(),
            }
            results = verify_raw_files(raw_dir, hashes)
            self.assertEqual(results, {"a.csv": "ok", "b.csv": "no_coincide", "c.csv": "falta"})


class Sha256HelpersTests(unittest.TestCase):
    def test_sha256_file_matches_sha256_bytes(self):
        data = b"hola mundo"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.txt"
            path.write_bytes(data)
            self.assertEqual(sha256_file(path), sha256_bytes(data))


class BuildManifestTests(unittest.TestCase):
    def test_manifest_contains_required_fields(self):
        stats = {
            "essays_con_corrected1": 10,
            "essays_con_corrected2": 3,
            "essays_alineados": 8,
            "pares_alineados": 40,
            "pares_descartados_filtro": 5,
            "pares_descartados_pipe": 1,
            "pares_identidad": 6,
            "total_filas_csv": 12,
        }
        manifest = build_manifest(
            source_commit="ebb11724f258f3ed377a27ed34f08897a8e5639c",
            file_hashes={"a.csv": "x" * 64},
            stats=stats,
            dev_essay_ids={"1", "2"},
            holdout_essay_ids={"3"},
            dev_rows=30,
            holdout_rows=10,
            seed=42,
            holdout_fraction=0.3,
            holdout_sha256="y" * 64,
            holdout_file_name=r"C:\ruta\holdout-cowsl2h.csv",
        )
        self.assertEqual(manifest["commit"], "ebb11724f258f3ed377a27ed34f08897a8e5639c")
        self.assertEqual(manifest["particion"]["semilla"], 42)
        self.assertEqual(manifest["particion"]["dev_essays"], 2)
        self.assertEqual(manifest["particion"]["holdout_essays"], 1)
        self.assertEqual(manifest["particion"]["dev_filas"], 30)
        self.assertEqual(manifest["particion"]["holdout_filas"], 10)
        self.assertEqual(manifest["holdout"]["sha256"], "y" * 64)
        self.assertEqual(manifest["pares_identidad_no_incluidos"], 6)
        # serializable sin sorpresas
        json.dumps(manifest, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
