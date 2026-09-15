# Teclado Adaptativo — Sistema NLP para Dislexia/Disgrafía

Arquitectura **Domain-Driven Design (DDD)** con pipeline neuro-simbólico-fonético.

## Estructura del proyecto

```
teclado_adaptativo/
│
├── domain/                         # Núcleo del negocio — sin dependencias externas
│   ├── entities/
│   │   ├── correction.py           # Entidad: resultado de corrección
│   │   └── user_feedback.py        # Entidad: feedback del usuario
│   ├── repositories/
│   │   └── interfaces.py           # Puerto (interfaz) IUserHistoryRepository
│   └── services/
│       ├── error_analyzer.py       # Clasificador de errores (dislexia / disgrafía)
│       └── noise_injector.py       # Inyector de ruido clínico (data augmentation)
│
├── application/                    # Casos de uso — orquestación sin detalles técnicos
│   ├── dtos.py                     # Data Transfer Objects
│   └── use_cases/
│       ├── correct_text.py         # Caso de uso: corregir texto
│       ├── save_feedback.py        # Caso de uso: guardar feedback
│       ├── train_model.py          # Caso de uso: entrenamiento del modelo base
│       └── evaluate_model.py       # Caso de uso: evaluación WER/CER/improvement
│
├── infrastructure/                 # Adaptadores técnicos — implementaciones concretas
│   ├── ml/
│   │   ├── t5_model.py             # T5 base (+ utilidades LoRA genéricas) y guardas
│   │   ├── guards.py               # Guarda anti-alucinación (puro Python)
│   │   └── dataset_loader.py       # Descarga OPUS-100 y genera pares de entrenamiento
│   ├── nlp/
│   │   ├── phonetic_engine.py      # Motor fonético + restauración de tildes
│   │   ├── context_judge.py        # BETO MLM para desambiguar homófonos
│   │   ├── grammar_rules.py        # Gramática por reglas (haber impersonal, gustar, número)
│   │   ├── alternatives.py         # Selector puro de alternativas (hasta 3 solo si hay ambigüedad)
│   │   └── correction_pipeline.py  # Pipeline global de corrección (reglas + BETO + T5)
│   └── persistence/
│       └── csv_user_repository.py  # Repositorio CSV del feedback aceptado (curación offline)
│
├── interfaces/                     # Entrypoints externos
│   └── api/
│       └── routes.py               # Rutas Flask (/interno/corregir, /interno/feedback)
│
├── main.py                         # Composition Root — ensambla el pipeline global y arranca Flask
├── train.py                        # CLI: entrenamiento del modelo base
├── train_grammar_lora.py           # CLI: entrenamiento del LoRA gramatical GLOBAL
├── scripts/merge_grammar_lora.py   # Fusiona el LoRA global en models/t5_correction
├── evaluate.py                     # Evaluación offline: P/R/F0.5 por edición (exact_token_edits_v1) + WER/CER
├── setup_model.py                  # Descarga el T5 base a models/t5_base (nunca pisa models/t5_correction)
├── test_global_runtime.py          # Tests del runtime global (fakes, sin torch)
├── test_alternatives.py            # Tests del selector de alternativas y de la ambigüedad BETO (sin torch)
├── test_evaluate_metrics.py        # Tests del scorer, del informe y de la guarda de setup_model (sin torch)
├── test_lora_guards.py             # Tests de la guarda anti-alucinación
└── requirements.txt
```

## Arquitectura de runtime: un solo modelo global

Todos los alumnos son atendidos por **el mismo pipeline** (reglas + BETO +
T5 base fusionado con un único LoRA gramatical). En el runtime **no existe**
memoria por alumno, vocabulario personalizado, adaptadores `models/loras/<studentId>`
ni entrenamiento en segundo plano; `studentId` solo viaja de vuelta en la
respuesta. El feedback aceptado (`POST /interno/feedback`) se persiste en
`data/users/<studentId>_history.csv` únicamente para curarlo offline como dato
de entrenamiento del LoRA global.

Cada respuesta de `POST /interno/corregir` incluye `modelVersion`, tomado de la
variable `MODEL_VERSION`, si no de `models/grammar_lora/manifest.json`
(clave `modelVersion`) y, en su defecto, `global-lora-unversioned`.

```
POST /interno/corregir  {"originalText": "...", "studentId": "..."}
→ {"studentId", "correctedText", "processingTimeMs", "suggestions": [...], "modelVersion"}

POST /interno/feedback  {"studentId", "originalText", "selectedSuggestion", "accepted"}
→ 204
```

Despliegue: `docker-compose.yml` levanta un único servicio `api` (gunicorn,
`--workers 1`) con los volúmenes `models/` y `data/`; no hay Redis ni worker.

## Archivos eliminados (scripts de diagnóstico, no forman parte del sistema)

| Archivo original        | Motivo de eliminación                              |
|-------------------------|----------------------------------------------------|
| `check_torch.py`        | Script de diagnóstico puntual, no es producción    |
| `debug_load.py`         | Script de debug puntual, no es producción          |

## Archivos refactorizados (DDD)

| Archivo original        | Destino DDD                                                         |
|-------------------------|---------------------------------------------------------------------|
| `beto_model.py`         | `infrastructure/ml/beto_model.py` + lógica de cuantización integrada |
| `dataset_loader.py`     | `infrastructure/ml/dataset_loader.py`                               |
| `text_preprocessor.py`  | `domain/services/noise_injector.py`                                 |
| `error_analyzer.py`     | `domain/services/error_analyzer.py`                                 |
| `evaluator.py`          | `application/use_cases/evaluate_model.py`                           |
| `optimize_model.py`     | Método `quantize_and_save()` en `infrastructure/ml/beto_model.py`   |
| `user_data_manager.py`  | `infrastructure/persistence/csv_user_repository.py`                 |
| `inference_api.py`      | Separado en 4 módulos (phonetic_engine, context_judge, correction_pipeline, routes) |
| `train.py`              | `train.py` (CLI) + `application/use_cases/train_model.py`           |
| `train_user.py`         | Eliminado: no hay fine-tuning por alumno (LoRA global en `train_grammar_lora.py`) |

## Uso



```bash
python -m venv .venv
# Instalar dependencias
python -m pip install -r requirements.txt

# Arrancar servidor de corrección (PORT, default 8080; MODEL_VERSION opcional)
python main.py

# Tests sin GPU ni modelo
python -m unittest -v test_global_runtime test_alternatives test_evaluate_metrics
python test_lora_guards.py

# Evaluar el pipeline (ver sección "Evaluación"); --t5-dir añade el T5 global fusionado
python evaluate.py --no-beto --out reports/rules-dev.json
python evaluate.py --t5-dir models/t5_correction --model-version "<modelVersion>" --out reports/eval.json

# Descargar el T5 base (a models/t5_base; models/t5_correction está protegido)
python setup_model.py

# Entrenar modelo base
python train.py --epochs 3 --batch_size 2

# Entrenar y fusionar el LoRA gramatical GLOBAL (único adaptador del sistema)
python train_grammar_lora.py
python scripts/merge_grammar_lora.py
```

## Pipeline de corrección (global, idéntico para todos los alumnos)

| Capa | Descripción |
|------|-------------|
| 1    | Palabras ya acentuadas: se respetan (solo restauración de tilde/ñ) |
| 1.1  | Correcciones manuales de alta prioridad (abreviaturas de chat, homófonos frecuentes) |
| 1.2  | Escudo de palabras cortas (evita alucinaciones en "y", "a", "mi") |
| 1.3  | Fonética dirigida a la palabra de mayor frecuencia |
| 2    | SymSpell (Levenshtein ≤ 2) + restauración de ñ y tildes |
| 2.5  | Desambiguación contextual de homófonos y tildes con BETO MLM |
| 3    | Gramática por reglas (haber impersonal, gustar, concordancia de número) |
| 4    | Refinamiento gramatical con T5 + LoRA global (beam search, 3 candidatos con score), aceptado solo si pasa la guarda anti-alucinación |
| 5    | Selección de alternativas: una recomendación, o hasta 3 opciones solo si la oración es ambigua (ver abajo) |

## Evaluación: Precisión, Recall y F0.5 a nivel de edición (`evaluate.py`)

`evaluate.py` evalúa **la sugerencia recomendada** (`correct(text, {})[0]`; las
alternativas de la capa 5 no se puntúan) contra un dataset
`categoria|entrada|esperado_1|esperado_2...` (una o más referencias; menos de
tres campos es un error) y guarda un informe JSON versionado. La métrica
principal es F0.5 sobre ediciones con el scorer `exact_token_edits_v1`; WER,
CER, mejora, exactitud y latencia se conservan como diagnósticos.

Scorer `exact_token_edits_v1`:

- tokens: `re.findall(r"\w+|[^\w\s]", texto)` (palabras y signos sueltos);
- ediciones: opcodes no-equal de `difflib.SequenceMatcher` entre la entrada y
  el texto, como `(inicio, fin, tokens_reemplazo)` sobre la entrada. Un bloque
  de reemplazo con el mismo número de tokens a ambos lados (`arbol esta` →
  `árbol está`) se divide palabra a palabra para que una corrección parcial
  cuente como TP + FN y no como FP + FN; `ala` → `a la` sigue siendo una edición;
- por frase se elige la referencia con mejor F0.5 (en empate, más TP y menos
  FP+FN); TP = predichas ∩ gold, FP = predichas − gold, FN = gold − predichas,
  sumadas sobre el corpus y por categoría;
- P = TP/(TP+FP), R = TP/(TP+FN), F0.5 = 1.25·P·R/(0.25·P+R), **0.0 cuando el
  denominador es 0** (la convención que valida el backend con tolerancia 1e-6;
  por eso la categoría `correcto`, sin ediciones gold, muestra P = R = 0 y solo
  aporta falsos positivos).

Informe JSON (`--out`): `development` (true para `data/eval_gold.csv`,
`pruebas.txt` o `--development`; esos informes no se registran en el backend),
`timestamp`, `source`, `modelVersion` (`--model-version` > `MODEL_VERSION` >
`models/grammar_lora/manifest.json` > `global-lora-unversioned`), `modelDir`
(`--t5-dir`), `scorerVersion`, `dataset`, `datasetSha256` (SHA-256 de los
bytes exactos del archivo), `global`, `por_categoria`, `latency_ms_avg`,
`casos` y el bloque `technicalEvaluation`, que es exactamente lo que valida el
panel Vue y acepta `POST /api/v1/research/technical-evaluations`:

```json
{"modelVersion": "...", "datasetSha256": "<64 hex>", "scorerVersion": "exact_token_edits_v1",
 "precision": 0.98, "recall": 1.0, "fZeroFive": 0.985,
 "truePositives": 54, "falsePositives": 1, "falseNegatives": 0,
 "categories": [{"category": "tilde", "tp": 12, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f05": 1.0}, ...]}
```

Fuentes: `--no-beto` (solo reglas), por defecto reglas + BETO, `--t5-dir
models/t5_correction` añade el T5 global fusionado (solo directorios locales
completos: nunca descarga), `--source http --url ...` mide un servidor. El
holdout final se identifica por su SHA-256 y no se usa para nada más.

Informes de desarrollo sobre `data/eval_gold.csv` (38 casos, GPU):

| fuente               | TP | FP | FN | P     | R     | F0.5  | exactos | ms/frase |
|----------------------|----|----|----|-------|-------|-------|---------|----------|
| reglas               | 45 | 0  | 9  | 1.000 | 0.833 | 0.962 | 29/38   | 0        |
| reglas + BETO        | 53 | 2  | 1  | 0.964 | 0.981 | 0.967 | 35/38   | 54       |
| reglas + BETO + T5   | 54 | 1  | 0  | 0.982 | 1.000 | 0.985 | 37/38   | 268      |

## Alternativas: hasta tres opciones solo cuando la oración es ambigua

`suggestions[0]` es siempre la recomendada. En el caso común hay una sola
opción (o dos: refinado de T5 + texto base de reglas+BETO, como siempre).
Solo cuando el texto admite lecturas distintas se ofrecen hasta **tres**.
La selección es pura (`infrastructure/nlp/alternatives.py`, sin torch) y
recibe como candidatos los 3 beams de T5 con su `sequences_scores`, las
"segundas lecturas" de BETO y el texto base.

Un candidato se ofrece únicamente si:

| Regla | Criterio |
|-------|----------|
| (a) seguro     | pasa `is_safe_refinement` respecto al texto base (longitud 0.6–1.5x, ≤ max(3, n/3) palabras cambiadas) |
| (b) distinto   | tras normalizar espacios y puntuación final es distinto del texto original y de los ya elegidos (el original nunca es una corrección) |
| (c) otra lectura | su conjunto de índices de palabra editados respecto al original (con tildes, sin la mayúscula inicial) es distinto del de los ya elegidos: dos textos que corrigen las mismas palabras son una interpretación y su variante, no dos |
| (d) empate     | su score está a menos de `_SCORE_MARGIN = 0.3` del mejor candidato (calibrado con beams reales: `juega→juegan` 0.09, `juega→jueguen` 0.14, `fue→fueron` 0.25 entran; `voy→iré` 0.37, `luego→después` 0.40, `ayer→anoche` 0.48, `ellos→él` 0.91 quedan fuera) |
| (e) léxico     | cada palabra que edita es una variante cercana de la original (similitud sin tildes ≥ 0.6): `esta/está`, `tubo/tuvo`, `fue/fueron` sí; `luego/después`, `voy/iré` (paráfrasis) no |

El texto base entra justo detrás del mejor candidato con su mismo score y
solo se filtra por (b): si T5 difiere de la base, la base se conserva como
alternativa. Si ningún candidato cumple, se devuelve solo la base.

Segunda fuente de ambigüedad, en la capa 2.5 (`_disambiguate_context`): cuando
BETO no separa las dos mejores lecturas de un homófono (`esta/está`,
`se/sé`, `tubo/tuvo`) por más de `margen · HOMOPHONE_AMBIGUITY_RATIO` (1.0,
es decir, toda la zona por debajo del umbral de sobrescritura), el texto no
cambia pero se genera la variante con la segunda lectura, con
score = −|diferencia| (más cerca de 0 cuanto más empatado), anclado al mejor
beam de T5 para que entre por (d) solo si el empate cabe en `_SCORE_MARGIN`.
Máximo una alternativa por posición y 3 posiciones (las más empatadas). Una
lectura más de `_AMBIGUITY_MAX_FREQ_RATIO` (20) veces más rara que la escrita
no cuenta como ambigüedad (`hoy/holly`, `mis/miss`: la PLL media de BETO
infla las palabras de varios subtokens).

Salida real del modelo fusionado (tests manuales, `test_alternatives.py`
cubre las reglas con fakes):

```
mañana voy al parque con mis amigos  → ["mañana voy al parque con mis amigos"]            (1: ya estaba bien)
esta bien, nos vemos luego           → ["está bien, nos vemos luego"]                     (1: T5 y BETO coinciden)
a mi me gusta que ellos juega mucho  → ["a mí me gusta que ellos juegan mucho",
                                        "a mí me gusta que ellos juega mucho"]            (2: refinado + base)
se que no vendra hoy                 → ["sé que no vendrá hoy", "se que no vendrá hoy"]   (2: se/sé empatados en BETO)
hola como estas                      → ["hola como estás", "hola cómo estás"]             (2: afirmación / pregunta)
```

El contrato HTTP no cambia (`suggestions: list[str]`); el teclado muestra como
mucho tres globos y nunca uno idéntico al texto original.
