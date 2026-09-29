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
│   ├── persistence/
│   │   └── csv_user_repository.py  # Repositorio CSV del feedback aceptado (curación offline)
│   └── versioning.py               # modelVersion compartido: env > compuesto desde manifiestos > LoRA > default (solo stdlib)
│
├── interfaces/                     # Entrypoints externos
│   └── api/
│       └── routes.py               # Rutas Flask (/interno/corregir, /interno/feedback)
│
├── main.py                         # Composition Root — ensambla el pipeline global y arranca Flask
├── train.py                        # CLI: entrenamiento del modelo base
├── train_grammar_lora.py           # CLI: entrenamiento del LoRA gramatical GLOBAL (semillas antes del modelo)
├── scripts/merge_grammar_lora.py   # Fusiona el LoRA global (base + revisión del manifiesto) y propaga el manifiesto
├── scripts/model_manifest.py       # Manifiestos reproducibles (training-manifest.json / model-manifest.json) y --write-current
├── scripts/calibrate_alternatives.py  # Calibración / prueba de aceptación (--check) de la capa 5 con el modelo real
├── scripts/analyze_sentence_tests.py  # Reproducción offline de los resultados de una prueba de oraciones (solo stdlib)
├── evaluate.py                     # Evaluación offline: P/R/F0.5 por edición (exact_token_edits_v1) + WER/CER
├── setup_model.py                  # Descarga el T5 base a models/t5_base (nunca pisa models/t5_correction)
├── data/eval_gold.csv              # Dataset de desarrollo (38 casos); el holdout final vive fuera del repo
├── data/ambiguity_calibration.csv  # 39 frases etiquetadas (expected_options, forbidden_readings) para la capa 5
├── test_fixtures/sentence_tests/   # CSV de exportación sintético + JSON de resultados para el análisis offline
├── test_global_runtime.py          # Tests del runtime global (fakes, sin torch)
├── test_alternatives.py            # Tests del selector de alternativas, la ambigüedad BETO y el --check (sin torch)
├── test_correction_layers.py       # Tests de CorrectionPipeline.correct() de punta a punta con fakes (sin torch)
├── test_evaluate_metrics.py        # Tests del scorer, del informe (réplica del validador del panel) y de setup_model (sin torch)
├── test_versioning.py              # Tests de infrastructure/versioning.py, incl. la versión compuesta y servida (sin torch)
├── test_model_manifest.py          # Tests de scripts/model_manifest.py y de la estructura de train/merge (sin torch)
├── test_analyze_sentence_tests.py  # Tests del análisis offline de pruebas de oraciones (sin torch)
├── test_lora_guards.py             # Tests de las guardas anti-alucinación y léxica
├── test_modelo.py                  # Script MANUAL de humo por HTTP contra un servidor arrancado (no es unittest)
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
variable `MODEL_VERSION`; si no, compuesto desde
`models/t5_correction/model-manifest.json` con el formato
`<base-tag>@<hash8>+<lora-tag>@<hash8>` (ver "Artefactos reproducibles");
si no hay manifiesto del modelo fusionado, el `modelVersion` de
`models/grammar_lora/training-manifest.json` (o `manifest.json`) y, en su
defecto, `global-lora-unversioned`. La regla vive en
`infrastructure/versioning.py` (`resolve_model_version`,
`compose_model_version` y `served_model_version`, solo librería estándar) y la
comparten `main.py`, el caso de uso y `evaluate.py`, para que las correcciones
y los informes se tracen al mismo identificador. Dos garantías:

- **Nunca se mezclan generaciones**: la versión compuesta sale solo del
  manifiesto del modelo fusionado (su `model_sha256` y su `adapter_sha256`).
  Si `training-manifest.json` describe un LoRA reentrenado y aún no fusionado,
  se avisa (`adaptador entrenado ≠ adaptador fusionado`) y la versión sigue
  describiendo los pesos que se sirven.
- **La versión describe el pipeline servido**: con `ENABLE_T5=false` o si la
  carga del T5 falla, `main.py` se niega a arrancar (`[ERROR] ...`) salvo que
  `MODEL_VERSION` sea explícita; el backend congela esa cadena por ejecución y
  no puede atribuir a `beto-t5-base@…+global-lora-v1@…` correcciones hechas
  solo con reglas + BETO.
- **Un manifiesto fusionado inválido falla cerrado**: si
  `models/t5_correction/model-manifest.json` no existe se cae a los escalones
  siguientes, pero si existe y es ilegible, no es un objeto o le falta
  `model_sha256`/`adapter_sha256`, `resolve_model_version` lanza
  `VersionResolutionError` (nunca degrada a `global-lora-v1@…` ni al default,
  que nombrarían un adaptador que quizá no es el fusionado). Con el T5
  cargado, `served_model_version` solo acepta una versión compuesta válida
  (`<tag>@<hash8>+<tag>@<hash8>`) o `MODEL_VERSION` explícita; en cualquier
  otro caso el servidor no arranca (`scripts/model_manifest.py --show`
  devuelve código 1 con el mismo error).

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
python -m unittest -v test_global_runtime test_alternatives test_correction_layers test_evaluate_metrics test_versioning test_model_manifest test_analyze_sentence_tests test_amalgams test_confusions
python test_lora_guards.py

# Prueba de aceptación de la capa 5 con el modelo real (offline; código 1 si falla)
python scripts/calibrate_alternatives.py --check

# Evaluar el pipeline (ver sección "Evaluación"); --t5-dir añade el T5 global fusionado
# (sin --t5-dir o con --no-beto, --model-version es obligatoria)
python evaluate.py --no-beto --model-version rules-dev --out reports/rules-dev.json
python evaluate.py --model-version rules-beto-dev --out reports/rules-beto-dev.json
python evaluate.py --t5-dir models/t5_correction --out reports/rules-beto-t5-dev.json   # versión desde su model-manifest.json

# Descargar el T5 base (a models/t5_base; models/t5_correction está protegido)
python setup_model.py

# Entrenar modelo base
python train.py --epochs 3 --batch_size 2

# Entrenar y fusionar el LoRA gramatical GLOBAL (único adaptador del sistema);
# cada paso escribe su manifiesto (ver "Artefactos reproducibles").
# IMPORTANTE: el dataset debe ser data/training_pairs_v4.csv (generales + concordancia +
# subjuntivo + controles de indicativo/prospectivo + predicado + té/él, 30 896 pares; ver
# data/training_pairs_v4.README.md, que también registra los intentos v4a-v4c). Entrenar
# solo con training_pairs_clean.csv produce un adaptador que NO aprende concordancia (9/12 en
# data/test_agreement.csv) ni subjuntivo irregular (3/16 en data/test_subjunctive.csv);
# ocurrió el 2026-09-12 y se corrigió el 2026-09-16 (v2 concordancia, v3 subjuntivo) y el
# 2026-09-19 (v4: sobrecorrección al subjuntivo tras "aunque", predicado plural, té/él).
# Para regenerar los pares dirigidos: scripts/generate_agreement_data.py,
# scripts/generate_subjunctive_data.py y scripts/generate_v4_data.py (semillas fijas).
$env:CSV_PATH = 'data/training_pairs_v4.csv'
python train_grammar_lora.py           # después, poner "lora_tag": "global-lora-v4" en models/grammar_lora/training-manifest.json
python scripts/merge_grammar_lora.py
# Las SEIS regresiones internas, todas antes de promover (valores de v4 + reglas del 2026-09-29):
python evaluate.py --dataset data/eval_gold.csv            --t5-dir models/t5_correction --development --out reports/gold.json             # 47/48 (dónde)
python evaluate.py --dataset data/test_agreement.csv       --t5-dir models/t5_correction --development --out reports/agreement.json        # 15/15
python evaluate.py --dataset data/test_subjunctive.csv     --t5-dir models/t5_correction --development --out reports/subjunctive.json      # 19/20 (Navidad)
python evaluate.py --dataset data/test_tiempos.csv         --t5-dir models/t5_correction --development --out reports/tiempos.json          # 60/60
python evaluate.py --dataset data/test_sobrecorreccion.csv --t5-dir models/t5_correction --development --out reports/sobrecorreccion.json  # 17/18 (el te/él te)
python evaluate.py --dataset data/test_ninos.csv          --t5-dir models/t5_correction --development --out reports/ninos.json            # 71/73 (el jirafa; que si/sí). Antes de las reglas del 29-sep: 53/73
# Dos sets EXTERNOS con oraciones reales (COWS-L2H dev, universitarios de español L2, corrección de
# un profesor; scripts/sample_cowsl2h.py, semilla 2026). No son la población objetivo: miden daño
# (precisión) y sirven para comparar versiones. Placeholders *FIRST_NAME* -> *FIRSTNAME* cuentan
# como FP y son artefacto del corpus (4-8 por muestra). Valores v4 + reglas del 2026-09-29 (entre
# paréntesis, los del 19-sep): el recall cae 1-2 aciertos porque los nombres propios en mitad de
# frase ya no se corrigen (Pizzaro, Ireland), a cambio de no estropear Stefani, Amtrak, Uber...
python evaluate.py --dataset data/test_cowsl2h_spelling200.csv --t5-dir models/t5_correction --development --out reports/cowsl2h_spelling200.json  # P 0,762 R 0,460 F0.5 0,674 (76/200; 19-sep: 0,738/0,461/0,659)
python evaluate.py --dataset data/test_cowsl2h_random200.csv   --t5-dir models/t5_correction --development --out reports/cowsl2h_random200.json    # P 0,604 R 0,150 F0.5 0,376 (27/200; 19-sep: 0,580/0,152/0,371)
# Una corrida con la misma semilla NO es determinista en GPU: v4c (mismos datos que v4b salvo
# 3 pares) perdió `mis amigos no vino` en test_agreement. Guardar el adaptador de cada intento
# (models/grammar_lora_v4a..c) y elegir por las regresiones, no por val_loss.

# Manifiestos de artefactos ya entrenados (solo hashes; no entrena ni fusiona)
python scripts/model_manifest.py --write-current
python scripts/model_manifest.py --show
```

## Artefactos reproducibles y `modelVersion` compuesto

El LoRA gramatical global se versiona por contenido, no por nombre. Dos
manifiestos JSON acompañan a los pesos (en `models/`, que sigue fuera de git):

| Manifiesto | Lo escribe | Contenido |
|------------|------------|-----------|
| `models/grammar_lora/training-manifest.json` | `train_grammar_lora.py`, justo después de `save_pretrained` | modelo base (`vgaraujov/t5-base-spanish` y su revisión en la caché HF), `dataset_sha256` y filas del CSV, **índices exactos** de la partición train/validación (`random.seed(42)`, 300 de validación), hiperparámetros (épocas 3, batch 8 × acumulación 2, lr 3e-4, rank 16, alpha 32, dropout 0.05, módulos `q`/`v`, longitud 96, prefijo `corrige: `), banderas `deterministic` (semillas de `random`/`torch`/CUDA fijadas **antes** de cargar el base y de `get_peft_model`, `torch.Generator` sembrado para el DataLoader, cuDNN determinista), versiones de Python/torch/transformers/peft, hora UTC, `adapter_sha256` (SHA-256 de `adapter_model.safetensors`) y un `modelVersion` parcial `global-lora-v1@<hash8>` |
| `models/t5_correction/model-manifest.json` | `scripts/merge_grammar_lora.py`, en el directorio temporal antes del intercambio con backup | copia del anterior + `adapter_sha256` (comprobado contra el adaptador fusionado), `model_sha256` (SHA-256 del `model.safetensors` fusionado: los pesos T5 que se sirven), hashes de todos los archivos fusionados, `merged_at_utc`, `merge_base_revision`, `base_tag` y el **`modelVersion` compuesto** |

`modelVersion = "<base-tag>@<hash8>+<lora-tag>@<hash8>"`: el primer hash son
los 8 primeros hex de `model_sha256` (etiqueta `beto-t5-base`) y el segundo
los de `adapter_sha256` (etiqueta `global-lora-v1`); las etiquetas se pueden
cambiar con `--base-tag`/`--lora-tag` (`BASE_TAG` en el merge). La composición
(`infrastructure.versioning.compose_model_version`) es la que usan `main.py`,
`evaluate.py` y el caso de uso; si cambian los pesos servidos o el adaptador,
cambia la cadena. `merge_grammar_lora.py` carga el base con el nombre y la
revisión que registró el entrenamiento (`from_pretrained(..., revision=)`;
un `BASE` por entorno distinto termina con `[ERROR]`, y un manifiesto
`provenance: trained` sin revisión se rechaza), valida el temporal (archivos
requeridos, vocabulario del tokenizer `spiece.model`/`tokenizer.json`,
manifiesto legible, `model_sha256` igual al archivo) y, ante cualquier error,
termina con `[ERROR]` sin reemplazar `models/t5_correction`.

Los artefactos actuales se entrenaron antes de que existieran los manifiestos:
`python scripts/model_manifest.py --write-current` los reconstruye
(`"provenance": "reconstructed"`) hasheando los archivos existentes y leyendo
los hiperparámetros de `adapter_config.json` y de `TRAINING_DEFAULTS`
(`scripts/model_manifest.py`, la única fuente de verdad que también usa
`train_grammar_lora.py`); si `data/training_pairs_clean.csv` sigue en el repo,
registra su hash y la partición. Nunca reentrena, fusiona ni descarga; se niega
a componer una versión si el `training-manifest.json` existente describe otro
adaptador (`--force` lo reconstruye). Versión actual:
`beto-t5-base@9f643870+global-lora-v4@cda68b25` (2026-09-19; anteriores en `models/*_v3`, `*_v2`, `*_v1_sep12`).

### Evaluación final sobre el holdout bloqueado (pendiente del autor)

El archivo aprobado por el asesor debe colocarse **fuera del repositorio**, en
`C:\Users\Dovamul\Desktop\TESIS\documentos\datos-reservados\holdout-final.csv`
(formato `categoria|entrada|esperado_1|...`), y no se usa para nada más; ni
`pruebas.txt` ni `data/eval_gold.csv` lo sustituyen. Cuando exista:

```powershell
# 1) Descarga del T5 base limpio (NECESITA RED: sin HF_HUB_OFFLINE), con la
#    revisión explícita que registró el entrenamiento del LoRA
#    (models/grammar_lora/training-manifest.json → base_model_revision).
$env:HF_HOME = "models\hf-cache"
$env:OUT = "models/t5_base"
$env:MODEL_REVISION = "<base_model_revision del training-manifest.json>"
python setup_model.py                       # T5 base limpio a models/t5_base (única descarga)
Remove-Item Env:OUT; Remove-Item Env:MODEL_REVISION

# 2) Versión hasheada del baseline (models/t5_base no tiene model-manifest.json,
#    así que evaluate.py exige --model-version; esta es la cadena que hay que pasar):
python scripts/model_manifest.py --base-version models/t5_base       # → t5-base@<hash8 de model.safetensors>

# 3) Las dos evaluaciones, ya sin red.
$env:HF_HUB_OFFLINE = "1"
python evaluate.py --dataset C:\Users\Dovamul\Desktop\TESIS\documentos\datos-reservados\holdout-final.csv --t5-dir models/t5_base --model-version "t5-base@<hash8>" --out reports/t5-base-holdout.json
python evaluate.py --dataset C:\Users\Dovamul\Desktop\TESIS\documentos\datos-reservados\holdout-final.csv --t5-dir models/t5_correction --out reports/global-lora-v1-holdout.json
```

(En la segunda, `evaluate.py` compone `beto-t5-base@04c598db+global-lora-v1@19ec2e4c`
desde `models/t5_correction/model-manifest.json`; el baseline se documenta con
su hash real, no solo como `t5-base`.) Antes de comparar F0.5, `datasetSha256`
debe ser idéntico en ambos informes y el bloque `pipeline.commit` debe ser el
mismo commit limpio (sin sufijo `-dirty` ni `-unknown`); ninguno de los dos
se ha generado todavía.

## Pipeline de corrección (global, idéntico para todos los alumnos)

| Capa | Descripción |
|------|-------------|
| 1    | Palabras ya acentuadas: se respetan (solo restauración de tilde/ñ) |
| 1.1  | Correcciones manuales de alta prioridad (abreviaturas de chat, homófonos frecuentes) |
| 1.2  | Escudo de palabras cortas (evita alucinaciones en "y", "a", "mi") |
| 1.25 | Formas verbales regulares fuera de `es_50k` (`nadaremos`, `dibujaremos`): si la terminación no sufre cambio de raíz y el infinitivo está en el léxico, se respetan (`candidates.is_regular_verb_form`) |
| 1.3  | Fonética dirigida a la palabra de mayor frecuencia |
| 2    | SymSpell (todos los candidatos a Damerau-Levenshtein ≤ 2) elegidos por **coste disléxico** (`candidates.pick_candidate`): tilde 0,1; vocal↔vocal, consonantes confundibles (b/v, c/s/z, g/j, m/n, r/l, d/t), inserción/omisión de vocal, duplicación (rr/ll), transposición y cambio de raíz o/u→ue, e→ie 0,5; `h` 0,3; resto 1; +0,5 si cambia la primera letra (salvo confundible); empate → frecuencia; piso 150. Antes: distancia entera y luego frecuencia (`jugan → jugar`, `aruz → cruz`, `camera → cadera`). El diccionario se carga con `encoding="utf-8"` (sin él, en Windows las palabras con tilde entraban como `dormirÃ¡n`). Después, restauración de ñ y tildes |
| 2.5  | Desambiguación contextual de homófonos y tildes con BETO MLM |
| 3    | Gramática por reglas (haber impersonal, gustar, concordancia de número) |
| 4    | Refinamiento gramatical con T5 + LoRA global (beam search, 3 candidatos con score); el beam 1 se recomienda solo si pasa la guarda de cantidad (`is_safe_refinement`) y la léxica (`is_lexically_plausible_refinement`: reemplazos cercanos, inserciones/borrados solo de la lista blanca, nunca negadores ni cuantificadores) |
| 4.5  | Ortografía vigente (RAE 2010): `sólo`, `guión` y los demostrativos con tilde se normalizan salvo que el alumno haya escrito la tilde (`normalize_modern_spelling`); el corpus de subtítulos del T5 y `es_50k` son anteriores a la reforma |
| 5    | Selección de alternativas: una recomendación, o hasta 3 opciones solo con señal positiva de ambigüedad (modo, tilde o empate de BETO; ver abajo) |

## Evaluación: Precisión, Recall y F0.5 a nivel de edición (`evaluate.py`)

`evaluate.py` evalúa **la sugerencia recomendada** (`correct(text, {})[0]`; las
alternativas de la capa 5 no se puntúan) contra un dataset
`categoria|entrada|esperado_1|esperado_2...` (una o más referencias; menos de
tres campos es un error) y guarda un informe JSON versionado. La métrica
principal es F0.5 sobre ediciones con el scorer `exact_token_edits_v1`; WER,
CER, mejora, exactitud y latencia se conservan como diagnósticos.

Scorer `exact_token_edits_v1`:

- tokens: `re.findall(r"\w+|[^\w\s]", texto)` (palabras y signos sueltos);
- alineación entrada → texto por programación dinámica sobre tokens
  (`align_tokens`), con estos costes: token idéntico 0; sustitución 1:1 0.1 si
  solo cambian tildes/mayúsculas (`esta` → `está`, `manana` → `mañana`), la
  distancia de Levenshtein de caracteres normalizada si es ≤ 0.5 (`iva` →
  `iba`) y 1 en otro caso (`iba` → `fue`); división 1:m o unión k:1 (m, k ≤ 4)
  solo cuando las letras coinciden sin tildes ni mayúsculas (`ala` → `a la`,
  `de los` → `delos`), 0.1; inserción y borrado 1. En empate gana siempre la
  misma operación, así que la alineación es determinista;
- ediciones: cada sustitución, división o unión es una edición `(inicio, fin,
  tokens_reemplazo)` sobre la entrada; una racha de inserciones/borrados
  consecutivos es una sola edición. Así una corrección vecina de otra nunca
  cambia cómo se cuentan las demás: `el arbol esta lleno` → `el árbol está
  muy lleno` son 3 ediciones (2 TP + 1 FP frente a `el árbol está lleno`),
  `iva ala` → `iba a la` son 2 (`iba ala` puntúa 1 TP + 1 FN), `ala` → `a la`
  es 1, y `hola` → `Hola.` son 2 (mayúscula y punto por separado). Una
  corrección acertada nunca deja de contar como TP por una edición vecina;
- por frase se elige la referencia con mejor F0.5 (en empate, más TP y menos
  FP+FN); TP = predichas ∩ gold, FP = predichas − gold, FN = gold − predichas,
  sumadas sobre el corpus y por categoría;
- P = TP/(TP+FP), R = TP/(TP+FN), F0.5 = 1.25·P·R/(0.25·P+R), **0.0 cuando el
  denominador es 0** (la convención que valida el backend con tolerancia 1e-6;
  por eso la categoría `correcto`, sin ediciones gold, muestra P = R = 0 y solo
  aporta falsos positivos).

Informe JSON (`--out`): **el archivo se sube tal cual al panel Vue** ("Evaluación
técnica"), que lo valida y lo registra en
`POST /api/v1/research/technical-evaluations`. Las claves del contrato van en
el nivel superior del JSON (`technicalEvaluation` las repite agrupadas, solo
para leerlas de un vistazo):

```json
{"modelVersion": "...", "datasetSha256": "<64 hex>", "scorerVersion": "exact_token_edits_v1",
 "precision": 0.98, "recall": 1.0, "fZeroFive": 0.985,
 "truePositives": 54, "falsePositives": 1, "falseNegatives": 0,
 "categories": [{"category": "tilde", "tp": 12, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f05": 1.0}, ...]}
```

`modelVersion` debe ser texto no vacío de hasta 160 caracteres y **describe lo
evaluado** (`evaluation_model_version`): `--model-version` > `MODEL_VERSION`
> con `--t5-dir` y BETO, compuesto desde `<t5-dir>/model-manifest.json` (el
manifiesto del directorio evaluado, nunca el de `models/t5_correction` por
defecto) > en `--source http`, el `modelVersion` que devuelve el servidor
(código 2 si no lo devuelve o cambia entre respuestas). Sin `--t5-dir` o con
`--no-beto`, `--model-version` es obligatoria (código 2 si falta): un informe
de "solo reglas" o del T5 base jamás hereda la cadena del LoRA.
`datasetSha256` es el SHA-256 de los bytes exactos del archivo (los CSV
hasheados llevan `-text` en `.gitattributes`, así que el checkout no cambia
sus finales de línea; `data/eval_gold.csv` = `a3e7f3a4…0af6`); las categorías
salen de la columna `categoria` (nombres recortados a 80 caracteres, únicos;
más de 50 es un error del dataset). Además, como diagnóstico: `development`
(true para `data/eval_gold.csv`, `pruebas.txt` o `--development`; esos
informes no se registran en el backend), `timestamp`, `source`, `modelDir`
(`--t5-dir`), `dataset`, `pipeline` (`thresholds` = `alternatives.THRESHOLDS`
del código que corrió y `commit` de git, con sufijo `-dirty` si el árbol
tenía cambios en archivos versionados y `-unknown` si `git status` falló:
`modelVersion` identifica pesos, no código; en `--source http` el código que
corrigió es el del servidor, del que solo llega `modelVersion`, así que
`pipeline` queda en `null` y los umbrales y el commit del checkout que
**evalúa** se escriben como `clientPipeline` con una `note` que lo aclara),
`global`,
`por_categoria`, `latency_ms_avg`, `casos` (cada caso con `tp/fp/fn`, la
referencia elegida y, si el modelo lanzó una excepción, `error`), `errors`
(conteo) y `errorList`. Un caso con error se puntúa como "sin cambios" (la
predicción es la entrada) y un informe que no es de desarrollo termina con
código de salida 2 si `errors > 0`. El panel rechaza archivos de más de 1 MB:
con `casos` incluido el informe pesa unos 500 B por caso, así que el tope
práctico son ~2 000 casos.

Fuentes: `--no-beto` (solo reglas), por defecto reglas + BETO, `--t5-dir
models/t5_correction` añade el T5 global fusionado, `--source http --url ...`
mide un servidor. `--t5-dir` exige `config.json` y `tokenizer_config.json`
locales y se comprueba antes de importar torch: por esa vía nunca se descarga
nada. BETO se lee de la caché de Hugging Face (`HF_HOME`); `evaluate.py` fija
`HF_HUB_OFFLINE=1` si no estaba definido, de modo que una caché incompleta es
un error y no una descarga. El holdout final se identifica por su SHA-256 y no
se usa para nada más.

Informes de desarrollo sobre `data/eval_gold.csv` (38 casos, `datasetSha256`
`a3e7f3a4…0af6`, GPU; regenerados en la ola final con la guarda léxica de
inserciones/borrados):

| fuente               | modelVersion                                   | TP | FP | FN | P     | R     | F0.5  | exactos | ms/frase |
|----------------------|------------------------------------------------|----|----|----|-------|-------|-------|---------|----------|
| reglas               | `rules-dev` (explícita)                        | 45 | 0  | 9  | 1.000 | 0.833 | 0.962 | 29/38   | 0        |
| reglas + BETO        | `rules-beto-dev` (explícita)                   | 53 | 0  | 1  | 1.000 | 0.981 | 0.996 | 37/38   | 39       |
| reglas + BETO + T5   | `beto-t5-base@04c598db+global-lora-v1@19ec2e4c` | 54 | 1  | 0  | 0.982 | 1.000 | 0.985 | 37/38   | 186      |

El único FP de la fila +T5 es `no se donde deje mi cuaderno → no sé dónde dejé
mi cuaderno` (el gold lleva `donde` sin tilde).

## Alternativas: hasta tres opciones solo cuando la oración es ambigua

`suggestions[0]` es siempre la **recomendada**: el beam 1 de T5 si es seguro
y, en cualquier otro caso (sin T5, T5 falla, beam 1 inseguro o léxicamente
implausible), el texto base de reglas+BETO. Nunca se salta al beam 2 para
recomendar: si el mejor beam alucina, T5 no cuenta para esa frase (regla
anterior a esta capa). "Seguro" son dos guardas sobre el beam 1
(`infrastructure/ml/guards.py`): `is_safe_refinement` (cantidad: longitud
0.6–1.5x, ≤ max(3, n/3) palabras cambiadas) y, desde el cierre de la Task 5,
`is_lexically_plausible_refinement` (calidad: cada reemplazo 1:1 de palabra
respecto a la base tiene similitud sin tildes ≥ `recommendedMinSimilarity =
0.3`; bloquea sustituciones léxicas como `pasto→maíz` 0.22 o `voy→iré` 0.0 y
conserva flexiones, incluso irregulares, como `es→son` 0.40, `hizo→hicieron`
0.50, `viene→vengan` 0.55). Desde la ola final las **inserciones, borrados y
bloques 1:n / n:1** también se juzgan, cerrados salvo lista blanca:
puntuación suelta; artículos, preposiciones, conjunciones, clíticos y
`al`/`del` (`llego tarde a clase → llegué tarde a la clase`); borrado de un
duplicado adyacente (`muy muy bien → muy bien`); unión/división con las
mismas letras (`ala → a la`, `por que → porque`). Insertar o borrar un
negador o cuantificador (`no nunca jamás nadie nada ni ningún ninguna
ninguno tampoco todos todas siempre`) se bloquea siempre (`no quiero ir →
quiero ir` recomienda la base), igual que insertar o borrar palabras de
contenido (`fui parque → fui al gran parque`, `voy a ir → iré`). Desde el
cierre de la rama la protección cubre también los **reemplazos 1:1**: si
alguno de los dos lados es una palabra protegida, la similitud no basta y
ambos deben ser la misma palabra o su flexión de número/género (`todos →
todas`, `ningún → ninguna` pasan; `lo quiero ir → no quiero ir`, `no quiero
ir → yo quiero ir`, `ni → mi`, `nunca → jamás` recomiendan la base). La capa 5 nunca
cambia la recomendada (`correctedText`, `evaluate.py` y TAS dependen de
ella) y una segunda lectura de BETO jamás la ocupa (van aparte, como
`variants=`, y ni siquiera sin `recommended` explícito pueden deducirse como
posición 0). En el caso común hay una sola opción; solo cuando el texto
admite lecturas distintas —con una **señal positiva de ambigüedad**— se
ofrecen hasta **tres**. La selección es pura
(`infrastructure/nlp/alternatives.py`, sin torch) y recibe los beams de T5
con su `sequences_scores` (el beam 1 y los beams 2/3 que pasen
`is_safe_refinement`), las "segundas lecturas" de BETO, el texto base y el
léxico de frecuencias del motor fonético (para decidir conjugaciones).

Una opción **extra** (la base cuando difiere de la recomendada, otro beam o
una segunda lectura de BETO) se ofrece únicamente si:

| Regla | Criterio |
|-------|----------|
| (a) segura     | pasa `is_safe_refinement` respecto al texto base (longitud 0.6–1.5x, ≤ max(3, n/3) palabras cambiadas) |
| (b) distinta   | tras normalizar espacios y puntuación final es distinta del texto original y de las ya elegidas (el original nunca es una corrección) |
| (c) otra lectura | su **resultado de edición** respecto al original —qué tramos cambian y por qué palabras, con tildes, sin la mayúscula inicial— es distinto del de las ya elegidas: `juegan` y `jueguen` son dos lecturas; `Está bien.` y `Está bien!` son la misma. Solo decide si dos candidatos son distintos; las paráfrasis las frenan (a)/(d)/(e)/(f) |
| (d) empate     | su score está a menos de `scoreMargin = 0.3` del mejor candidato (beams reales: `juega→juegan` 0.09, `juega→jueguen` 0.14, `fue→fueron` 0.25 entran; `botar→tirar` 0.35, `voy→iré` 0.37, `luego→después` 0.40, `ayer→anoche` 0.48, `ellos→él` 0.91 quedan fuera). La base recibe el mejor score y nunca cae por (d) |
| (e) léxica     | cada tramo que edita **respecto a la base** (lo que T5 tocó realmente, no el original que las reglas ya corrigieron) es una variante cercana (similitud sin tildes ≥ `minSpanSimilarity = 0.6`): `esta/está`, `tubo/tuvo`, `fue/fueron` sí; `luego/después`, `voy/iré` (paráfrasis) no |
| (f) señal positiva | estar cerca del mejor beam no prueba ambigüedad. Un beam de T5 (o la base) solo es alternativa si **(i)** difiere de la **recomendada** en **una sola palabra** (un reemplazo 1:1; cualquier edición extra —`jueguen muchos` frente a `juegan mucho`, o dos tildes a la vez— lo descalifica), esa palabra es una flexión del mismo lexema (prefijo común sin tildes ≥ `inflectionMinPrefix = 0.6` del más corto y similitud ≥ 0.6) y el contraste es **de modo** (indicativo ↔ subjuntivo: `juegan/jueguen`, `llegan/lleguen`, `gana/gane`, `comen/coman`, con los cambios ortográficos g/gu, c/qu, z/c, g/j) o **solo de tilde** (`callo/calló`, `papa/papá`, `como/cómo`), y la alternativa no vuelve a la forma del original. Un contraste de modo exige además que la raíz sea un **verbo del léxico `es_50k`** con una sola conjugación decidible (`jug-ar`, `lleg-ar`, `com-er`) y que ninguna de las dos formas sea más frecuente que su infinitivo (`casa` 496977 ≫ `casar` 10364 y `esta` ≫ `estar` son lecturas nominal/demostrativa: `casa/case` y `esta/este` no son pares de modo; `llega` < `llegar`, `juegan` < `jugar` sí); corregir número o persona (`juega/juegan`, `fue/fueron`, `lleguen/llegue`) lo fija el sujeto y nunca es "otra lectura", ni lo es una inserción o un borrado; o **(ii)** es una segunda lectura de BETO (empate en un homófono), que trae su propia señal |
| (g) filtro de modo | si la recomendada contiene un disparador de subjuntivo obligatorio (`ojalá`, `es posible que`, `es necesario que`, `espero que`, `quiero que`, `me alegra que`, `me gusta que`, `para que`, `antes de que`, `dudo que`, `no creo que`), la variante en **indicativo** del verbo que gobierna no se ofrece (solo cabe el subjuntivo, como recomendada o como alternativa). La conjugación (-ar: subjuntivo en -e; -er/-ir: en -a) se decide buscando el infinitivo en el léxico `es_50k`; si no decide (p. ej. `est-`: `estar` y `Ester`), el par no se ofrece (fail-closed). No cambia la recomendada (beam 1) |

La base va justo detrás de la recomendada (el tope no la elimina) y el resto
por score descendente. Si ninguna extra cumple, se devuelve solo la
recomendada. Consecuencia de (f): el par histórico "refinado + base" ya no
aparece (`sé que no vendrá hoy` + `se que no vendrá hoy`): la base solo
volvería a la forma del original en el tramo que T5 corrigió.

Segunda fuente de ambigüedad, en la capa 2.5 (`_disambiguate_context`): cuando
BETO puntúa las dos mejores lecturas de un homófono a menos de
`betoTieMargin = 0.3` (diferencia de PLL media, unidades de BETO), el texto no
cambia pero se genera la variante con la segunda lectura, con
score = −|diferencia| (solo para ordenarla junto a los beams; nunca es la
recomendada). Este umbral es independiente de los márgenes de sobrescritura
(0.5 / 3.0 / 5.0): `tubo/tuvo` con diferencia 1.0 o `se/sé` con 1.4 ni se
sobrescriben ni cuentan como empate; `esta/está` con 0.13 sí. Máximo una
alternativa por posición y 3 posiciones (las más empatadas). Una lectura más
de `maxFreqRatio = 20` veces más rara que la escrita no cuenta como
ambigüedad (`hoy/holly`, `mis/miss`: la PLL media de BETO infla las palabras
de varios subtokens). Una tilde **escrita por el alumno** se respeta (como en
la capa 1): BETO no sobrescribe `quizás → quizas` aunque prefiera la forma
sin tilde, ni la ofrece como segunda lectura; una tilde que puso la capa 2
(`restore_accent`: `papa → papá`) sí sigue a criterio de BETO (`el papa` /
`el Papa`).

Los seis umbrales viven en `alternatives.THRESHOLDS`
(`{"scoreMargin": 0.3, "minSpanSimilarity": 0.6, "maxFreqRatio": 20,
"betoTieMargin": 0.3, "recommendedMinSimilarity": 0.3, "inflectionMinPrefix":
0.6}`), como constantes de código sin override por entorno: el mismo
`modelVersion` se comporta igual en el servicio, en `evaluate.py` y en una
ejecución manual, y `evaluate.py` los registra en el bloque `pipeline` del
informe junto con el commit. Se calibran con `data/ambiguity_calibration.csv`
(39 frases etiquetadas lingüísticamente, no según lo que hace el modelo:
20 claras → 1 opción, 12 homófonos ambiguos → 2, 7 con ambigüedad modal
real → 2–3: `quizás`, `tal vez`, `aunque`, `cuando`, `mientras`, `no creo
que` y `a mí me gusta que`; tras `espero que`, `es posible que`, `ojalá que`
y `me alegra que` solo cabe el subjuntivo, así que esas frases se esperan con
1 opción; la columna `forbidden_readings` lista, por frase, las lecturas
inválidas conocidas —`quizas`, `llegan` tras `es posible que`, `juega` con
`ellos`, `se` sin tilde…— que ninguna alternativa puede contener) y
`scripts/calibrate_alternatives.py`, que corre el pipeline real y resume
precisión/recall de "ofrece ≥ 2 opciones" frente a la etiqueta. **`--check`
es la prueba de aceptación** (código de salida 1 si falla): más de
`MAX_CLEAR_FP = 2` de las 20 frases claras con ≥ 2 opciones, o cualquier
alternativa (posiciones 1..n; la recomendada sigue la regla del beam 1) con
una lectura prohibida. El recall se informa, no se exige. Como el gate es
ciego a la recomendada, el resumen añade la línea informativa (no gatea)
`recomendada contiene lectura prohibida=N` con la lista de frases, para que
"0 lecturas prohibidas · PASA" no se lea como una validación del beam 1 (ver
abajo: tras `espero que` y `me alegra que` este modelo recomienda el
indicativo). `--verbose --out`
guarda la base, TODOS los beams crudos con el veredicto de las dos guardas
del beam recomendado, los beams que llegan a la capa 5 y las variantes de
BETO, para barrer umbrales y guardas offline con el selector puro:

```powershell
$env:HF_HOME = "models\hf-cache"
.venv\Scripts\python.exe scripts\calibrate_alternatives.py --check --verbose
.venv\Scripts\python.exe -m unittest -v test_alternatives test_correction_layers   # reglas, capas y --check con fakes, sin torch
.venv\Scripts\python.exe test_lora_guards.py                                        # guardas del beam recomendado
```

Última calibración (modelo fusionado, umbrales de arriba, cierre de la
rama; idéntica frase a frase a la de la ola final): 39 frases, 19 ambiguas
esperadas, 5 ofrecidas: **TP 4 · FP 1 · FN 15 · TN 19 → precisión 0.80,
recall 0.21, F1 0.33**, máximo 2 opciones, 0 lecturas prohibidas ofrecidas;
`--check` **PASA** (FP 1/20 ≤ 2). Línea informativa: `recomendada contiene
lectura prohibida=8` (no gatea): `espero que ellos viene mañana → vienen` y
`me alegra que ustedes esta aqui → están` (el beam 1 recomienda el indicativo
tras el disparador y `vengan`/`estén` no son alternativa: raíz irregular
`veng-` ≠ `vien-`; `est-` no decide entre `estar` y `Ester`), más seis frases
en las que el beam 1 dejó la forma del original sin corregir (`los niño
juega`, `ellos fue`, `hola como`, `quizás… llega`, `tal vez… gana`). Aciertos:
`papa/papá llegó`, `termino/terminó` (empate BETO), `callo/calló` y
`juegan/jueguen` (`a mí me gusta que…`). El único FP es `hola como estás` +
`hola cómo estás` (contraste de tilde sobre una palabra que el beam 2
corrige; la etiqueta solo admite la lectura interrogativa, que es
precisamente la alternativa). Los 15 FN son lecturas que ni T5 (no
generadas, o fuera del margen: `lleguen`, `coman`, `taza`, `votar`, `ganamos`
0.45) ni BETO (sin empate) producen, más las que la señal (f) ya no admite
por diseño: `quizás… llega/llegan` (número, no modo) y `llego tarde a clase`
(la segunda opción solo difería en `la`). `no creo que… vienen` es un FN
conocido con otra causa: la recomendada **es** `vienen` (beam 1, −0.009) y
`vengan` (−0.625) queda fuera por el margen (d) (0.616 > 0.3), no por el
filtro de modo (g), que solo retira indicativos candidatos y nunca toca la
recomendada; la etiqueta (ambigua) y el disparador `no creo que` se
mantienen. Antes de la ola final (sin la
señal positiva ni el filtro de modo) la misma calibración daba TP 6 · FP 8 ·
FN 13 · TN 12 (precisión 0.43, recall 0.32) con lecturas inválidas ofrecidas
(`llegan`/`llegue` tras `es posible que`, `gana` tras `ojalá`, `está/esta`
tras `me alegra que`, `quizas`). Ningún umbral numérico cambió.

**Limitación conocida.** La capa de alternativas es una heurística
**exploratoria**: con la señal positiva de ambigüedad casi no ofrece ruido en
frases claras (1/20), pero solo recupera el 21 % de las frases etiquetadas
como ambiguas (4/19) porque T5 rara vez genera la segunda lectura dentro del
margen y BETO rara vez empata. La afirmación "hasta tres opciones solo cuando
la oración es ambigua" debe leerse como "nunca más de tres, y una segunda
opción solo con señal de modo, de tilde o de empate de BETO"; hasta contar
con un conjunto etiquetado mayor que el de 39 frases (y con más de una
anotación por frase), el recall medido no sostiene la funcionalidad como
característica principal y en la tesis se presenta como exploratoria.

Salida real del modelo fusionado tras la ola final (`test_alternatives.py` y
`test_correction_layers.py` cubren las reglas y las capas con fakes):

```
mañana voy al parque con mis amigos  → ["mañana voy al parque con mis amigos"]            (1: ya estaba bien)
esta bien, nos vemos luego           → ["está bien, nos vemos luego"]                     (1: T5 y BETO coinciden)
a mi me gusta que ellos juega mucho  → ["a mí me gusta que ellos juegan mucho",
                                        "a mí me gusta que ellos jueguen mucho"]          (2: indicativo (beam 1), subjuntivo; la base "juega" ya no)
es posible que ellos llega tarde     → ["es posible que ellos lleguen tarde"]             (1: llegan/llegue filtrados)
se que no vendra hoy                 → ["sé que no vendrá hoy"]                           (1: la base "se" vuelve al original)
hola como estas                      → ["hola como estás", "hola cómo estás"]             (2: afirmación / pregunta; el FP admitido)
xq ellos juega mucho                 → ["porque ellos juegan mucho"]                      (1: la base "juega" es concordancia, no lectura)
quizás ellos llega tarde             → ["quizás ellos llega tarde"]                       (1: la tilde escrita se respeta; "llegan" es número)
```

El contrato HTTP no cambia (`suggestions: list[str]`); el teclado muestra como
mucho tres globos y nunca uno idéntico al texto original.

## Análisis offline de pruebas de oraciones (`scripts/analyze_sentence_tests.py`)

Reproduce los resultados de una prueba de oraciones
(`GET /api/v1/research/tests/{id}/results`, `TestResultsResponse`) a partir
del CSV de exportación del backend (`GET …/tests/{id}/export.csv`, 21
columnas, una fila por respuesta de cada intento `COMPLETED`, los excluidos
con `excluded=true`) con **las mismas fórmulas y el mismo bootstrap
determinista que `TestResultsService`**, de modo que el análisis
independiente valide número a número el JSON del backend
(`backend/docs/research-api.md`, §6):

- Por alumno y condición (`ASSISTED` / `UNASSISTED`):
  `errorsPer100Words = 100 · Σerrores / Σpalabras` (solo respuestas con
  `word_count > 0` y `error_count` conocido) y
  `wordsPerMinute = Σpalabras / (Σduration_first_key_ms / 60000)` (solo con
  `duration_first_key_ms > 0`). Sumas enteras, sin reordenar.
- `MetricInterval`: media aritmética simple de los valores por alumno
  (ordenados por `student_username`) con IC 95 % bootstrap percentil
  **determinista**: PRNG SplitMix64 (misma aritmética de 64 bits que la clase
  Java), semilla 42, 2000 remuestreos, `nextIndex(n) = nextLong() mod n` con
  resto sin signo, cuantil R-7 sobre las medias ordenadas en 0.025 y 0.975.
  Con `n < 2` los límites son `null`. Los valores dorados de la Task 6
  (`SplitMix64(42)` → `-4767286540954276203, 2949826092126892291,
  5139283748462763858`; `bootstrap([1,2,3,4,5])` → `[1.8, 4.2]`) están
  afirmados en los tests a 1e-9.
- `paired` (con − sin, alumnos con ambas condiciones): el mismo bootstrap
  sobre las diferencias (`bootstrapLower/Upper`) más el resumen t pareado
  clásico (`tLower/tUpper`, `t`, `p` bilateral, `dz`; `null` con `n < 2` o
  varianza nula). La t de Student se implementa con la beta incompleta
  regularizada (`math.lgamma` + fracción continua), sin `scipy`.
- `acceptanceRate` (solo con ayuda): `Σaccepted / Σoffered × 100` con
  intervalo de Wilson (`Z_95 = 1.959964`, acotado a `[0, 100]`).
- `sentences[]` por posición: `n` (respuestas terminadas, omitidas incluidas),
  `meanErrors` (solo conteos conocidos), `meanDurationFirstKeyMs` (solo con
  duración) y `skippedCount`.
- Reglas de muestra: una respuesta `skipped=true` no aporta a ninguna métrica
  (ni a `participants` ni a los contadores de sugerencias); los intentos
  `excluded=true` solo cuentan en `sample.completed` / `sample.excluded`;
  `unannotatedFree` cuenta libres sin anotar (`error_source=PENDING`) no
  omitidas de intentos analizados e `incomplete = unannotatedFree > 0`.
- Validación determinista: cabecera **exactamente** igual a las 21 columnas y
  en ese orden, tipos (`position ≥ 1`, enteros ≥ 0, booleanos `true/false`),
  `kind`, `assistance` y `error_source ∈ {AUTO, ANNOTATED, PENDING}`
  conocidos, coherencia `kind`/`error_source`/`error_count`, un solo
  `test_code`, sin duplicados `attempt_id × position`.
- Campos que el CSV no puede conocer (`testId`, `title`, `status`,
  `sample.assigned/cancelled/inProgress`, `provenance.backendVersions`) van a
  `null` y no se comparan; `minSample` es `--min-sample` (por defecto 8, el
  del backend).

```powershell
.venv\Scripts\python.exe scripts/analyze_sentence_tests.py test-PRUEBA-01-responses.csv --out reports/prueba-01.json
# Reconciliación con el backend (código 1 si algún campo difiere; flotantes con tolerancia 1e-6):
.venv\Scripts\python.exe scripts/analyze_sentence_tests.py test_fixtures/sentence_tests/responses-sample.csv `
    --compare-results test_fixtures/sentence_tests/results-sample.json
```

`--compare-results` compara `sample.{completed,excluded,unannotatedFree}`,
`incomplete`, por condición `participants`, `errorsPer100Words` y
`wordsPerMinute` (`n, mean, lower, upper`), `acceptanceRate`
(`accepted, offered, ratePct, wilsonLower, wilsonUpper`), `paired.*` (los 11
campos de `PairedDelta`), `sentences[]` y `datasetSha256` frente al SHA-256
de los bytes del CSV de entrada (el fixture lleva `-text` en
`.gitattributes` para que el checkout no cambie los finales de línea);
enteros, booleanos y cadenas exactos, flotantes con `--tolerance` (≥ 0,
default 1e-6), `null` ⇔ `None`. Códigos de salida: 0 sin diferencias, 1 con
diferencias, 2 si el CSV o el JSON no se pueden leer o validar. Fixtures:
`test_fixtures/sentence_tests/responses-sample.csv` (réplica byte a byte de la
cohorte sintética de `TestResultsServiceTests` del backend: 3 alumnos × 4
oraciones más un cuarto alumno excluido) y `results-sample.json` (generado
por este mismo script sobre la fixture, con la forma de
`TestResultsResponse`; la reconciliación contra una exportación real del
backend se hace en la Task 16). Tests:
`.venv\Scripts\python.exe -m unittest -v test_analyze_sentence_tests`.
