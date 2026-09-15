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
│   └── versioning.py               # modelVersion compartido: env > manifiesto del LoRA > default (solo stdlib)
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
├── test_evaluate_metrics.py        # Tests del scorer, del informe (réplica del validador del panel) y de setup_model (sin torch)
├── test_versioning.py              # Tests de infrastructure/versioning.py (sin torch)
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
variable `MODEL_VERSION`, si no de `models/grammar_lora/training-manifest.json`
(o `manifest.json`, clave `modelVersion`) y, en su defecto,
`global-lora-unversioned`. La regla vive en `infrastructure/versioning.py`
(`resolve_model_version`, solo librería estándar) y la comparten el caso de
uso y `evaluate.py`, para que las correcciones y los informes se tracen al
mismo identificador.

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
python -m unittest -v test_global_runtime test_alternatives test_evaluate_metrics test_versioning
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

`modelVersion` debe ser texto no vacío de hasta 160 caracteres
(`--model-version` > `MODEL_VERSION` > `models/grammar_lora/training-manifest.json`
o `manifest.json` > `global-lora-unversioned`; ver `infrastructure/versioning.py`);
`datasetSha256` es el SHA-256 de los bytes exactos del archivo; las categorías
salen de la columna `categoria` (nombres recortados a 80 caracteres, únicos;
más de 50 es un error del dataset). Además, como diagnóstico: `development`
(true para `data/eval_gold.csv`, `pruebas.txt` o `--development`; esos
informes no se registran en el backend), `timestamp`, `source`, `modelDir`
(`--t5-dir`), `dataset`, `global`, `por_categoria`, `latency_ms_avg`, `casos`
(cada caso con `tp/fp/fn`, la referencia elegida y, si el modelo lanzó una
excepción, `error`), `errors` (conteo) y `errorList`. Un caso con error se
puntúa como "sin cambios" (la predicción es la entrada) y un informe que no es
de desarrollo termina con código de salida 2 si `errors > 0`. El panel rechaza
archivos de más de 1 MB: con `casos` incluido el informe pesa unos 500 B por
caso, así que el tope práctico son ~2 000 casos.

Fuentes: `--no-beto` (solo reglas), por defecto reglas + BETO, `--t5-dir
models/t5_correction` añade el T5 global fusionado, `--source http --url ...`
mide un servidor. `--t5-dir` exige `config.json` y `tokenizer_config.json`
locales y se comprueba antes de importar torch: por esa vía nunca se descarga
nada. BETO se lee de la caché de Hugging Face (`HF_HOME`); `evaluate.py` fija
`HF_HUB_OFFLINE=1` si no estaba definido, de modo que una caché incompleta es
un error y no una descarga. El holdout final se identifica por su SHA-256 y no
se usa para nada más.

Informes de desarrollo sobre `data/eval_gold.csv` (38 casos, GPU):

| fuente               | TP | FP | FN | P     | R     | F0.5  | exactos | ms/frase |
|----------------------|----|----|----|-------|-------|-------|---------|----------|
| reglas               | 45 | 0  | 9  | 1.000 | 0.833 | 0.962 | 29/38   | 0        |
| reglas + BETO        | 53 | 0  | 1  | 1.000 | 0.981 | 0.996 | 37/38   | 88       |
| reglas + BETO + T5   | 54 | 2  | 0  | 0.964 | 1.000 | 0.971 | 36/38   | 411      |

## Alternativas: hasta tres opciones solo cuando la oración es ambigua

`suggestions[0]` es siempre la **recomendada** y se decide como antes de esta
capa: el primer beam seguro de T5 (`is_safe_refinement` respecto al texto
base) o, si no hay ninguno (sin T5, T5 falla, todos los beams inseguros), el
texto base de reglas+BETO. La capa 5 nunca la cambia (`correctedText`,
`evaluate.py` y TAS dependen de ella) y una segunda lectura de BETO jamás la
ocupa. En el caso común hay una sola opción (o dos: refinado + base, como
siempre); solo cuando el texto admite lecturas distintas se ofrecen hasta
**tres**. La selección es pura (`infrastructure/nlp/alternatives.py`, sin
torch) y recibe los 3 beams de T5 con su `sequences_scores`, las "segundas
lecturas" de BETO y el texto base.

Una opción **extra** (la base cuando difiere de la recomendada, otro beam o
una segunda lectura de BETO) se ofrece únicamente si:

| Regla | Criterio |
|-------|----------|
| (a) segura     | pasa `is_safe_refinement` respecto al texto base (longitud 0.6–1.5x, ≤ max(3, n/3) palabras cambiadas) |
| (b) distinta   | tras normalizar espacios y puntuación final es distinta del texto original y de las ya elegidas (el original nunca es una corrección) |
| (c) otra lectura | su **resultado de edición** respecto al original —qué tramos cambian y por qué palabras, con tildes, sin la mayúscula inicial— es distinto del de las ya elegidas: `juegan` y `jueguen` son dos lecturas; `Está bien.` y `Está bien!` son la misma. Solo decide si dos candidatos son distintos; las paráfrasis las frenan (a)/(d)/(e) |
| (d) empate     | su score está a menos de `scoreMargin = 0.3` del mejor candidato (beams reales: `juega→juegan` 0.09, `juega→jueguen` 0.14, `fue→fueron` 0.25 entran; `botar→tirar` 0.35, `voy→iré` 0.37, `luego→después` 0.40, `ayer→anoche` 0.48, `ellos→él` 0.91 quedan fuera). La base recibe el mejor score y nunca cae por (d) |
| (e) léxica     | cada tramo que edita **respecto a la base** (lo que T5 tocó realmente, no el original que las reglas ya corrigieron) es una variante cercana (similitud sin tildes ≥ `minSpanSimilarity = 0.6`): `esta/está`, `tubo/tuvo`, `fue/fueron` sí; `luego/después`, `voy/iré` (paráfrasis) no. Borrar un token igual a su vecino (`muy muy → muy`) vale 1.0 |

La base va justo detrás de la recomendada (el tope no la elimina) y el resto
por score descendente. Si ninguna extra cumple, se devuelve solo la
recomendada.

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
de varios subtokens).

Los cuatro umbrales viven en `alternatives.THRESHOLDS`
(`{"scoreMargin": 0.3, "minSpanSimilarity": 0.6, "maxFreqRatio": 20,
"betoTieMargin": 0.3}`), como constantes de código sin override por entorno:
el mismo `modelVersion` se comporta igual en el servicio, en `evaluate.py` y
en una ejecución manual, y el informe/manifiesto puede registrarlos tal cual.
Se calibran con `data/ambiguity_calibration.csv` (31 frases etiquetadas:
16 claras → 1 opción, 10 homófonos ambiguos → 2, 5 indicativo/subjuntivo →
2–3) y `scripts/calibrate_alternatives.py`, que corre el pipeline real y
resume precisión/recall de "ofrece ≥ 2 opciones" frente a la etiqueta
(`--verbose --out` guarda beams y variantes para barrer umbrales offline con
el selector puro):

```powershell
$env:HF_HOME = "models\hf-cache"
.venv\Scripts\python.exe scripts\calibrate_alternatives.py --verbose
.venv\Scripts\python.exe -m unittest -v test_alternatives test_correction_layers   # reglas y capas con fakes, sin torch
```

Última calibración (modelo fusionado, umbrales de arriba): 31 frases,
15 ambiguas esperadas, 12 ofrecidas: TP 7 · FP 5 · FN 8 · TN 11 →
precisión 0.58, recall 0.47, máximo 3 opciones. De los 5 FP, 2 son el par
"refinado + base" de siempre (`sé que no vendrá hoy`, `porque ellos juegan
mucho`) y 3 son un segundo beam a < 0.25 del mejor (`los niños juegan`,
`ellos fueron`, `hola cómo estás`; en los dos primeros la alternativa es la
corrección que el mejor beam no hizo). Los 8 FN son lecturas que ni T5 (gap
0.5–1.1) ni BETO producen dentro de los márgenes (`esta/está tarde`,
`si/sí`, `casa/caza`, `botar/votar`, `compro/compró`, `hablo/habló`) más
`viene/vengan` (gap 0.05, similitud 0.545 < 0.6). Subir `scoreMargin` a 0.4
ya admite la paráfrasis `botar→tirar`; bajar `minSpanSimilarity` a 0.5
recupera solo `vengan` en este conjunto: no se cambió ningún umbral.

Salida real del modelo fusionado (`test_alternatives.py` y
`test_correction_layers.py` cubren las reglas y las capas con fakes):

```
mañana voy al parque con mis amigos  → ["mañana voy al parque con mis amigos"]            (1: ya estaba bien)
esta bien, nos vemos luego           → ["está bien, nos vemos luego"]                     (1: T5 y BETO coinciden)
a mi me gusta que ellos juega mucho  → ["a mí me gusta que ellos juegan mucho",
                                        "a mí me gusta que ellos juega mucho",
                                        "a mí me gusta que ellos jueguen mucho"]          (3: indicativo, base, subjuntivo)
se que no vendra hoy                 → ["sé que no vendrá hoy", "se que no vendrá hoy"]   (2: refinado + base; BETO no empata se/sé, dif 1.4)
hola como estas                      → ["hola como estás", "hola cómo estás"]             (2: afirmación / pregunta)
xq ellos juega mucho                 → ["porque ellos juegan mucho", "porque ellos juega mucho"] (2: refinado + base; "xq→porque" no tira el beam)
```

El contrato HTTP no cambia (`suggestions: list[str]`); el teclado muestra como
mucho tres globos y nunca uno idéntico al texto original.
