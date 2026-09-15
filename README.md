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
├── evaluate.py                     # Evaluación del pipeline sobre un dataset
├── test_global_runtime.py          # Tests del runtime global (fakes, sin torch)
├── test_alternatives.py            # Tests del selector de alternativas y de la ambigüedad BETO (sin torch)
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
python -m unittest -v test_global_runtime test_alternatives
python test_lora_guards.py

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
