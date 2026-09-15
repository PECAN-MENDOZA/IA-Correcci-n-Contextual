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
python -m unittest -v test_global_runtime
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
| 4    | Refinamiento gramatical con T5 + LoRA global, aceptado solo si pasa la guarda anti-alucinación |
