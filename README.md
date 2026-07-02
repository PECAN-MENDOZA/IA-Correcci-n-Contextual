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
│       ├── train_model.py          # Casos de uso: entrenamiento base y por usuario
│       └── evaluate_model.py       # Caso de uso: evaluación WER/CER/improvement
│
├── infrastructure/                 # Adaptadores técnicos — implementaciones concretas
│   ├── ml/
│   │   ├── beto_model.py           # EncoderDecoder BETO + cuantización
│   │   └── dataset_loader.py       # Descarga OPUS-100 y genera pares de entrenamiento
│   ├── nlp/
│   │   ├── phonetic_engine.py      # Motor fonético + restauración de tildes
│   │   ├── context_judge.py        # BETO MLM para desambiguar homófonos
│   │   └── correction_pipeline.py  # Pipeline de 7 capas de corrección
│   └── persistence/
│       └── csv_user_repository.py  # Repositorio CSV (implementa IUserHistoryRepository)
│
├── interfaces/                     # Entrypoints externos
│   └── api/
│       └── routes.py               # Rutas Flask (/correct, /feedback)
│
├── main.py                         # Composition Root — ensambla todo y arranca Flask
├── train.py                        # CLI: entrenamiento del modelo base
├── train_user.py                   # CLI: fine-tuning por usuario
└── requirements.txt
```

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
| `train_user.py`         | `train_user.py` (CLI) + `application/use_cases/train_model.py`      |

## Uso



```bash
python -m venv venv
# Instalar dependencias
python -m pip install -r requirements.txt


# Arrancar servidor de corrección
python main.py

# Entrenar modelo base
python train.py --epochs 3 --batch_size 2

# Fine-tuning por usuario
python train_user.py --user_id usuario_dislexia_visual --epochs 5
```

## Pipeline de corrección (7 capas)

| Capa | Descripción |
|------|-------------|
| -1   | Heurísticas manuales (palabras irrecuperables) |
| -0.5 | Dislexia visual: corrección geométrica b↔d, q↔p |
| 0    | Vocabulario personalizado del usuario |
| 0.1  | Escudo de palabras cortas (evita alucinaciones en "y", "a", "mi") |
| 0.5  | Desambiguación de homófonos con BETO MLM |
| 1    | Fonética dirigida a la palabra de mayor frecuencia |
| 2    | SymSpell (Levenshtein) + BETO como árbitro final |
