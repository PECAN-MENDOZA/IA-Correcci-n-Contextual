# training_pairs_v2.csv

Dataset GOLD del LoRA gramatical global v2 (2026-09-16): concatenación deduplicada de
`training_pairs_clean.csv` (14 066 pares generales) y `training_pairs_agreement.csv`
(10 038 pares de concordancia sujeto-verbo generados por `scripts/generate_agreement_data.py`
en julio de 2026) → 18 104 pares `erronea,corregida`. SHA-256: `595b2e61a17a65c6ac9fa1ccbbce06d23f3833be8071b1e26f1e432a698bbcd7`.

Motivo: el adaptador entrenado solo con los pares generales no aprende concordancia
(9/12 en `test_agreement.csv`); con este dataset vuelve a 12/12 sin cambios en `eval_gold.csv`
(37/38, F0.5 0,9854). Resultado del entrenamiento: val_loss 0,526 → 0,434 → 0,388 (3 épocas,
semilla 42, lr 3e-4, rank 16); adaptador `5979a430…`, modelo fusionado
`beto-t5-base@f159b99c+global-lora-v2@5979a430`.
