# training_pairs_v3.csv

Dataset GOLD del LoRA gramatical global v3 (2026-09-16): `training_pairs_v2.csv`
(18 104 pares: generales + concordancia) + `training_pairs_subjunctive.csv` (6 544 pares
generados por `scripts/generate_subjunctive_data.py`, semilla 11), deduplicado y sin pares de
identidad → 24 648 pares `erronea,corregida`.
SHA-256: `6cb7a9323b5bf2d03c010673b6510abaf380f90b4082210f79769445a320eaa8`.

Contenido del bloque de subjuntivo:

- 5 200 plantillas «disparador + sujeto + verbo irregular» (quiero que / espero que / ojalá /
  es posible que / para que / antes de que / dudo que / no creo que …) con el verbo en
  indicativo como error y en subjuntivo como corrección (vienes→vengas, tienes→tengas,
  haces→hagas, dices→digas, sabes→sepas, vas→vayas, eres→seas, llegas→llegues …; 33 verbos,
  4 personas) más `hay→haya`; 45 % tal cual, 30 % sin tildes en la entrada, 25 % con
  mayúscula y punto.
- 12 pares minados del corpus general (es todo lo que contiene: por eso el T5 no lo aprendía).
- 900 controles de indicativo legítimo («creo que / sé que / dice que + indicativo») en los que
  la entrada solo pierde tildes; el verbo no cambia. Sin pares de identidad: un primer intento
  con un 7 % de identidad hizo que el T5 dejara `se callo` sin corregir (empate de beams).
- 500 pares de pretérito tras el clítico `se` (`se callo→se calló`, `se cayo→se cayó`,
  `se quedo→se quedó` …), donde la 1.ª persona sin tilde es imposible.

Motivo: el modelo v2 corregía 3/16 casos de `data/test_subjunctive.csv` (`quiero que tu vienes
conmigo` quedaba en indicativo). Resultado v3: val_loss 0,309 → 0,249 → 0,219 (3 épocas, semilla
42, lr 3e-4, rank 16); `test_subjunctive.csv` 15/16 (el restante es `navidad`→`Navidad`),
`test_agreement.csv` 12/12, `eval_gold.csv` 37/38 (F0.5 0,9854, igual que v2); sonda de 24 frases
23/24 iguales a v2 y el cambio restante a mejor (`jueguen`). Adaptador `2198f706…`, modelo
fusionado `beto-t5-base@fa22e0d9+global-lora-v3@2198f706`. Copias de v2 en `models/*_v2`.
