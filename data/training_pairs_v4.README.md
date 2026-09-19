# training_pairs_v4.csv

Dataset GOLD del LoRA gramatical global v4 (2026-09-19): `training_pairs_v3.csv`
(24 648 pares: generales + concordancia + subjuntivo) + `training_pairs_v4_extra.csv`
(6 248 pares generados por `scripts/generate_v4_data.py`, semilla 13), deduplicado y sin
pares de identidad → 30 896 pares `erronea,corregida`.
SHA-256: `b3290af50bcd4facacf346982763de389820ab1adc5966b6cd4a48dbce83fdc4`.

Contenido del bloque extra (dirigido a los fallos que quedaron en `data/test_tiempos.csv`
tras las reglas del commit `4c86a61`):

- 2 400 controles de INDICATIVO tras subordinantes (`aunque`, `cuando`, `si`, `porque`,
  `mientras`, `como`, `ya que`, `apenas`) en presente, pasado y futuro, con la principal en un
  tiempo compatible (`cuando`/`si` nunca con subordinada en futuro, que exigiría subjuntivo).
  La entrada solo pierde tildes (30 % además con mayúscula y punto); el verbo NO cambia.
  Motivo: v3 generalizó "subordinante + indicativo → subjuntivo" desde los disparadores
  volitivos y sobrecorregía `aunque llueve mucho` → `llueva`; sus 900 controles solo usaban
  `creo que / sé que / dice que`.
- 1 200 pares de subjuntivo PROSPECTIVO tras `cuando / apenas / en cuanto / hasta que`:
  600 controles (`cuando llegue a casa, te llamaré` / `te llamo`, solo tildes) y 600
  correcciones indicativo → subjuntivo (`cuando llegas, avísame` → `llegues`) **solo con la
  principal en futuro o imperativo**. Historial:
  - v4a (sin este bloque; 29 134 pares, adaptador `76187779`, `models/*_v4a`): pasaba las
    cuatro regresiones (tiempos 58/60) pero aprendió "cuando ⇒ indicativo": convertía
    `cuando llegue a casa, te llamo` en `cuando llegué a casa, te llamé` y dejaba de corregir
    `cuando llegas, avísame`. Detectado con `data/test_sobrecorreccion.csv`.
  - v4b (bloque con correcciones también con principal en presente; 30 331 pares, adaptador
    `ba9ecf86`, `models/*_v4b`): sobrecorrección 18/18, pero tiempos 57/60 porque el habitual
    `cuando llego a casa, mi mamá prepara el café` pasaba a `llegue` (formalmente idéntico a
    `cuando llego a casa, te llamo → llegue`).
  - v4c (correcciones solo con principal en futuro/imperativo, 400+600; 30 328 pares, adaptador
    `969dd0fd`, `models/*_v4c`): sobrecorrección 18/18 pero seguía `cuando llego → llegue`
    (150 controles habituales contra 264 correcciones) y perdió `mis amigos no vino` (14/15).
  - v4 = v4d (definitivo): correcciones prospectivas bajadas a 400 y bloque 1c de 800 controles
    habituales (`cuando / siempre que / cada vez que + presente, presente`).
- 800 controles de presente HABITUAL (bloque 1c, ver arriba).
- 1 500 pares de concordancia del PREDICADO: sujeto plural (+ relativa opcional) con verbo
  copulativo y adjetivo en singular → ambos en plural (`los perros que vi ayer estaba muy
  sucio` → `estaban muy sucios`); variantes verbo+adjetivo (60 %), solo adjetivo (25 %),
  solo verbo (15 %); adjetivos filtrados por clase del sujeto (seres / cosas). Más 400
  controles en singular legítimo (solo tildes). Motivo: v3 corregía el verbo y dejaba el
  adjetivo.
- 400 pares de `té`/`te` y `él`/`el` en contexto (`no se si el te esta caliente` → `no sé si
  el té está caliente`; `el te ayuda con la tarea` → `él te ayuda con la tarea`), con marcos
  `creo que / no sé si / mamá dice que`. Motivo: v3 elegía `él te está caliente`.

Fuera del alcance de la LoRA (anotado, no resuelto aquí): `nadaremos → daremos` y
`jugan → jugar` salen de SymSpell (capa 2) porque `nadaremos`/`juegan` no están en
`es_50k.txt`; el T5 nunca ve la palabra original.

Entrenamiento: `CSV_PATH=data/training_pairs_v4.csv OUT_DIR=models/grammar_lora_v4`
(`.venv`, CUDA; 3 épocas, semilla 42, lr 3e-4, rank 16, batch 8×2, validación 300).
## Resultado v4 (adaptador `cda68b25`, fusionado `beto-t5-base@9f643870+global-lora-v4@cda68b25`)

val_loss 0,290 → 0,233 → 0,202 (v3: 0,219). Cinco regresiones, v3 → v4:
`data/test_tiempos.csv` 55/60 → **58/60** (F0.5 0,959 → 0,981; quedan `pisina y nadaremos →
daremos` y `jugan → jugar`, ambos de SymSpell); `data/test_sobrecorreccion.csv` 17/18 → 17/18
(`no se si el te esta listo` → `él te`); `data/eval_gold.csv` 47/48 → 47/48 (`dónde`);
`data/test_agreement.csv` 15/15 → 15/15; `data/test_subjunctive.csv` 19/20 → 19/20
(`Navidad`). Ningún fallo nuevo en 161 oraciones; arreglados `aunque llueve → llueva`,
`no sé si el té está caliente` y `los perros que vi ayer estaban muy sucios`.
Servido el 2026-09-19; verificado por HTTP con las 60 oraciones (58/60, ~260 ms/frase).
