# Datos externos (Task 0: sin recolección propia)

> Revisado el 2026-09-16: no hay textos propios de la población disponibles.
> Se sustituye la recolección por recursos públicos verificados; la limitación
> de población se declara explícitamente en la tesis (ver abajo).

Este directorio documenta las fuentes públicas usadas como sustituto de datos
propios, generadas y verificadas por `scripts/fetch_external_data.py`.

## 1. COWS-L2H (`data/external/cowsl2h/`, gitignored)

- **Fuente:** [`github.com/ucdaviscl/cowsl2h`](https://github.com/ucdaviscl/cowsl2h),
  commit fijado `ebb11724f258f3ed377a27ed34f08897a8e5639c`.
- **Licencia:** Apache-2.0.
- **Qué es:** 28 CSV (`csv/*.csv`, ~30 MB) con 2 881 ensayos de estudiantes
  universitarios de español como lengua extranjera, con corrección holística de
  un profesor (columna `corrected1`, y `corrected2` como segunda referencia en
  algunos ensayos).
- **Tamaño real tras procesar** (revisión 1, tras corregir la clave de ensayo,
  la partición por autor y el separador de oraciones; ver
  `data/external/cowsl2h-manifest.json`):
  - 2 881 ensayos con `corrected1` no vacío (de 5 382 filas totales en los 28 CSV).
  - 565 ensayos con `corrected2` también presente.
  - 2 106 ensayos alinean 1:1 oración a oración entre `essay` y `corrected1`.
  - 23 355 pares de oraciones sobreviven el filtro (cambiadas, no idénticas, no
    reescrituras); 5 393 se descartan por reescritura (> 40 % de palabras
    distintas), 13 772 son pares de identidad (oración sin cambios: se cuentan
    pero no se escriben en ningún CSV, ver "Decisión: pares de identidad" más
    abajo) y 0 se descartan por contener el delimitador `|`.
  - Partición por **autor** (semilla 42, 30 % de los autores -> holdout, ver
    "Decisión: partición por autor, no por ensayo" abajo): 583 autores / 1 495
    ensayos / 16 691 filas en `pairs-dev.csv`; 250 autores / 602 ensayos /
    6 664 filas en el holdout bloqueado (28.53 % de las filas, 28.71 % de los
    ensayos — el 30 % exacto es sobre autores, no sobre filas ni ensayos, ver
    la nota del manifiesto).
- **`categoria = sin_anotar` en todas las filas.** El brief de la Task 0 pedía
  que `categoria` la asignara "el anotador de la Task 1", que todavía no
  existe (es circular con este mismo paso). Se resuelve emitiendo
  `sin_anotar` en cada fila; la Task 1 debe reescribir esta columna antes de
  usar `pairs-dev.csv` para entrenar o evaluar por categoría.
- **Decisión: pares de identidad fuera de `pairs-dev.csv`.** Las oraciones que
  el profesor dejó sin cambios (par idéntico) no se escriben ni en
  `pairs-dev.csv` ni en el holdout: entrenar con identidad perjudicó al T5 en
  una lección previa de este proyecto (ver `training_pairs_subjunctive.csv`).
  Sí se cuentan en el manifiesto (`pares_identidad_no_incluidos`) para que la
  Task 1 pueda medir qué proporción de oraciones ya estaban correctas.
- **Decisión: filas con `|`.** El formato de salida usa `|` como delimitador
  (igual que `evaluate.load_dataset`). Las oraciones que contienen `|` en el
  propio texto se descartan enteras en vez de reescribirlas, para no alterar
  el corpus (en la práctica, 0 casos en los 28 CSV del commit fijado).
- **Decisión: partición por autor, no por ensayo (corrección de un bug real,
  revisión 1).** En COWS-L2H la columna `id` identifica al **estudiante**, no
  al ensayo: el corpus es longitudinal (el mismo estudiante escribe en varios
  prompts/quarters, hasta 28 archivos). De las 5 382 filas crudas solo hay
  1 935 `id` distintos; entre las 2 881 filas con `corrected1`, solo 969 `id`
  distintos. La implementación original usaba `id` como clave de ensayo, así
  que los ensayos posteriores del mismo estudiante sobrescribían a los
  anteriores y ~60 % de los ensayos alineados se perdían en silencio (2 089
  ensayos alineados, pero solo 837 sobrevivían a la partición). La corrección:
  cada ensayo se identifica con la clave única `f"{archivo}#{fila}"` (nunca
  con `id`), y `id` se conserva como `autor_id` para particionar. El plan
  original pedía partición "por ensayo... para que no haya fugas de estilo";
  particionar por **autor** es la lectura más estricta que de verdad cumple
  esa intención, porque dejar varios ensayos del mismo autor repartidos entre
  dev y holdout sería precisamente la fuga de estilo que se busca evitar.
- **Descarga y verificación:** `scripts/fetch_external_data.py --cowsl2h`
  descarga los 28 CSV desde `raw.githubusercontent.com` en el commit fijado,
  verifica cada uno contra un SHA-256 hardcodeado en el script
  (`FILE_HASHES_SHA256`, fijado tras la primera descarga real) y es
  idempotente: si el archivo local ya coincide con el hash esperado, no
  vuelve a descargar. `--verify-only` revisa los archivos ya descargados sin
  red.
- **Guarda de congelamiento del holdout (revisión 1).** Si el holdout ya
  existe en disco, el script se niega a sobrescribirlo (ni el CSV ni
  `HOLDOUT-SHA256.txt`) salvo que se pase `--force-holdout` explícitamente.
  Que el archivo exista puede significar que ya se usó para evaluar (Task 6);
  antes de esta revisión, `--cowsl2h` lo regeneraba en cada ejecución sin
  avisar.
- **Separador de oraciones consciente de abreviaturas y comillas (revisión
  1).** El separador original solo miraba `.`/`!`/`?` seguido de mayúscula, y
  fallaba con abreviaturas seguidas de nombre propio (`Sr. García` se cortaba
  en dos) y con diálogo entre comillas (`"Hola. ¿Cómo estás?"` se cortaba
  dentro de la cita). Ahora hay una lista de abreviaturas frecuentes (Sr.,
  Sra., Srta., Dr., Dra., Ud., Uds., etc., p. ej., EE. UU., núm., pág., Av.)
  tras las que no se corta, y no se corta dentro de comillas (`"…"` / `«…»`;
  cortar justo después de la comilla de cierre sí está permitido). Medido
  sobre el corpus real: 164 de los 2 881 ensayos usables cambian con el
  fix (61 cambian su estado de alineación —2 089 → 2 106 ensayos alineados—
  y 103 siguen alineados pero cambian los pares que producen); el total de
  pares sube de 23 165 a 23 355.

### `pairs-dev.csv` vs. holdout bloqueado

- `data/external/cowsl2h/pairs-dev.csv` (autores de dev, ~70 %): dentro del
  repo, pero en una carpeta gitignored (`data/external/cowsl2h/`); se
  regenera con el script, no se versiona.
- El holdout (autores de holdout, ~30 %) se escribe **fuera del
  repositorio**, en
  `C:\Users\Dovamul\Desktop\TESIS\documentos\datos-reservados\holdout-cowsl2h.csv`,
  con su hash en `HOLDOUT-SHA256.txt` (SHA-256, nombre de archivo y número de
  filas). El script nunca vuelve a leer ni imprimir el contenido del holdout
  después de escribirlo, salvo para recalcular el hash. **Nadie abre el
  holdout hasta la Task 6.**
- La partición es por **autor** (`autor_id`, columna `id` del CSV =
  estudiante), no por ensayo ni por oración, para que no haya fuga de estilo
  de un mismo autor entre dev y holdout (ver "Decisión: partición por autor,
  no por ensayo" arriba). Semilla fija 42
  (`scripts/fetch_external_data.split_by_author`). Cada ensayo tiene su
  propia clave única `f"{archivo}#{fila}"` (`essay_key`); `id` nunca se usa
  como clave de ensayo.
- Si el holdout ya existe en disco, `--cowsl2h` se niega a sobrescribirlo
  salvo que se pase `--force-holdout` (ver "Guarda de congelamiento del
  holdout" arriba).
- `data/external/cowsl2h-manifest.json` (sí versionado) trae la procedencia
  completa: commit, hash de cada CSV, conteos de cada paso del filtro,
  autores/ensayos/filas de cada lado de la partición (con su porcentaje real)
  y el hash del holdout, para que la Task 1 y la Task 6 puedan auditar el
  dataset sin volver a tocar el holdout.

## 2. Perfil de errores disléxicos (DysList)

- **Fuente:** Rello, L., Baeza-Yates, R., Llisterri, J. (2016). "A resource of
  errors written in Spanish by people with dyslexia and its linguistic,
  phonetic and visual analysis". *Language Resources and Evaluation*, 51(2),
  379–408. DOI [10.1007/s10579-015-9329-0](https://doi.org/10.1007/s10579-015-9329-0).
- **`data/external/dyslist-profile.json`:** proporciones transcritas
  directamente del artículo (PDF de acceso abierto en changedyslexia.org, ver
  campo `cita.acceso_abierto`), cada una con su sección/tabla de origen
  (`verificado: true` en todos los campos; no hay valores inventados). Incluye
  tipo de edición (sustitución 58.84 %, eliminación 26.30 %, inserción
  13.40 %, transposición 1.45 %; Tabla 5), sustituciones de vocales y
  consonantes por rasgo fonético compartido (Tablas 12 y 13), errores de
  palabra real, uniones/separaciones de palabras (Tabla 15) y motivación
  visual (letras borrosas, espejo, rotación).
- **Es solo una especificación de proporciones**, no un dataset de pares
  entrada/esperado: sirve para calibrar el generador de datos sintéticos de
  la Task 3 (por ejemplo, para que las sustituciones sintéticas respeten la
  proporción real de confusiones fonéticamente motivadas).
- **`data/external/dyslist-solicitud-correo.md`:** borrador de correo a la
  autora de contacto (`luzrello@gmail.com`, única dirección publicada en el
  artículo) solicitando `DysList_resource.csv.gz`, porque el servidor que lo
  aloja (`grupoweb.upf.es`) no responde. El agente redactó el borrador; el
  autor de la tesis debe enviarlo desde su propio correo. Si el recurso llega,
  se guarda en `documentos/datos-reservados/dyslist/` (fuera del repo) y se
  usa como conjunto de contraste en la Task 6, no para entrenar.
- **El corpus fuente de DysList sí es la población objetivo:** 83 textos
  manuscritos de niños y adolescentes de 6 a 15 años con dislexia diagnosticada
  (54 de profesores, 29 de padres), a diferencia de COWS-L2H. Por eso sus
  proporciones —y no sus textos, que no tenemos— son la pieza que compensa la
  limitación de población descrita abajo.

## 3. Holdout "de población" (diferido, Task 6)

Regla definida ahora, para cuando exista el piloto real (no se implementa en
esta tarea, no hay recolección propia todavía):

- Los **textos finales de las ejecuciones del piloto real**, con sus errores
  ya adjudicados por el corrector (los produce el propio backend en
  producción/piloto), forman un segundo holdout, pequeño, de la población
  objetivo real (niños con dislexia usando el teclado).
- Este holdout **nunca se usa para entrenar ni para ajustar** hiperparámetros
  ni umbrales; solo para evaluación final.
- Se evalúa en la **Task 6**, junto al holdout de COWS-L2H, y se **reporta
  aparte** (no se promedia con COWS-L2H): son poblaciones distintas y mezclar
  las métricas ocultaría si el modelo generaliza a la población real.
- Cuando exista, seguirá las mismas reglas de manejo que el holdout de
  COWS-L2H: fuera del repositorio, hash registrado, nunca releído salvo para
  verificar su integridad.

## 4. Limitación de población y cómo se compensa

**Limitación:** COWS-L2H son ensayos de estudiantes universitarios aprendiendo
español como segunda lengua (la mayoría con inglés como L1). No es escritura
de niños con dislexia: ni la edad, ni el perfil de errores (errores de
aprendiz de L2 vs. errores fonético/visuales de dislexia), ni el contexto de
producción (ensayo universitario vs. escritura escolar/teclado) coinciden con
la población objetivo de la tesis. Esta limitación debe declararse
explícitamente en la tesis al reportar cualquier métrica sobre el holdout de
COWS-L2H.

**Cómo se compensa, con dos piezas independientes:**

1. El **perfil DysList** (`dyslist-profile.json`) especifica las proporciones
   reales de errores disléxicos en español para calibrar los datos
   **sintéticos** de la Task 3 (frecuencia de sustitución vs. inserción vs.
   eliminación vs. transposición, y qué confusiones fonéticas/visuales son más
   probables), de modo que el entrenamiento no dependa solo del sesgo de
   COWS-L2H (errores de L2, no de dislexia).
2. El **holdout de población** (Sección 3, diferido a la Task 6) evalúa el
   modelo final contra texto real de la población objetivo, reportado por
   separado del holdout de COWS-L2H, para que ninguna métrica agregada
   esconda una brecha de generalización.

Ninguna de las dos piezas sustituye a la otra: el perfil DysList entra en el
entrenamiento (vía datos sintéticos), el holdout de población solo entra en la
evaluación final.
