# Datos de LoRA v5 (escritura infantil), 2026-09-30

Motivación (auditoría externa C y ablación del T5, ver README §entrenamiento): v4 se entrenó con
pares de **subtítulos de adultos** (frases de 6-7 palabras, 1-2 errores, temas adultos y algunos
objetivos con faltas, "Yo vere"), mientras que la escritura infantil real tiene frases más largas,
3-6 errores (media 2,97 en `test_ninos_reales_dev`), mucha segmentación y vocabulario escolar.
Además el T5 no recibe el texto crudo sino lo que dejan las capas 0-3 (reglas + BETO), con sus
propios errores residuales (`pdio -> podio`, `mipis -> mapas`), que v4 nunca vio.

## Frases limpias (`clean_<fuente>.txt`, `scripts/build_v5_clean.py`)

| Fuente | Frases | Origen | Licencia / declaración |
|---|---|---|---|
| `claude` | 4 798 | 8 agentes Claude (2026-09-30), un tema cada uno (escuela, familia, animales, cuentos, comida y fiestas, juegos, cartas, lugares del Perú), registro escolar peruano, sin errores. Sin acceso a ningún archivo del proyecto. Originales en `claude/`. | **Generadas por IA**: declararlo en la tesis. |
| `cowsl2h` | 21 744 | COWS-L2H, solo autores de DESARROLLO (misma partición por autor que el holdout bloqueado): oraciones corregidas por el profesor y oraciones que dejó sin cambios. | Apache-2.0. Adultos L2; alguna falta que el profesor no corrigió. |
| `subtitulos` | 11 249 | Objetivos de `training_pairs_clean.csv` (v1-v4) con todas las palabras válidas en el léxico, sin formas sin tilde ("vere") y sin temas adultos. | Los de v1-v4. |
| `dominio_publico` | 868 | Project Gutenberg #63424 (Laval, *Cuentos populares en Chile*, 1923) y #36558 (Coloma, *Ratón Pérez*, 1902), ortografía modernizada (`scripts/fetch_v5_public_domain.py`). Originales en `dominio_publico/`. | Dominio público (EE. UU.). |

Ninguna frase que coincida (normalizada) con una entrada o referencia de `data/test_*.csv` o
`data/eval_gold.csv` entra al corpus (577 de COWS-L2H y 1 de Claude excluidas). La reserva de
`test_ninos_reales` solo se usa para excluir; su contenido no se imprime.

## Corrupción infantil (`scripts/corrupt_child.py`)

Densidad: 30 % sin errores, 25 % 1-2, 30 % 3-4, 15 % 5-6 errores por frase. Operaciones
(peso): partir palabra (0,13: solo tras `a/al/de/en/con/es/por/para/sin` o donde ambos trozos
son palabras: `en contró`, `a bajo`), pegar (0,12: nunca si la unión es otra palabra real,
`se paró` no es `separó`), b/v, h, c/s/z, tilde, letra omitida, g/j, ll/y, haber (ha/he/hay/a
ver), concordancia de número del verbo, r/rr, género del artículo, minúscula de nombre, m/n,
qu, porque. Por frase y sin contar: quitar todas las tildes (35 %) o la puntuación (30 %).

## Pares de entrenamiento (`scripts/generate_v5_data.py`)

Claude y dominio público con dos corrupciones por frase; 12 000 de COWS-L2H y 4 000 de
subtítulos con una. Cada entrada corrompida pasa por las capas 0-3 del pipeline y el par es
(lo que recibiría el T5, frase limpia); traza completa en `pairs_core.csv`. Se suman los
bloques dirigidos de v2-v4 (concordancia con el generador corregido, subjuntivo, controles de
indicativo/prospectivo/predicado). Dos variantes por la proporción de pares idénticos
(enseñan a no tocar lo correcto; en v3 un 7 % hizo que T5 dejara "se callo"):
`../training_pairs_v5a.csv` (~15 %) y `../training_pairs_v5b.csv` (~5 %).
