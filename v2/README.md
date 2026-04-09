# DocuMentor v2 — Hybrid Search con BM25 + RRF

## Qué se intentó en v2

Atacar el cuello de botella estructural identificado al final de v1: **retrieval**. El eval de v1 mostró que solo 3 de 21 chunks recuperados eran buenos bajo criterio estricto (14%), y que la "utilidad final" del 48% estaba inflada porque el LLM rescataba con conocimiento pre-entrenado cuando el retrieval fallaba.

La hipótesis a probar: **dense retrieval (embeddings semánticos) + sparse retrieval (BM25), combinados vía Reciprocal Rank Fusion, recuperan más chunks relevantes que cualquiera de los dos por separado**. Es la arquitectura estándar de hybrid search en RAG de producción.

**Objetivo cuantitativo fijado al inicio:** subir `Retrieval: bueno` de 3/21 a al menos 10/21 contra el mismo `eval_dataset.jsonl`, sin cambiar el LLM, sin cambiar el prompt, sin cambiar el dataset.

## Qué se construyó

v2 se dividió en tres bloques, cada uno con su propia medición.

### Bloque 1 — BM25 puro

Se implementó `finder_bm25.py` usando `rank_bm25` con dos tokenizadores distintos:

- **`tokenize`** (dumb): lowercase + split por espacios.
- **`tokenize_smart`**: lowercase + split por espacios, puntos, guiones, paréntesis, y otros separadores comunes en nombres de API.

Se corrieron las 21 queries del eval contra BM25 puro con ambos tokenizadores. El script `measure_bm25.py` produce una tabla comparativa lado a lado con los scores de embeddings de v1.

**Hallazgo clave del Bloque 1:** el tokenizer tonto destruye el único caso donde BM25 debería brillar sobre embeddings (queries con nombres de API como `torch.nn.functional.linear`). El tokenizer smart mejora dramáticamente ese caso específico (Q11 pasó de 9.56 a 20.12 en score), pero introduce un trade-off en otras queries donde el name-drop literal ayudaba a dumb.

**Hallazgo inesperado:** BM25 dumb convergía sistemáticamente a `community/governance.rst` para tres queries completamente distintas (Q2, Q5, Q16). El análisis mostró que ese archivo tiene un chunk con alta repetición de palabras comunes (`how`, `do`, `if`, `you`) que domina el score de BM25 a pesar del IDF. Es el modo de falla clásico de BM25 cuando las queries son conversacionales.

### Bloque 1.5 — Verificación con criterio estricto

Durante el análisis de BM25, se descubrió que el criterio de evaluación original de v1 era demasiado laxo. Específicamente, en varias queries se había clasificado como "bueno" un chunk que solo mencionaba el nombre literal de la API buscada (`torch.from_numpy`, por ejemplo) sin explicar su uso. El análisis posterior mostró que esos chunks eran tangenciales, no buenos.

**Consecuencia:** se re-calibró el criterio a una definición más estricta: *"bueno = el chunk responde la pregunta, no solo la menciona"*. Bajo este criterio, el baseline real de v1 embeddings es probablemente 2-3/21, no 3/21.

Este re-análisis es importante para comparar v1 vs v2 bajo la misma vara. Sin el re-análisis, cualquier mejora medida en v2 habría estado contaminada por el cambio de metodología.

### Bloque 2 — Hybrid search con RRF

Se implementó `finder_hybrid.py` combinando embeddings + BM25-smart usando Reciprocal Rank Fusion con `k=60` (estándar de la literatura). El script `measure_hybrid.py` corre las 21 queries con el pipeline completo y produce una tabla con clasificación del tipo de fusión (consensus, dense_wins, sparse_wins, compromise).

**Resultado:** 2 de 21 chunks buenos bajo criterio estricto. **Peor que v1 embeddings puros.**

## Por qué v2 no cumplió el objetivo

El resultado no es un bug de implementación. RRF con k=60 funciona exactamente como está diseñado. El problema es que fue aplicado en el régimen incorrecto.

**El mecanismo del fallo:** RRF con k=60 premia chunks que están "decentes en ambos rankings" sobre chunks que están "perfectos en uno solo". La fórmula `1/(k+rank)` es muy suave en los ranks bajos — la diferencia entre rank 1 y rank 5 es de apenas 0.001 puntos. Eso significa que un chunk que está en rank 3 de embeddings y rank 3 de BM25 obtiene un score más alto que un chunk en rank 1 de embeddings pero rank 200 de BM25.

En un sistema donde **ambos componentes son de calidad similar**, ese comportamiento es correcto — el consenso entre dos señales independientes es más confiable que el top de una sola. En v2, las dos señales no son de calidad similar: embeddings es mediocre (3/21 buenos) y BM25 es peor (probablemente 1-2/21 buenos en este corpus). RRF combinó un sistema mediocre con un sistema peor y produjo algo peor que ambos.

**Esto tiene un nombre en la literatura:** naive ensembling hurts when component quality differs significantly. Es un resultado clásico que se enseña en cursos de ensemble learning y combinación de modelos. v2 lo vivió en la práctica con datos propios.

## Análisis específico de las queries

Cinco queries se analizaron en detalle en `NOTES.md`:

- **Q11 (`what does torch.nn.functional.linear do`):** hybrid eligió `fx.md` por influencia de BM25-smart. Tangencial. Ni embeddings solo ni BM25 solo ni hybrid encontraron un chunk que realmente responda la pregunta, probablemente porque el chunk canónico no existe en el corpus en forma de párrafo bien formado.
- **Q7 (`how do I stop my model from memorizing the training data`):** regresión clara. Hybrid eligió un chunk sobre `torch.compile` en Intel XPU. Es el fallo de vocabulario semántico que nadie puede resolver sin conocer el dominio.
- **Q13 (`convert numpy array to pytorch tensor`):** el único caso genuino donde RRF agregó valor. Hybrid encontró `tensors.rst` con un ejemplo de `torch.tensor(np.array(...))` que ninguno de los dos sistemas individuales había elegido como top-1. Serendipia del compromise — bueno, pero no replicable.
- **Q2 (`how do I move a tensor to GPU`):** lateral. Hybrid eligió `notes/mps.rst`, que muestra el patrón `.to(device)` pero para Apple Silicon en vez de CUDA. Técnicamente relacionado, prácticamente irrelevante para un usuario con GPU NVIDIA.
- **Q18 (`my model weights are not changing during training`):** sin cambio. Hybrid eligió el mismo chunk de `optim.md` que embeddings — un ejemplo de MoE training loading con optimizer state, no relacionado con el problema del usuario.

## Conteos agregados bajo criterio estricto

| Sistema | Bueno | Tangencial | Malo |
|---------|-------|------------|------|
| v1 embeddings (criterio laxo, original) | 3/21 | 8/21 | 10/21 |
| v1 embeddings (criterio estricto, re-analizado) | 2/21 | 9/21 | 10/21 |
| v2 BM25-smart solo | ~1-2/21 | ~5/21 | ~14/21 |
| v2 hybrid RRF k=60 | 2/21 | 12/21 | 7/21 |

**Lectura honesta:** v2 no rompió la barrera de 10/21. Tampoco rompió la de 5/21. No mejoró el retrieval de forma medible. El único cambio real fue mover queries de "malo" a "tangencial" — el sistema encuentra chunks más relacionados al tema, pero todavía no encuentra chunks que respondan la pregunta.

## Lo que v2 destapó para v2.1

El fracaso de v2 descarta una hipótesis ("hybrid search resuelve el retrieval") y revela una más profunda: **el cuello de botella quizás no es el algoritmo de retrieval, sino la estructura del corpus indexado**.

Evidencia:

- La query Q10 (`difference between .size() and .shape`) pide una comparación directa entre dos APIs. La respuesta canónica probablemente es una sola oración tipo *"`.shape` es un alias de `.size()` para compatibilidad con NumPy"*. Si esa oración existe en los docs de PyTorch, está enterrada adentro de un chunk de 50+ palabras que habla de muchas otras cosas. Ni embeddings ni BM25 la pueden recuperar como señal dominante porque está diluida.
- El chunking actual fusiona párrafos hasta llegar a 50 palabras mínimo. Eso fue la solución correcta en v0 (cuando el problema era chunks demasiado cortos), pero probablemente es excesivo para queries específicas que necesitan respuestas puntuales.

**Tres caminos posibles para v2.1**, en orden de invasividad creciente:

1. **Weighted RRF:** agregar un parámetro `α` que pondere más fuerte embeddings sobre BM25. Costo: 10 minutos. Mejora esperada: marginal (pasar de 2/21 a quizás 3-4/21).
2. **Chunking más granular:** reducir `min_palabras` de 50 a 15-20 para que los detalles específicos no se diluyan. Costo: re-indexar todo el corpus, regenerar caché, re-correr todo el eval. Mejora esperada: incierta — algunos casos mejoran, otros empeoran.
3. **Modelo de embeddings más grande:** reemplazar `all-MiniLM-L6-v2` (384 dim, 80MB) por `BAAI/bge-base-en-v1.5` (768 dim, ~440MB) o similar. Costo: 30 minutos de setup, latencia 2-3x. Mejora esperada: depende de si el cuello de botella era capacidad semántica del modelo o estructura del corpus.

La decisión entre los tres se toma en v2.1 con los datos actuales en la mano. El camino 3 es el menos invasivo estructuralmente y ataca la hipótesis "MiniLM-L6 es demasiado chico para docs técnicos con vocabulario especializado". Si no mueve la aguja, entonces el problema es chunking y va el camino 2.

## Artefactos producidos en v2

- `finder_bm25.py` — implementación de BM25 con dos tokenizadores, funciones reutilizables para retrieval sparse.
- `measure_bm25.py` — script de medición que compara embeddings vs BM25-dumb vs BM25-smart.
- `finder_hybrid.py` — implementación de RRF combinando embeddings + BM25-smart.
- `measure_hybrid.py` — script de medición con clasificación de tipo de fusión.
- `bm25_results_dumb.jsonl` — resultados crudos de BM25 con tokenizer tonto.
- `bm25_results_smart.jsonl` — resultados crudos de BM25 con tokenizer smart.
- `hybrid_results.jsonl` — resultados crudos del pipeline hybrid.
- `NOTES.md` — análisis cualitativo de las queries, bugs en la medición detectados durante el análisis, re-calibración del criterio.
- `eval_dataset.jsonl` — copia del dataset de v1, inalterado.

## Lecciones transferibles de v2

**1. Implementar una solución estándar no garantiza que funcione en tu dominio específico.** Hybrid search con RRF es la arquitectura canónica para RAG en 2024-2026. Existe literatura, tutoriales, y sistemas en producción usándola. Y en este corpus específico, no mejoró nada. El hecho de que una técnica sea estándar no la hace automáticamente aplicable — hay que medir.

**2. Una métrica que no se mueve es información valiosa.** Si v2 hubiera subido de 3 a 5 o 6 buenos, probablemente se habría aceptado "hybrid search funciona" sin entender por qué solo parcialmente. El hecho de que *empeoró* forzó el diagnóstico profundo y descubrió que el problema es más fundamental que el algoritmo de retrieval.

**3. Los criterios de evaluación evolucionan con el entendimiento del problema.** El criterio de v1 era laxo (premiaba name-drops). El criterio de v2 es estricto (exige respuesta real). Ese cambio no es inconsistencia — es calibración. Pero implica que cualquier comparación v1 vs v2 tiene que usar el criterio nuevo en ambos lados, no comparar laxo con estricto.

**4. Ensemble de señales de calidad dispar puede ser peor que la mejor señal individual.** Contraintuitivo pero demostrado con datos propios. La intuición de "más información siempre ayuda" es falsa cuando la información extra es de peor calidad que la existente y la función de combinación no sabe ponderar adecuadamente.

**5. El debugging de RAG requiere mirar cada query individualmente.** Los conteos agregados esconden patrones cruciales. Fue solo leyendo los chunks ganadores uno por uno que se descubrió el patrón de `governance.rst` dominando por palabras comunes, y solo así se detectó el sesgo del criterio laxo hacia name-drops.
