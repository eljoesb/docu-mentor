# DocuMentor v2.1 — BGE-base-en-v1.5 (camino 3: modelo de embeddings más grande)

## Qué se intentó en v2.1

Probar la hipótesis central que emergió del fracaso de v2: **el cuello de botella de retrieval era la capacidad semántica del modelo de embeddings, no el algoritmo de búsqueda**. v2 demostró que hybrid search (RRF) con un modelo chico (MiniLM-L6-v2, 384 dim) empeora las cosas. v2.1 ataca directamente el modelo.

Se reemplazó `all-MiniLM-L6-v2` por `BAAI/bge-base-en-v1.5` (768 dim, ~440MB). Sin cambiar chunking, sin cambiar corpus, sin cambiar queries. Un solo cambio controlado.

**Objetivo cuantitativo fijado al inicio:** subir `Retrieval: bueno` de 4/21 (baseline rebaselined en v2) a al menos 10/21. Umbral pre-comprometido en `prediction.md` antes de correr el experimento.

## Qué se construyó

### Pipeline de retrieval

`measure_bge.py` — retrieval puro con BGE-base. Sin BM25, sin RRF, sin LLM. Un solo cambio vs v1: el modelo de embeddings y el query prefix asimétrico (`"Represent this sentence for searching relevant passages: "` para queries, nada para documentos). Mismo chunking que v1 (split por `\n\n`, merge hasta ≥50 palabras).

### Metodología de evaluación ciega

La innovación metodológica de v2.1 no es el modelo — es cómo se evaluó. En v2, las clasificaciones se hicieron editando el archivo de código directamente, sin separación entre clasificar y contar, lo que introdujo riesgo de sesgo de confirmación.

v2.1 separó físicamente cada paso:

1. **`prediction.md`** — predicción desagregada por confianza, escrita *antes* de correr el experimento.
2. **`measure_bge.py`** — genera `bge_base_results.jsonl` con los chunks ganadores.
3. **`classify.py`** — muestra un chunk a la vez, sin contadores, sin predicciones visibles, `clear` entre cada query. Pide b/t/m y escribe a CSV. Crash-safe.
4. **`count.py`** — lee el CSV y produce una línea de conteos. Nada más.

Ningún paso tiene acceso a la información del siguiente. La clasificación es ciega por diseño.

## Resultado: 11/21 buenos

| Categoría | Queries | Count |
|-----------|---------|-------|
| Bueno | Q1, Q2, Q7, Q9, Q10, Q11, Q12, Q14, Q15, Q18, Q21 | 11 |
| Tangencial | Q3, Q6, Q8, Q19, Q20 | 5 |
| Malo | Q4, Q5, Q13, Q16, Q17 | 5 |

Comparación con sistemas anteriores:

| Sistema | Bueno | Tangencial | Malo |
|---------|-------|------------|------|
| v1 embeddings MiniLM (criterio estricto) | 4/21 | — | — |
| v2 hybrid RRF k=60 | 2/21 | 12/21 | 7/21 |
| **v2.1 BGE-base puro** | **11/21** | **5/21** | **5/21** |

**Lectura:** BGE-base casi triplica el baseline de v1 y más que quintuplica el resultado de v2 hybrid. El modelo de embeddings *era* el cuello de botella. Camino 3 confirmado.

## Predicción vs realidad

Se predijo 9/21 (rango 7-11, mediana 9). Resultado: 11/21, en el extremo optimista del rango.

Pero el número acertado oculta que la composición falló casi completamente:

| Bucket | Predicción | Resultado | Acierto |
|--------|-----------|-----------|---------|
| Baseline (Q1, Q9, Q19, Q21) | 4/4 retiene | 3/4 (perdió Q19) | Parcial |
| Alta confianza (Q2, Q13, Q20) | 3/3 flipean | 1/3 (solo Q2) | Mal |
| Confianza media (Q3, Q12, Q15) | 2/3 flipean | 2/3 (Q12, Q15) | Bien |
| Apuestas largas (Q8, Q11, Q14) | 0/3 flipean | 2/3 (Q11, Q14) | Mal |
| Descartadas gap corpus (Q7, Q10, Q18) | 0/3 | 3/3 flipean | Mal |

**El modelo de predicción por query fue pobre.** Las "altas confianzas" fallaron (1/3), las "imposibles" acertaron (3/3). El total se acercó por compensación de errores, no por calibración.

### Errores de predicción más informativos

**Las "descartadas por gap de corpus" (Q7, Q10, Q18) fueron el error más grande.** Se asumió que no existían chunks buenos para estas queries en el corpus. BGE los encontró. La hipótesis "gap de corpus, no de modelo" era incorrecta — era gap de modelo todo el tiempo. MiniLM no podía conectar "memorizing training data" con el chunk de faq.rst sobre overfitting, ni "weights not changing" con el chunk de optim.md sobre gradientes. BGE sí.

**Q13 (numpy to tensor) fue la sorpresa diagnósticamente más rica.** Se predijo como alta confianza de flipear. BGE fue a sparse.rst — un chunk sobre tensores sparse que no menciona "numpy" ni una vez. El análisis post-hoc reveló que BGE matcheó el patrón semántico "convert X to Y tensor" (sparse.rst muestra `torch.tensor(...)` + `a.to_sparse()`) perdiendo la especificidad del sustantivo "numpy". En contraste, hybrid de v2 acertaba Q13 porque BM25 (rank 2) matcheaba "numpy" literalmente y RRF lo surfaceaba. Es el caso canónico donde retrieval puramente denso pierde especificidad que BM25 capturaría.

**Q19 (loss NaN) fue la única regresión de un bueno de v1.** MiniLM acertaba con autograd.rst (ejemplo de división por cero → NaN en gradientes). BGE fue a faq.rst (acumulación de historial en training loop — problema real, pero no NaN). El chunk correcto quedó en rank 4 con score 0.6157, vs rank 1 del ganador con score 0.6341. Margen: 0.018.

A diferencia de Q13 (fallo representacional profundo — BGE fue a un tema completamente distinto), Q19 es un fallo de densidad léxica: faq.rst tiene "loss" 5 veces en contexto de training loop; autograd.rst tiene "loss" pero embebido en vocabulario de gradientes. "nan" como token probablemente pesa poco en el embedding promediado. BGE *casi* acierta — es un near-miss, no una confusión conceptual. Q13 necesita hybrid search para corregirse. Q19 podría corregirse con hybrid (BM25 daría señal fuerte a "nan") pero también podría corregirse con un modelo más grande o con fine-tuning.

## Conclusiones principales

### 1. La regla de decisión se cumplió — camino 3 confirmado

11/21 ≥ 10/21. La regla pre-comprometida en prediction.md dice: "camino 3 confirmado. El modelo de embeddings era el cuello de botella." BGE-base, el experimento más barato del camino 3, fue suficiente. No hace falta BGE-large.

### 2. La calibración agregada fue un espejismo de cancelación

Predicción: 9. Resultado: 11. Δ = +2. Parece buena calibración. Pero los buckets "alta confianza" y "baseline" sobreestimaron por 3, mientras que "apuestas largas" y "descartadas" subestimaron por 5. El neto (+2) es compensación de errores, no modelo mental calibrado. Para futuros experimentos: medir calibración por bucket, no por total.

### 3. La frontera corpus/modelo no se puede determinar con un solo modelo

Q7, Q10, Q18 fueron descartadas como "gap de corpus" — asumiendo que ningún chunk podía responderlas. BGE encontró chunks buenos para las tres. Lo que parecía "el documento no existe" era "MiniLM no lo encuentra". La conclusión "el corpus no tiene la respuesta" solo es afirmable después de probar con un modelo suficientemente capaz.

### 4. Los embeddings capturan intención pero pierden especificidad nominal — y eso reabre hybrid search

Q13 lo demuestra: BGE matcheó "convert X to Y tensor" pero no distinguió X=numpy de X=sparse. Es una limitación estructural del retrieval denso, no de BGE específicamente. BM25 preserva la especificidad nominal (matchea "numpy" literalmente) mientras embeddings preserva la intención. En queries donde el sustantivo clave es raro y específico, BM25 tiene ventaja estructural sobre cualquier modelo denso.

v2 hybrid falló porque MiniLM + BM25-smart eran componentes de calidad dispar (4/21 vs ~2/21). Con BGE como componente denso (11/21), ambos serían de calidad comparable — exactamente el régimen donde hybrid search funciona. Esta hipótesis no existía antes del deep dive de Q13. Ahora tiene razonamiento mecánico concreto.

## Qué deja abierto v2.1

**11/21 es suficiente para confirmar el camino, no para declarar victoria.** 10 queries siguen sin retrieval bueno. De esas 10:

- 5 son estructuralmente imposibles (OOD/opinión): Q4, Q5, Q6, Q16, Q17. No hay chunk correcto en el corpus. Esto no es un problema de retrieval.
- 5 son fallos reales del sistema: Q3, Q8, Q13, Q19, Q20. El corpus tiene contenido relevante pero BGE no lo encontró.

Los 5 fallos reales sugieren dos direcciones:

1. **Hybrid con BGE como componente denso.** Q13 es el caso más claro: BM25 encontraba tensors.rst por keyword match literal. BGE puro pierde esa señal. Un hybrid (BGE + BM25-smart) con RRF recalibrado podría capturar lo mejor de ambos. Con BGE como base en vez de MiniLM, el problema de v2 ("señal densa mediocre diluida por RRF") desaparece.

2. **Chunking más granular para queries específicas.** Q3, Q8, Q20 podrían beneficiarse de chunks más cortos o de chunking semántico en vez de chunking por párrafo.

## Artefactos producidos en v2.1

- `measure_bge.py` — retrieval con BGE-base-en-v1.5, prefix asimétrico, caché de embeddings.
- `classify.py` — clasificación ciega: un chunk a la vez, b/t/m, sin contadores, crash-safe.
- `count.py` — cuenta labels de un CSV. 10 líneas.
- `compare.py` — cruza predicciones contra labels reales por bucket.
- `prediction.md` — predicción pre-comprometida con reglas de decisión.
- `bge_base_results.jsonl` — resultados crudos de BGE para las 21 queries.
- `labels_bge_base_results.csv` — clasificaciones ciegas (b/t/m) de los 21 chunks.
- `eval_dataset.jsonl` — copia del dataset de v1, inalterado.
- `embeddings_cache_bge_base.pkl` — caché de embeddings del corpus con BGE-base.

## Lecciones transferibles de v2.1

**1. En un pipeline RAG denso, el modelo de embeddings es el componente de mayor leverage — más que el chunking, más que el algoritmo de fusión, más que el reranker.** Un sistema con retrieval simple y un modelo bueno supera a un sistema con retrieval sofisticado y un modelo mediocre. Antes de optimizar la arquitectura de búsqueda, conviene agotar el eje de calidad del modelo.

**2. Una regresión en medio de una mejora neta es información concreta sobre lo que el modelo nuevo pierde, y merece un deep dive independiente.** En cualquier migración de modelo, el balance neto positivo tiende a ocultar los casos donde el modelo anterior era mejor. Cada regresión es una pista específica sobre las debilidades del modelo nuevo — ignorarla porque el neto es +7 es desperdiciar la señal diagnóstica más barata que existe.

**3. La metodología de evaluación es un artefacto del proyecto tanto como el código.** En cualquier sistema de ML donde la evaluación involucra juicio humano, separar físicamente la clasificación del conteo y de la predicción tiene valor medible en la confiabilidad del resultado. El sesgo de confirmación no se resuelve con disciplina mental — se resuelve con diseño de proceso que haga imposible ver lo que no debés ver mientras clasificás.
