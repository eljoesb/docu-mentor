# v3 — Pipeline completo: BGE retrieval + Ollama generation

## Que es v3

v3 es la primera evaluacion end-to-end del pipeline RAG completo. v1 evaluo los tres ejes pero con MiniLM (retrieval debil). v2.1 evaluo solo retrieval con BGE-base. v3 combina ambos: BGE-base para retrieval + Ollama llama3.2:3b para generation, midiendo retrieval, generation y utilidad sobre las mismas 21 queries.

Ademas, v3 incluyo un micro-experimento de prompt engineering que resulto ser la intervencion con mejor ratio impacto/esfuerzo del proyecto.

## Cambios respecto a v2.1

| Componente | v2.1 | v3 |
|------------|------|----|
| Retrieval | BGE-base, solo retrieval | BGE-base, pipeline completo |
| Generation | No evaluada | Ollama llama3.2:3b |
| Evaluacion | 1 eje (retrieval) | 3 ejes (retrieval / generation / utilidad) |
| Prompt | N/A | Dos variantes probadas (v3a y v3b) |

## Dos corridas: v3a y v3b

### v3a — Prompt original

```
Use ONLY the following context to answer the question.
If the context doesn't contain the answer, say "I don't know based on the provided context."
```

### v3b — Prompt mejorado (una linea agregada)

```
Use ONLY the following context to answer the question.
If the context doesn't contain the answer, say "I don't know based on the provided context."
If the context discusses a related topic but doesn't directly answer the question,
say so explicitly instead of adapting the context to fit the question.
```

Los chunks recuperados son **identicos** en ambas corridas (verificado: 0/21 diferencias). Solo cambio la generacion.

## Resultados oficiales (v3b, criterio estricto)

| Eje | Positivo | Medio | Negativo |
|-----|----------|-------|----------|
| Retrieval | **7/21** bueno | — | — |
| Generation | **14/21** fiel | 5/21 mixto | 2/21 inventado |
| Utilidad | **11/21** util | 8/21 parcial | 2/21 danino |

### Comparacion historica

| Eje | v1 | v3b | Delta |
|-----|----|----|-------|
| Retrieval bueno | 3/21 | 7/21 | +4 |
| Generation fiel | 8/21 | 14/21 | **+6** |
| Generation inventado | 11/21 | 2/21 | **-9** |
| Utilidad util | 10/21 | 11/21 | +1 |
| Utilidad danino | 6/21 | 2/21 | **-4** |

## Tres hallazgos

### 1. Una linea de prompt elimino 7/9 invenciones

El cambio mas impactante de v3 no fue de retrieval ni de modelo — fue una frase en el prompt. En v3a, el LLM inventaba respuestas cuando el chunk era tangencial. En v3b, ante el mismo chunk, decia "no se" o "el contexto habla de X pero no de Y".

| Metrica | v3a | v3b |
|---------|-----|-----|
| Generation fiel | 7/21 | 14/21 |
| Generation inventado | 9/21 | 2/21 |

Las 2 invenciones resistentes (Q10: .size/.shape, Q19: loss NaN) comparten un patron: el chunk contiene terminos de la pregunta en contexto semanticamente diferente, y 3B no distingue la diferencia a pesar de la instruccion.

### 2. Habia dos cuellos de botella simultaneos

En v1 se identifico retrieval como el cuello de botella. v3 revelo que **generation tambien era cuello de botella**. Arreglar solo retrieval (v2.1) no alcanzaba porque el LLM destruia los buenos chunks con invenciones. Arreglar solo generation tampoco habria alcanzado porque los chunks eran malos. Habia que arreglar ambos.

Se arreglaron con costos radicalmente distintos:
- Retrieval: modelo BGE-base de 440MB, cache de embeddings, cambio de arquitectura. **Costo: horas.**
- Generation: una linea de texto. **Costo: minutos.**

### 3. Varianza intra-evaluador de ±8 queries

La clasificacion de retrieval sobre los **mismos chunks** dio 15/21 bueno (sesion cansada) vs 7/21 bueno (sesion fresca). La diferencia de 8 queries es ruido del clasificador humano, no senal del sistema.

Implicacion: con 21 queries y clasificacion manual, diferencias menores a 8 puntos no son estadisticamente confiables. LLM-as-judge es necesario para estabilizar metricas en versiones futuras.

## Regla de decision: perros en la nieve

Del prediction.md:
- Se cierran si utilidad >= 13/21 y daninos <= 3.

Resultado: **parcialmente cerrados.**
- Daninos: 2/21 <= 3. Si.
- Utilidad: 11/21 < 13. No.

El sistema dejo de mentir pero no es suficientemente util. Para subir utilidad, el retrieval necesita traer mejores chunks — el LLM ya hace lo que puede con lo que recibe.

## Metodologia

### Pipeline

1. **Chunking:** split en `\n\n`, buffer hasta >= 50 palabras, residuo pegado al ultimo chunk del archivo. 2912 chunks totales.
2. **Retrieval:** BGE-base-en-v1.5 con prefijo asimetrico para queries. Busqueda exacta por coseno (no ANN/ivfflat). Top-1 chunk.
3. **Generation:** Ollama llama3.2:3b, `stream=False`. Prompt template con contexto + pregunta.

### Evaluacion de tres ejes

Tres pasadas independientes de clasificacion ciega, cada una mostrando solo la informacion relevante:

| Pasada | Info visible | Etiquetas |
|--------|-------------|-----------|
| Retrieval | query + chunk | b(ueno) / t(angencial) / m(alo) |
| Generation | chunk + respuesta LLM | f(iel) / x(mixto) / i(nventado) |
| Utilidad | query + respuesta LLM | u(til) / p(arcial) / d(anino) |

La separacion por pasada minimiza sesgo: al clasificar retrieval no ves la respuesta del LLM; al clasificar utilidad no ves el chunk.

## Archivos

```
v3/
  prediction.md                 — Predicciones pre-comprometidas
  measure_full_pipeline.py      — Script de evaluacion (BGE + Ollama)
  classify.py                   — Clasificacion ciega por eje (--axis retrieval/generation/utility)
  count.py                      — Conteo de etiquetas (detecta eje automaticamente)
  eval_dataset.jsonl            — Las 21 queries (inalteradas desde v1)
  pipeline_results.jsonl        — Resultados v3a (prompt original)
  pipeline_results_v3b.jsonl    — Resultados v3b (prompt mejorado)
  labels_retrieval.csv          — Clasificacion retrieval v3a
  labels_generation.csv         — Clasificacion generation v3a
  labels_utility.csv            — Clasificacion utilidad v3a
  labels_retrieval_v3b.csv      — Clasificacion retrieval v3b (oficial)
  labels_generation_v3b.csv     — Clasificacion generation v3b (oficial)
  labels_utility_v3b.csv        — Clasificacion utilidad v3b (oficial)
  embeddings_cache_bge_base.pkl — Cache de embeddings BGE
  NOTES.md                      — Analisis detallado por query y diagnosticos
```

## Siguiente paso

1. **LLM-as-judge calibrado** — La varianza intra-evaluador hace insostenible la clasificacion manual. Calibrar contra v3b como ground truth.
2. **Hybrid search BGE + BM25** — Retrieval en 7/21 tiene margen. BM25 aporta senal lexica que embeddings puros pierden (ej: Q13 "numpy").
3. **Modelo 8B** — Solo si 1 y 2 no cierran los perros en la nieve.
