# v4.2 — Hybrid search BGE + BM25 con weighted RRF

## Que es v4.2

Segundo intento de hybrid search. En v2, hybrid (MiniLM + BM25) dio 2/21 — peor que embeddings puros. En v4.2 se repite el experimento con componentes mejores: BGE-base (7/21 buenos) como denso y BM25-smart como sparse, con weighted RRF (alpha_dense=0.7, alpha_sparse=0.3) en vez del RRF sin pesos de v2.

La pregunta: hybrid con un denso fuerte + sparse debil mejora al denso solo, o lo empeora?

## Cambios respecto a v3/v4

| Componente | v3 (baseline) | v4.2 |
|------------|---------------|------|
| Retrieval denso | BGE-base cosine top-1 | BGE-base como componente de RRF |
| Retrieval sparse | — | BM25Okapi + tokenize_smart |
| Fusion | — | Weighted RRF (k=60, alpha=0.7/0.3) |
| Generation | Ollama llama3.2:3b | — (retrieval-only) |
| Evaluacion | Humano 3 ejes | Humano retrieval + juez retrieval |

## Resultado

**7/21 bueno. Igual que BGE puro.**

| Metrica | v3b (BGE puro) | v4.2 (hybrid) |
|---------|---------------|---------------|
| Buenos | 7/21 | 7/21 |
| Tangenciales | 8/21 | 0/21 |
| Malos | 6/21 | 14/21 |

### Composicion del resultado

| Tipo | Queries | Detalle |
|------|---------|---------|
| Gains (+bueno) | Q8, Q13 | Q8: sparse_wins (distributed.md → cuda.rst). Q13: compromise (sparse.rst → tensors.rst) |
| Losses (-bueno) | Q2, Q14 | Q2: compromise (cuda.rst → mps.rst). Q14: compromise (named_tensor.md → pipelining.md) |
| Net | 0 | Gains y losses se cancelan |

8/21 chunks cambiaron vs BGE puro. Change types: dense_wins 9, compromise 7, sparse_wins 2, consensus 3.

### Vs prediccion

| Prediccion | Esperado | Real | Error |
|------------|----------|------|-------|
| Retrieval bueno | 7/21 | 7/21 | **0** |
| Q13 se corrige | "sigue malo" (35%) | bueno | Subestimado |
| Buenos que se rompen | 0 | 2 (Q2, Q14) | Subestimado |

Primera prediccion exacta del proyecto en el agregado. La composicion sigue parcialmente equivocada (Q8 y Q2/Q14 no anticipados).

## Validacion cruzada del juez

GPT-4o-mini (mismo prompt de v4.1) sobre los chunks hibridos de v4.2:

| | Humano | Juez | Acuerdo |
|---|---|---|---|
| Escala ternaria (b/t/m) | 7b / 0t / 14m | 5b / 4t / 12m | **17/21 (81%)** |
| Escala binaria (b vs no-b) | 7b / 14(t+m) | 5b / 16(t+m) | **19/21 (90%)** |

4 desacuerdos: Q9, Q15 (humano=b, juez=t) y Q17, Q20 (humano=m, juez=t). Todos involucran la categoria tangencial. Las 4 queries que cambiaron de chunk (Q2, Q8, Q13, Q14) tienen acuerdo perfecto.

## Decision

**BGE puro queda como definitivo. Hybrid no se sube al producto.**

Regla pre-comprometida: "7-8/21 = hybrid es un wash, BGE puro queda como definitivo." El resultado cae exactamente en ese rango.

## Archivos

```
v4.2/
  prediction.md        — Prediccion pre-comprometida
  measure_hybrid.py    — Weighted RRF (BGE + BM25)
  hybrid_results.jsonl — Resultados de las 21 queries
  classify.py          — Clasificacion ciega (copiado de v3)
  labels_retrieval.csv — Clasificaciones humanas
  judge.py             — LLM-as-judge retrieval
  judge_retrieval.csv  — Clasificaciones del juez
  compare.py           — Comparacion hybrid vs baseline
  count.py             — Conteos de labels
  NOTES.md             — Analisis detallado
  README.md            — Este archivo
```

## Cierre de Fase 2

v4.2 cierra el ultimo frente abierto de retrieval. Proxima fase: Agents (v5).
