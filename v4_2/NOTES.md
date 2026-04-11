# v4.2 — Hybrid search BGE + BM25: Notas de evaluacion

## Resultado

**7/21 bueno.** Igual que el baseline (BGE puro en v3b). Prediccion: 7/21. Error: 0. Primera prediccion exacta del proyecto.

## Configuracion

- Dense: BAAI/bge-base-en-v1.5 (alpha=0.7)
- Sparse: BM25Okapi con tokenize_smart (alpha=0.3)
- Fusion: Weighted Reciprocal Rank Fusion, k=60
- Formula: `score(d) = 0.7/(60 + rank_dense) + 0.3/(60 + rank_sparse)`

## Composicion: +2 gains, -2 losses, net = 0

### Gains

**Q8 (multiple GPUs): tangencial → bueno.** sparse_wins. BM25 mando a notes/cuda.rst (contenido sobre CUDA device management) en vez de distributed.md (deprecated DistributedDataParallel). Ganancia no anticipada en la prediccion — BM25 encontro un chunk mejor que BGE por match lexico directo con "GPU".

**Q13 (numpy to tensor): malo → bueno.** compromise. El caso canonico del proyecto. BGE tenia tensors.rst en rank 2 (cosine 0.7314), no en rank 10-15 como se predijo. El rescate funciono por un mecanismo distinto al predicho: no fue BM25 tirando hacia arriba el chunk correcto, sino BM25 no apoyando al impostor (sparse.rst). tensors.rst tenia soporte bilateral (D:2, S:2 → rrf=0.016129); sparse.rst tenia soporte unilateral (D:0, S:7 → rrf~0.01167 sin contribucion BM25 del chunk especifico). Margen de victoria: 0.000015.

### Losses

**Q2 (move tensor to GPU): bueno → malo.** compromise. BGE tenia cuda.rst (NVIDIA GPU, rank 0). Hybrid fue a mps.rst (Apple Silicon GPU, D:6, S:2). El chunk de mps.rst muestra `model.to(mps_device)` — tecnicamente responde la pregunta, pero clasificado como malo porque es especifico a un device que no es el caso general.

**Q14 (flatten before linear): bueno → malo.** compromise. BGE tenia named_tensor.md (bueno). Hybrid fue a distributed.pipelining.md (D:3, S:3). Mismo mecanismo de compromiso que Q13 pero en direccion contraria: el chunk incorrecto tenia soporte bilateral que desplazo al correcto.

### Patron

Los 4 cambios significativos son "compromise" (3) o "sparse_wins" (1). Ninguno es consensus o dense_wins. Esto confirma que weighted RRF con alpha=0.7 protege los buenos de BGE cuando BGE tiene rank 0 firme, pero no cuando BGE tiene rank 2-3 — ahi el compromiso puede ir en cualquier direccion.

## Verificacion de hipotesis mecanica: Q13 y Q19

### Q13 — hipotesis parcialmente confirmada, mecanismo distinto

**Prediccion:** tensors.rst en BGE rank 10-15, BM25 lo rescata via "numpy" como keyword de IDF alto.

**Realidad:** tensors.rst en BGE rank **2** (cosine 0.7314). sparse.rst (impostor) en BGE rank 0 (cosine 0.7508). Margen: 0.019 — mucho mas chico de lo que el modelo mental sugeria. BM25 no le dio a tensors.rst un rank particularmente alto (rank 2, no rank 0), pero tampoco le dio soporte al impostor (sparse.rst en BM25 rank 7).

**Error del modelo mental:** sobreestime la distancia entre el chunk correcto y el incorrecto en BGE. Asumi que BGE "pierde especificidad nominal" de forma severa (rank 10-15). En realidad la pierde marginalmente (rank 2, 0.019 de cosine). La correccion: los embeddings densos no pierden la especificidad nominal tanto como se pensaba — la pierden justo lo suficiente para que otro chunk gane por margen minimo.

### Q19 — hipotesis refutada

**Prediccion:** BM25 da senal fuerte a autograd.rst via "nan" como token de IDF alto. autograd.rst (BGE rank 3, margen 0.018) sube por encima de faq.rst.

**Realidad:** autograd.rst no esta en el BM25 top-10. "nan" aparece dentro de un code block como output (`# [nan, 1]`), no como token prominente. BM25 no distingue "nan es el tema del chunk" de "nan aparece como un valor en una salida de codigo". Peor: faq.rst (el impostor) tiene BM25 rank 2, asi que tiene soporte bilateral y gana comodo.

**Error del modelo mental:** asumi que un token raro en cualquier posicion del chunk genera senal BM25 fuerte. En realidad, un token en un comentario de codigo diluido por cientos de otros tokens no genera senal suficiente. BM25 necesita densidad de keywords, no solo presencia.

## Calibracion del predictor

| Prediccion | Esperado | Real | Error |
|------------|----------|------|-------|
| v2.1 retrieval | 9/21 | 11/21 | +2 |
| v3 retrieval | 11/21 | 7/21 | **-4** |
| v3 utilidad | 14/21 | 11/21 | **-3** |
| v4.0 acuerdo | 75% | 56% | **-19pp** |
| v4.1 acuerdo | 70% | 59% | **-11pp** |
| **v4.2 retrieval** | **7/21** | **7/21** | **0** |

La correccion funciono: el sesgo optimista desaparecio cuando el prior cambio de "el cambio mejora" a "el cambio no cambia nada" (null hypothesis). El punto central de 7/21 fue elegido explicitamente como null hypothesis, no como prediccion optimista con correccion. La metodologia correcta para corregir sesgo optimista es adoptar el null hypothesis como default, no restar puntos mecanicamente.

La composicion sigue estando parcialmente equivocada: predije Q13 como 35% de rescate (se rescato) y 0 losses (hubo 2). Q8 como gain y Q2/Q14 como losses no fueron anticipadas. El modelo mental query-por-query necesita mas trabajo.

## Validacion cruzada del juez

Acuerdo humano vs GPT-4o-mini sobre los datos de v4.2 (chunks hibridos, datos nuevos para el juez):

| | Humano | Juez | Acuerdo |
|---|---|---|---|
| Buenos | 7/21 | 5/21 | — |
| Tangenciales | 0/21 | 4/21 | — |
| Malos | 14/21 | 12/21 | — |
| **Acuerdo total** | | | **17/21 (81%)** |

Salto de 10/21 (48%) en v4.1 a 17/21 (81%). Pero no es comparable directo: el humano cambio de criterio (0 tangenciales en v4.2 vs 8 en v3b). Los 4 desacuerdos son todos el juez usando tangencial donde el humano fue binario:

- Q9, Q15: humano=bueno, juez=tangencial
- Q17, Q20: humano=malo, juez=tangencial

Si colapsas tangencial→malo: acuerdo sube a 19/21 (90%). Solo Q9 y Q15 quedan en desacuerdo.

**Hallazgo principal:** el juez generaliza a datos nuevos. Las 4 queries que cambiaron de chunk (Q2, Q8, Q13, Q14) tienen acuerdo perfecto humano-juez. El juez no se confunde con chunks que no vio antes. Esto sugiere que el acuerdo bajo de v4.1 (48%) era mas un problema de la categoria tangencial que de la capacidad del juez.

**Implicacion para el roadmap:** escala binaria (bueno/malo) para retrieval da 90%+ de acuerdo. La categoria tangencial era la fuente principal de desacuerdo humano-juez Y de varianza intra-evaluador. Eliminarla simplifica el eval sin perder senal.

## Decision

**BGE puro queda como definitivo.** Hybrid no se sube al producto.

Razonamiento:
1. Net = 0. No hay ganancia neta de retrieval.
2. Los 2 gains (Q8, Q13) son reales pero se cancelan con 2 losses (Q2, Q14).
3. Hybrid agrega complejidad (indice BM25, tokenizer, RRF) sin beneficio neto.
4. Los gains son fragiles — Q13 gano por margen de 0.000015.

El resultado confirma la regla de decision pre-comprometida en prediction.md: "7-8/21 = hybrid es un wash, BGE puro queda como definitivo."

## Cierre de Fase 2

v4.2 cierra el ultimo frente abierto de retrieval. El recorrido completo:

| Version | Retrieval bueno | Que se probo |
|---------|----------------|-------------|
| v1 | 3/21 | MiniLM baseline |
| v2 | 2/21 | Hybrid MiniLM + BM25 (fallo) |
| v2.1 | 11/21 (laxo) / 7/21 (estricto) | BGE-base (modelo como cuello de botella) |
| v3 | 7/21 | BGE-base en pipeline completo |
| **v4.2** | **7/21** | **Hybrid BGE + BM25 (wash)** |

El retrieval esta en 7/21 con criterio estricto. Para subirlo habria que cambiar el modelo denso (BGE-large, GTE, fine-tuning) o el chunking. Esas no son prioridades del roadmap actual.

Proxima fase: Agents (v5).
