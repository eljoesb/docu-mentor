# v5.1 — Model swap: llama3.2:3b → llama3.1:8b

## Que intente

Reemplazar el modelo de generation (llama3.2:3b → llama3.1:8b) sin tocar nada mas. Mismo retrieval (BGE-base, top-1), mismo prompt (v3b), mismo corpus, cero retry. La pregunta: "el cuello de botella es capacidad del modelo, o es algo estructural?"

## Que predije

Pre-commit (v5_1/prediction_v5_1.md):

| Bucket | Calibrada |
|--------|-----------|
| v3b mixto → fiel (6) | 5/6 |
| v3b inventado → mejora (2) | 1/2 |
| v3b fiel se mantiene (13) | 12/13 |

Categorica: **ROJA** — v5.1 no cierra perros en la nieve.

Tesis de la prediccion: "en un sistema RAG con retrieval imperfecto, mejorar la obediencia del modelo al prompt reduce invenciones Y aciertos por suerte simultaneamente; la utilidad neta depende de cual de los dos efectos domina."

## Que paso

### Resultados vs v3b

| Eje | v3b | v5.1 | Delta |
|-----|-----|------|-------|
| Retrieval bueno | 7/21 | 7/21 | 0 |
| Generation fiel | 13/21 | 19/21 | **+6** |
| Generation mixto | 6/21 | 2/21 | **-4** |
| Generation inventado | 2/21 | 0/21 | **-2** |
| Utility util | 11/21 | 9/21 | **-2** |
| Utility danino | 2/21 | 0/21 | **-2** |

Generation mejoro brutalmente: +6 fiel, 0 inventado, 0 empty. El 8b es dramaticamente mas honesto que el 3b.

Utility empeoro: -2 util (11→9). Cero queries ganaron utilidad. Dos la perdieron. Los daninos bajaron a 0 — el sistema es mas seguro, pero menos util.

Perros en la nieve: **ROJA confirmada.** 9 util (necesitaba 13), 0 daninos (pasa). El eje de daninos cierra; el eje de utilidad esta mas lejos que antes.

### Las dos queries que perdieron utilidad

**Q13 (numpy→tensor):** En v3b, 3b invento `torch.tensor()` (correcto) desde un chunk de sparse.rst (irrelevante). En v5.1, 8b ve que el chunk es sobre sparse tensors y rechaza honestamente. Generation mejora (mixto→fiel), utilidad baja (util→parcial). **Este es el caso canonico de la paradoja:** la invencion era correcta por suerte, y la honestidad la destruye.

**Q14 (flatten before linear):** En v3b, 3b invento un ejemplo de `tensor.flatten(start_dim=1)` (correcto) desde un chunk que mencionaba flatten. En v5.1, 8b dice "the context does not mention anything about flattening" — **pero el chunk dice literalmente "Use Tensor.flatten and Tensor.unflatten."** 8b no vio la mencion porque esta envuelta en markup Sphinx: `{meth}~Tensor.flatten`. Generation mejora (mixto→fiel), utilidad baja (util→parcial). **Esto no es la paradoja — es markup blindness.**

### Tres modos de fallo distintos bajo el mismo sintoma

Las 12 queries con utilidad=parcial en v5.1 son "I don't know" o rechazos parciales. Pero tienen tres causas distintas:

| Modo | Ejemplo | Causa raiz | Se arregla con |
|------|---------|-----------|----------------|
| Paradoja de honestidad | Q13 (numpy→tensor) | Retrieval malo + modelo honesto = pierde acierto por suerte | Mejor retrieval |
| Markup blindness | Q14 (flatten) | El modelo no parsea `{meth}~Tensor.flatten` como mencion funcional | Preprocesamiento del corpus |
| Literalismo semantico | Q21 (learning rate) | 8b interpreta "middle of training" como "middle of epoch" en vez de "during training" | Reformulacion de query o top-K chunks |

Tambien hay dos queries donde 8b mezclo (Q15, Q18) — un modo de fallo distinto donde el modelo mas capaz "conecta puntos" y cruza la linea a mixto. Q18 es regresion directa: era fiel en v3b, ahora es mixto. Pero solo 2 de 21 — el efecto dominante es honestidad excesiva, no mezcla excesiva.

### Prediccion vs resultado

| Metrica | Prediccion | Real | Error |
|---------|-----------|------|-------|
| Mixto → fiel | 5/6 | 4/6 | -1 |
| Inventado → mejora | 1/2 | 2/2 | +1 |
| Fiel se mantiene | 12/13 | 12/13 | 0 |
| Util total | 11-12 | 9 | -2 a -3 (optimista) |
| Daninos | 0-1 | 0 | 0 a -1 |
| Categorica (ROJA) | ROJA | ROJA | **Correcta** |

La direccion fue correcta en todos los ejes: generation sube, utility baja, categorica roja. El error fue de magnitud en utilidad — predije 11-12, dio 9. No identifique Q14 como perdida de utilidad (no vi el markup blindness), ni Q21 como falso negativo (no anticipe el literalismo semantico). Q13 si la identifique como caso canonico de la paradoja antes de medir.

## La paradoja, confirmada

"En un sistema RAG con retrieval imperfecto, mejorar la obediencia del modelo al prompt reduce invenciones Y aciertos por suerte simultaneamente; la utilidad neta depende de cual de los dos efectos domina."

En v5.1 domino la reduccion de aciertos por suerte. El 8b elimino toda invencion (0i, 0e) y convirtio 4 de 6 mixto a fiel. Pero cero queries ganaron utilidad y dos la perdieron. La honestidad del modelo no se traduce en utilidad cuando el retrieval trae chunks malos: un rechazo honesto de un chunk malo es parcial, no util. Para que la honestidad sea util, el chunk tiene que ser bueno — y eso es retrieval, no generation.

## El techo distribuido

El hallazgo central de v5.1 — y la conclusion de Fase 2 — no es "8b mejoro generation pero no utility". Es que **el techo de utilidad de DocuMentor no es capacidad del modelo.** Es una combinacion de tres problemas estructurales:

1. **Retrieval top-1 (7/21 bueno).** La mayoria de queries no encuentran el chunk correcto. Un modelo perfecto con un chunk malo produce "I don't know" perfecto — que es parcial, no util. Q8, Q11, Q12 son ejemplos: chunks tangenciales o irrelevantes donde ningun modelo puede dar una respuesta util sin violar "use ONLY the context."

2. **Markup blindness.** El corpus tiene contenido en Sphinx/MyST con markup como `{meth}~Tensor.flatten`, `:ref:`, `{class}~torch.utils.data.DataLoader`. Los modelos no parsean esto como menciones funcionales. Q14 es el caso canonico: el chunk dice "Use Tensor.flatten" pero 8b no lo ve. El contenido existe; el formato lo esconde.

3. **Literalismo semantico.** El modelo interpreta queries de forma mas estrecha que la intencion del usuario. Q21: "middle of training" = entre epochs, no intra-batch, pero 8b lee "middle of an epoch". Q7 (en v3b): "memorizing the training data" ≠ "accumulate history", pero 3b no distinguia. 8b distingue correctamente en Q7 pero falla en Q21 con un literalismo inverso.

Cambiar el modelo deja dos de los tres intactos. Ir a 13b, 70b, o Claude via API probablemente suba generation de 19 a 20 y utility de 9 a 10. **El camino "modelo mas grande" tiene retornos decrecientes garantizados** porque el cuello de botella real es la interfaz entre retrieval y generation, no la generation en si.

## Latency

| Queries 10-15 | t_llm promedio | t_total promedio |
|----------------|---------------|-----------------|
| Con Q15 (outlier) | 4.32s | 4.40s |
| Sin Q15 | 2.97s | 3.05s |

8b corre a ~3s por query en regimen estacionario. Q1 tardo 6.5s (startup del modelo en Ollama). El pipeline completo tomo ~80 segundos. Latency no es problema.

## Calibracion del predictor

| Version | Prediccion | Real | Error |
|---------|-----------|------|-------|
| v3 retrieval | 11/21 | 7/21 | -4 (optimista) |
| v3 generation | 14/21 | 14/21 | 0 |
| v3 utilidad | 14/21 | 11/21 | -3 (optimista) |
| v4.0 global | 75% | 56% | -19pp (optimista) |
| v4.1 global | 70% | 59% | -11pp (optimista) |
| v5.0 Grupo B | 4/6 | 0/6 | -4 (optimista) |
| v5.0 canarios | 1/5 | 1/5 | 0 (query incorrecta) |
| **v5.1 categorica** | **ROJA** | **ROJA** | **Correcta** |
| v5.1 gen direction | sube | sube | Correcta |
| v5.1 util direction | baja | baja | Correcta |
| v5.1 util magnitud | 11-12 | 9 | -2 a -3 (optimista) |

El patron cambio en v5.1. Por primera vez, el predictor anticipo el **mecanismo** (paradoja honestidad-vs-suerte) antes de medir, no solo los numeros. La direccion fue correcta en todos los ejes. El error sigue siendo de magnitud (optimista), pero el modelo mental del sistema ya es estructuralmente correcto.

Este cambio — de calibrar numeros a calibrar mecanismos — es la meta-leccion de Fase 2. En v3 y v4, las predicciones eran sobre "cuantas queries mejoran" sin un modelo de POR QUE mejorarian o no. En v5.1, la prediccion incluia una tesis causal (la paradoja) que se confirmo. Los numeros estuvieron mal; el mecanismo estuvo bien. Eso es progreso real en la capacidad de razonar sobre el sistema.

## Conclusion de Fase 2

v5.1 cierra la pregunta que abrio Fase 2: "el cuello de botella es capacidad del modelo?"

**No.** El cuello de botella es estructural y esta distribuido. 11/21 util (v3b) y 9/21 util (v5.1) son variaciones dentro del mismo techo — un techo impuesto por retrieval top-1 (7/21 bueno), markup blindness, y literalismo semantico. Ningun swap de modelo rompe ese techo porque el modelo no es el cuello de botella.

Lo que v5.1 SI demostro: el 8b es mas seguro (0 daninos), mas honesto (19 fiel), y mas predecible (rechaza consistentemente en vez de inventar inconsistentemente). Si el objetivo fuera "no mentir nunca", v5.1 es estrictamente mejor que v3b. Pero el objetivo era utilidad, y ahi retrocedio.

## Archivos

```
v5_1/
  prediction_v5_1.md          — Prediccion pre-comprometida (buckets + categorica)
  measure_full_pipeline.py    — Pipeline v5.1 (identico a v3b, solo cambia modelo)
  pipeline_results_v5_1.jsonl — Resultados de las 21 queries
  labels_retrieval_v5_1.csv   — Clasificacion retrieval v5.1
  labels_generation_v5_1.csv  — Clasificacion generation v5.1 (f/e/x/i)
  labels_utility_v5_1.csv     — Clasificacion utilidad v5.1
  README.md                   — Este archivo
```
