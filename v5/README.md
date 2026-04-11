# v5.0 — Re-busqueda con query reformulada

## Que intente

Agregar un agente de re-busqueda al pipeline RAG: cuando la respuesta contiene "I don't know", reformular la query con un LLM y buscar de nuevo.

## Que predije

Pre-commit (hash pendiente):
- **Grupo B:** recupero 4 de 6 (intuicion 5, calibracion historica 5-1=4). Composicion: 5 verdes, 1 roja.
- **Grupo C:** 1 canario roto de 5 (Q6). Rango: 0-2.

## Que paso

### Trigger

7 de 21 queries activaron retry: Q4, Q5, Q6, Q8, Q11, Q17, Q20.

Problemas inmediatos:
- **3 queries del Grupo B no activaron retry** (Q12, Q15, Q18). Sus respuestas eran invenciones o respuestas vacias — no contenian "I don't know". El detector era ciego a estos modos de fallo.
- **4 de 5 canarios activaron retry** (Q4, Q5, Q6, Q17). Todos decian "I don't know" honestamente — que es el comportamiento correcto para OOD/opinion. La re-busqueda solo podia empeorarlos.

### Resultados vs v3b

| Eje | v3b | v5.0 | Delta |
|-----|-----|------|-------|
| Retrieval bueno | 7/21 | 7/21 | 0 |
| Generation fiel | 14/21 | 9/21 | **-5** |
| Generation inventado | 2/21 | 2/21 | 0 |
| Utility util | 11/21 | 9/21 | **-2** |
| Utility danino | 2/21 | 3/21 | **+1** |

v5.0 empeoro el sistema en los dos ejes que importan: generation y utilidad.

### Canarios

1 canario roto: **Q17** (pytorch vs tensorflow). La re-busqueda encontro un chunk mas descriptivo sobre PyTorch (main components) y el LLM elaboro una comparacion inventada en vez de rechazar. Paso de util (rechazo honesto) a parcial (comparacion superficial con claims inventados sobre TensorFlow).

Q16 (batch insert DB) invento SQLAlchemy sin activar retry — eso es comportamiento preexistente de v3b, no regresion de v5.0. No cuenta como canario roto.

Q4, Q5, Q6 sobrevivieron: los chunks nuevos eran lo suficientemente irrelevantes como para que el LLM siguiera rechazando.

### Prediccion vs resultado

| Metrica | Prediccion | Real | Error |
|---------|-----------|------|-------|
| Grupo B recuperadas | 4/6 | **0/6** | -4 (optimista) |
| Canarios rotos | 1/5 (Q6) | 1/5 (Q17) | Numero correcto, query incorrecta |

**Recuperacion neta del Grupo B: 0/6.** No 3, no 2 — cero. Las 3 queries que activaron retry (Q8, Q11, Q20) terminaron peor o igual. Las 3 que no activaron (Q12, Q15, Q18) siguieron rotas. El mecanismo es 0-for-6 en su caso de uso declarado.

La prediccion de canarios acerto la magnitud (1 de 5) pero fallo el mecanismo: predije Q6 (optimizer GANs) como el mas vulnerable, rompio Q17 (pytorch vs tf). Mismo patron que v4 — calibracion en magnitud, error en mecanismo. El modelo mental de "cual canario es vulnerable" no esta bien calibrado.

## Por que fallo

El detector buscaba "I don't know" pero el modo de fallo dominante del LLM de 3B en v3b era invencion silenciosa, no rechazo.

De las 6 queries del Grupo B:
- **Q8, Q11, Q20** — decian "I don't know" y activaron retry. Pero los chunks nuevos eran tangenciales (AMP+DDP, meta device Linear, mismo tensor_view.rst). El LLM genero respuestas mixtas con los chunks nuevos en vez de rechazar con los viejos. **La re-busqueda convirtio rechazos honestos en mezclas.**
- **Q12, Q15, Q18** — inventaban o daban respuestas vacias sin decir "I don't know". El trigger nunca se activo. **La re-busqueda no llego a intervenir.**

El mecanismo de daño en las queries retried: los chunks nuevos de la re-busqueda eran mas ambiguos que los originales (tematicamente mas cercanos pero sin responder directamente). El LLM de 3B, ante material ambiguo, mezcla en vez de rechazar — incluso con el prompt mejorado de v3b. Eso bajo generation fiel de 14 a 9. Los 5 puntos perdidos son queries donde v3b decia "I don't know" honestamente y v5.0 mezcla.

## Que aprendi de agentes

Las herramientas de agente no son mejoras incondicionales — son sensibles a donde esta el cuello de botella actual. v5.0 diseno una intervencion para el cuello de botella de v3 (retrieval malo). Pero entre v3a y v3b, el prompt mejorado ya habia movido el cuello de botella a generation: el LLM ahora rechazaba honestamente en vez de inventar, y la utilidad estaba limitada por la calidad del modelo, no por la calidad de los chunks.

Intervenir en la capa equivocada es regresion activa, no neutralidad. La re-busqueda le dio al LLM material nuevo para mezclar, y el LLM uso ese material para producir respuestas peores que los rechazos honestos de v3b. El sistema no quedo igual — empeoro.

Esto es general: cada vez que una intervencion "no toca" el cuello de botella actual, en realidad si lo toca, porque cambia el contexto en el que opera el componente critico. Neutralidad es la excepcion, no la regla.

## Por que no lo arreglo

Podria agregar un detector de empty al trigger, o un clasificador de groundedness, o un filtro OOD para los canarios. Pero estaria construyendo complejidad sobre un mecanismo que demostro empeorar el sistema en los casos que si capturaba. Las 3 queries que activaron retry y eran del Grupo B (Q8, Q11, Q20) todas empeoraron o quedaron igual. El problema no es solo el trigger — es que la re-busqueda con chunks tangenciales + LLM de 3B produce mezclas, no mejoras.

La decision honesta es revertir: volver a v3b como baseline y atacar el cuello de botella real, que es capacidad del modelo. v5.1 prueba llama3.1:8b con el pipeline de v3b sin ningun mecanismo de retry.

## Archivos

```
v5/
  prediction.md              — Prediccion pre-comprometida (Grupo B + canarios)
  search_agent.py            — Pipeline v5.0 con re-busqueda
  pipeline_results_v5.jsonl  — Resultados de las 21 queries
  classify.py                — Clasificacion ciega (agrega categoria 'empty')
  labels_retrieval_v5.csv    — Clasificacion retrieval v5.0
  labels_generation_v5.csv   — Clasificacion generation v5.0 (f/e/x/i)
  labels_utility_v5.csv      — Clasificacion utilidad v5.0
  manual_trace.md            — Trazas manuales de razonamiento experto
  notes.md                   — Notas de analisis detalladas
  README.md                  — Este archivo
```

## Siguiente paso

v5.1: reemplazar llama3.2:3b por llama3.1:8b en generation. Todo lo demas identico a v3b. Cero retry, cero agentes, cero capas nuevas. La pregunta: "¿el cuello de botella es capacidad del modelo, o es algo estructural?"
