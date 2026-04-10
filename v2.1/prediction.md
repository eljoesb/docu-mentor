# Predicción: BGE-base-en-v1.5 retrieval

**Fecha:** 2026-04-09

## Hipótesis global del proyecto

El cuello de botella de v2 era capacidad semántica del modelo de embeddings. Esta hipótesis se considera **confirmada** si algún experimento del camino 3 alcanza ≥10/21 retrieval bueno. Si ninguno llega, la hipótesis se considera **refutada** y el cuello de botella está en otro lado (chunking, cobertura del corpus, o reformulación de queries).

**Rol de BGE-base:** es el experimento más barato del camino 3. Paso de diagnóstico, no definitivo.

- Si BGE-base ≥10/21 → hipótesis confirmada, no hace falta BGE-large.
- Si BGE-base entre 5 y 9 → hipótesis parcialmente sostenida, escalar a BGE-large es justificable.
- Si BGE-base ≤4 → el tamaño del modelo no mueve la aguja, camino 3 muerto, ir a camino 2.

## Método de predicción

Uso desagregación por confianza porque mi razonamiento por query es heterogéneo — algunas tienen chunk candidato identificado, otras no. Predecir un número monolítico ocultaría esa heterogeneidad.

## Queries descartadas de entrada

**Estructuralmente imposibles (OOD/opinion):** Q4, Q5, Q6, Q16, Q17 — ningún chunk del corpus puede responderlas. 5 queries fuera.

**Gap de corpus, no de modelo:** Q7 (memorizing → overfitting no existe como concepto en los docs), Q10 (.size vs .shape → dato de una línea enterrado), Q18 (weights not changing → síntoma sin match terminológico). 3 queries fuera.

Total descartadas: 8. Candidatas reales: 21 − 8 = 13 (incluyendo las 4 del baseline).

## Baseline heredado: 4 buenos

Q1 (autograd), Q9 (leaf tensor), Q19 (loss NaN), Q21 (LR scheduling).

Hipótesis: BGE-base los retiene. Si pierde alguno, es señal muy mala — significa que BGE es peor que MiniLM en queries donde MiniLM ya acertaba.

## Alta confianza — flipean casi seguro (3 queries)

**Q13 (numpy to tensor)** → tensors.rst con `torch.tensor(np.array([[1,2,3],[4,5,6]]))`. Hybrid ya lo encontró. BGE con mejor espacio semántico debería conectar "convert numpy" → "torch.tensor(np.array)" directamente. Asterisco: hybrid lo encontró por serendipia de RRF (posición 14 en MiniLM, posición 2 en BM25). BGE podría mover tensors.rst de posición 14 a posición 3 pero seguir sin ser top-1. Si no flipea, el razonamiento no es "BGE es malo" sino "BGE no sube suficiente el rank para desplazar al ganador actual".

**Q20 (view vs reshape)** → tensor_view.rst con "reshape can return either a view or new tensor". BM25-smart ya lo encontraba. BGE debería encontrarlo también — la semántica "view vs reshape" es directa y el chunk contiene ambas palabras clave.

**Q2 (move tensor to GPU)** → el patrón `.to('cuda')` / `.to(device)` es pervasivo en el corpus (cuda.rst, mps.rst, múltiples tutoriales). BGE con prefix asimétrico debería conectar "move" como intención de búsqueda con "device transfer". MiniLM fue a tensor_attributes.rst (creación en device, no movimiento) — BGE debería discriminar mejor.

## Confianza media — probables pero no garantizadas (3 queries)

**Q12 (save/load model)** → serialization.rst abre con "how you can save and load PyTorch tensors and module states". BM25-dumb lo encontraba directamente. MiniLM fue a hub.md (tangencial). BGE debería discriminar mejor "save/load model" de "load from hub" — pero no sé si serialization.rst está suficientemente arriba en el ranking de embeddings como para ganar.

**Q3 (DataLoader workers)** → data.md tiene varios chunks sobre multi-process loading, incluyendo uno que explica qué pasa cuando se crean los workers. MiniLM fue a un chunk de Windows compat. BGE podría encontrar el chunk de mecanismo de workers que es más directamente relevante, pero data.md tiene muchos chunks y no estoy seguro de cuál rankea primero.

**Q15 (custom autograd function)** → extending.func.rst y extending.rst ambos existen y son directamente relevantes. MiniLM fue a la intro general de autograd. BGE con hard negative training podría discriminar "autograd intro" de "extending autograd". Pero el chunk de extending.rst que encontraría probablemente introduce el tema sin mostrar forward/backward concreto — podría ser tangencial fuerte, no bueno.

## Apuestas largas — bajo prior (3 queries)

**Q14 (flatten before linear)** → nn.rst lista Flatten/Unflatten en autosummary. named_tensor.md menciona flatten. Pero listar un API name no es responder la pregunta bajo criterio estricto. Hipótesis alternativa: gap de corpus — no hay un chunk que explique el patrón CNN flatten→linear.

**Q11 (F.linear)** → extending.rst menciona Linear. fx.md tiene nn.Linear(512, 512). Pero la descripción real de F.linear (y = xA^T + b) probablemente no existe como párrafo independiente. Hipótesis alternativa: gap de corpus disfrazado de gap de modelo.

**Q8 (multiple GPUs)** → cuda.rst habla de device management, no de DataParallel/DDP. Para ser bueno necesitaría encontrar un chunk sobre entrenamiento distribuido, no solo selección de device. Hipótesis alternativa: el corpus no tiene un chunk que responda "how to use multiple GPUs" en el sentido de paralelismo.

## Predicción numérica

4 (baseline) + 3 (alta confianza) + 2 de 3 (media) + 0 de 3 (apuestas largas) = **9/21**

Rango: 7-8 si fallan las de confianza media, 9 si sale como espero, 10-11 si tengo suerte con las apuestas largas.

Mediana: **9**.

## Regla de decisión post-experimento

- **≥10 buenos** → camino 3 confirmado. El modelo de embeddings era el cuello de botella. Fin del camino 3, evaluar si el resultado es suficiente o si vale refinar.
- **7-9 buenos** → camino 3 vivo. BGE-base más que duplicó el baseline (de 4 a 7-9). Escalar a BGE-large inmediatamente — hay margen real para romper los últimos puntos.
- **5-6 buenos** → camino 3 con poco upside. La mejora es real pero insuficiente. BGE-large probablemente no llega a 10. Pivotar a camino 2 (chunking más granular).
- **≤4 buenos** → camino 3 muerto. Cambiar el modelo no mueve la aguja. El problema es corpus/chunking. Ir a camino 2 sin pasar por BGE-large.
