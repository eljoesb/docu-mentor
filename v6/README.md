# v6 — Top-K=3 retrieval

## Que intente

Pasar de top-1 a top-3 retrieval. Todo lo demas identico a v5.1: mismo modelo (llama3.1:8b), mismo embedding (BGE-base), mismo corpus. El prompt se adapto al plural (contexts, numbered format "Context 1: ... Context 2: ... Context 3: ..."). K=3 y no K=5 por riesgo de dilucion y porque las queries prometedoras necesitaban K=2-3, no K=5.

La pregunta: "el cuello de botella #1 del techo distribuido (retrieval top-1, 7/21 bueno) se puede atacar con top-K sin pagar un costo neto en generation?"

## Que predije

Pre-commit (v6/prediction.md):

| Query | v5.1 | v6 pred | Confianza | Mecanismo |
|-------|------|---------|-----------|-----------|
| Q8 | p | u | Alta | Inter-file: cuda.rst multi-GPU |
| Q12 | p | u | Alta | Inter-file: serialization.rst |
| Q13 | p | u | Alta | Inter-file: tensors.rst inicio |
| Q20 | p | u | Alta | Intra-file: tensor_view.rst cabeza |
| Q21 | p | u | Media | Intra-file: optim.md schedulers |
| Q15 | p | u o p | Media | Inter-file: extending.rst tutorial |
| Q10 | p | p | Alta | Contenido no existe |
| Q7 | p | p | Alta | Contenido no existe |
| Q14 | p | p | Alta | Markup blindness, top-K no ataca |
| Q11 | p | p | Alta | Contenido no existe en prosa |
| Q18 | p | p o x | Media | Riesgo de mas mezcla |
| Q19 | p | p | Baja | Incierto |

Totales calibrados: 13/21 util, 17/21 fiel, 0 daninos.

Categorica: **VERDE** — primera intervencion que ataca el cuello de botella real.

Condicion de falsacion: "Si 2+ de las 4 de alta confianza fallan, caigo a 11 y la categorica es ROJA."

Riesgo principal documentado: "no verifique los embeddings" — sabia que el contenido existia en el corpus, pero no confirme que BGE lo rankeara en top-3 para cada query.

## Que paso

### Resultados vs v5.1

| Eje | v5.1 | v6 | Delta |
|-----|------|-----|-------|
| Retrieval bueno | 7/21 | 9/21 | **+2** |
| Retrieval tangencial | 8/21 | 7/21 | -1 |
| Retrieval malo | 6/21 | 5/21 | -1 |
| Generation fiel | 19/21 | 20/21 | **+1** |
| Generation mixto | 2/21 | 1/21 | -1 |
| Generation inventado | 0/21 | 0/21 | 0 |
| Generation empty | 0/21 | 0/21 | 0 |
| **Utility util** | **9/21** | **13/21** | **+4** |
| Utility parcial | 12/21 | 8/21 | -4 |
| Utility danino | 0/21 | 0/21 | 0 |

**Perros en la nieve: VERDE confirmada.** 13 util (= 13 necesarias), 0 daninos (< 3 permitidas). Primera version del proyecto que cierra el criterio.

### Las cuatro queries que ganaron utilidad

**Q8 (multiple GPUs), p→u:** Top-1 seguia siendo distributed.md (seccion de funciones deprecadas, tangencial). Posiciones 2-3 trajeron cuda.rst con contenido real: CUDA_VISIBLE_DEVICES, torch.device('cuda:X'), torch.cuda.device context manager. 8b distinguio chunks relevantes de tangenciales, uso solo cuda.rst, e ignoro el chunk de distributed.md sobre Symmetric Memory. Mecanismo predicho: inter-file, contenido en multiples archivos. **Confirmado.**

**Q13 (numpy→tensor), p→u:** Top-1 seguia siendo sparse.rst (irrelevante). Posicion 3 trajo tensors.rst con `torch.tensor(np.array(...))` y el warning que menciona `torch.as_tensor`. 8b descarto Context 1 (sparse) y Context 2 (NumpySort reverse), extrajo `torch.as_tensor` de Context 3. Mecanismo predicho: inter-file, chunk al inicio de tensors.rst con vocabulario directo. **Confirmado.** Es el cierre de la paradoja de v5.1 — Q13 perdio utilidad en v5.1 por honestidad (rechazo de chunk malo); en v6, la honestidad se combina con retrieval bueno y produce utilidad real.

**Q20 (view vs reshape), p→u:** Las 3 posiciones trajeron chunks de tensor_view.rst — el mismo archivo pero diferentes secciones. La cola (listado de metodos, top-1 en v5.1) mas la cabeza (semantica de views, contiguity, storage compartido). 8b sintetizo correctamente: views comparten datos subyacentes sin copiar, reshape puede copiar. Mecanismo predicho: intra-file, chunks de la cabeza del mismo archivo. **Confirmado.**

**Q10 (.size vs .shape), p→u:** Los 3 chunks (export.md, dynamic_shapes.md x2) usan `.size()` y `.shape` indistintamente en codigo sobre dynamic shapes. 8b dedujo que son equivalentes: "I can tell that x.size()[0] and x.shape[0] refer to the same thing." La respuesta es correcta — .size() y .shape son funcionalmente identicos en PyTorch. **No predicho.** La prediccion decia "el contenido no existe" (alta confianza), y era cierto: ningún chunk explica la diferencia. Pero el modelo infirió la respuesta desde uso contextual implicito. Modo de mejora no anticipado.

### Mecanismos confirmados

**Top-K ataca retrieval como predicho.** De 4 queries de alta confianza, 3 mejoraron por los mecanismos exactos descritos: inter-file (Q8 desde cuda.rst, Q13 desde tensors.rst) e intra-file (Q20 desde tensor_view.rst). La tesis era que el contenido correcto existia en el corpus y rankearia top-3 — para estas 3 queries, ambas condiciones se cumplieron.

**Formato numerado funciona.** 8b referencio chunks por numero ("Based on Context 2 and Context 3...") y distinguio chunks relevantes de tangenciales en Q8, Q13, Q14. El riesgo de "sopa de contexto" donde los chunks se fusionan no se materializo. La numeracion cumple su objetivo: señalar que los chunks son piezas distintas de informacion.

**La mezcla multi-chunk fue over-predicted.** La prediccion esperaba 17-18 fiel (19→17-18) por riesgo de mezcla. El resultado fue 20 fiel (19→20). Solo una regresion (Q19) y dos mejoras (Q15, Q18 ambos x→f). La coherencia tematica de los 3 chunks de la misma busqueda — vs. la incoherencia de la re-busqueda en v5.0 — explica la diferencia. Chunks del mismo ranking tienen mas coherencia que chunks de queries reformuladas.

### Mecanismo refutado: Q12

**Q12 (save/load model), p→p:** La prediccion decia "alta confianza, serialization.rst tiene save/load/state_dict/checkpoint y deberia rankear top-3." Los 3 chunks que llegaron: aot_inductor.md (posiciones 1 y 3) y autograd.rst (posicion 2). serialization.rst no rankeo top-3 para esta query.

La leccion de calibracion: **confianza alta requiere verificar tanto contenido como embedding, no solo contenido.** Confirme que serialization.rst existia y tenia el contenido correcto, pero no verifique que BGE producia un similarity score suficiente para "how to save and load a model for inference" → serialization.rst. El riesgo estaba documentado en la prediccion ("no verifique los embeddings"), pero la confianza era "alta" cuando debio ser "media." La diferencia entre "el contenido existe" y "el contenido rankea" es exactamente el tipo de paso que un predictor confiado omite y un predictor calibrado verifica.

### Queries persistentes: modos de fallo que top-K no ataca

**Q21 (learning rate), p→p — literalismo semantico persiste.** 3 chunks de optim.md, incluyendo seccion "How to adjust learning rate" con ExponentialLR. 8b rechaza: "none of them explicitly describe updating the learning rate in the middle of training, only at the end of each epoch or using specific schedulers like SWALR." El contenido esta; el modelo lee "middle of training" como "intra-epoch" cuando el usuario pregunta "during training." Prediccion anticipaba este riesgo ("confianza media") y el mecanismo especifico (literalismo semantico) es el mismo de v5.1. Esto confirma que literalismo semantico es independiente del retrieval — no se arregla trayendo mas chunks del mismo tipo. Requiere reformulacion de query o intervencion en el prompt.

**Q14 (flatten), p→p — markup blindness parcialmente resuelta.** En v5.1, 8b dijo "the context does not mention anything about flattening" cuando el chunk decia `{meth}~Tensor.flatten`. En v6, 8b dice "Context 1 and Context 2 discuss tensor manipulation methods like permute, align_to, flatten, and unflatten." El modelo AHORA VE flatten a traves del markup Sphinx. Pero la utilidad sigue parcial: el chunk menciona que flatten existe pero no explica como usarlo antes de una linear layer. Generation mejoro (ve la mencion), utility no (la mencion no es suficiente). El problema migro de markup blindness a falta de contenido how-to.

**Q7, Q11, Q18, Q19 — el contenido no existe o el retrieval falla.** Q7 y Q11 siguen sin contenido accesible en el corpus (regularizacion, F.linear). Q18 sigue con chunks tangenciales de optim.md. Q19 regresiono (ver abajo).

## Hallazgos emergentes

### Transferencia inferencial intra-contexto (Q10)

Q10 mejoro sin que ningun chunk explicara la diferencia entre .size() y .shape. El modelo vio ambos usados intercambiablemente en chunks sobre dynamic shapes y dedujo equivalencia. Esto es un modo de exito que el techo distribuido no contemplaba: **multiples chunks debiles producen una respuesta correcta por triangulacion.**

La taxonomia de v5.1 tenia tres modos de fallo (paradoja honestidad-suerte, markup blindness, literalismo semantico). v6 añade una contraparte: un modo de exito donde la respuesta no viene de un chunk sino de la relacion entre chunks. La condicion para que funcione: los chunks tienen que estar en el mismo vocabulario que la query. Q10 funciono porque "size" y "shape" aparecen literalmente en los chunks. Q7 no funciona porque "memorizing training data" no comparte vocabulario con los chunks sobre dropout/BatchNorm/requires_grad (ver abajo).

### Cross-vocabulary transfer failure (Q7 — cuarta capa del techo)

Q7 fallo en v1, v3a, v3b, v5.1, y v6 — cinco versiones, mismo modo de fallo. Top-K trajo faq.rst y autograd.rst, ninguno menciona "memorizing" u "overfitting." Pero el corpus SI tiene contenido tangencialmente relevante: dropout (en nn.rst), BatchNorm (en nn.rst), model.train()/model.eval() (en autograd.rst). Ningun chunk menciona "memorizing" o "overfitting" literalmente, y 8b no hace la conexion entre el vocabulario coloquial de la query y el vocabulario tecnico de los chunks.

Esto es lo opuesto a Q10: Q10 funciono porque la query y los chunks comparten vocabulario (".size", ".shape"); Q7 falla porque no lo comparten ("memorizing training data" vs "dropout", "regularization"). La inferencia contextual tiene un requisito: **solapamiento lexico entre query y chunks.** Cuando el solapamiento no existe, ni top-K ni modelos mas grandes resuelven el problema — se necesita reformulacion de la query al vocabulario del corpus, o clarificacion al usuario.

Q7 revela una cuarta capa del techo distribuido que v5.1 no nombro: **jargon mismatch / cross-vocabulary transfer failure.** No es exactamente literalismo semantico (donde el modelo ve el contenido pero lo interpreta demasiado estrecho, como Q21). Es que el modelo nunca ve el contenido relevante porque el retrieval no puede conectar vocabularios distintos. El problema esta antes del modelo — en la interfaz query→embedding→ranking.

### Especulacion post-rechazo (Q19 — regresion)

Q19 es la unica regresion de generation (f→x). En v5.1 con un chunk (faq.rst sobre accumulating history), 8b rechazo limpiamente: "I don't know." En v6 con tres chunks (faq.rst, cuda.rst, named_tensor.md — todos tangenciales), 8b dice "I don't know based on the provided context" y DESPUES añade: "However, if I had to make an educated guess, it could be related to: division by zero, gradient explosion."

Esos guesses son conocimiento externo, no del contexto. El prompt dice "Use ONLY the following contexts." El modelo viola la restriccion despues de haber reconocido que no puede responder.

El patron: **multi-chunk emboldece la especulacion post-rechazo.** Con un chunk, el modelo tiene una decision binaria clara (usar o rechazar). Con tres chunks tangenciales, el modelo acumula material "cercano" que lo empuja a especular despues de rechazar, como si la proximidad tematica de los chunks le diera permiso para ir mas alla. Es el inverso de la paradoja de v5.1: mas contexto tangencial empeora la generation en vez de mejorarla.

La buena noticia: solo 1 de 21 queries mostro este patron. La mala: no es predecible que queries lo activan — depende de la combinacion especifica de chunks tangenciales y del "atractivo" del tema (NaN loss es un tema donde la tentacion de dar consejos genericos es alta).

## Prediccion vs resultado

| Metrica | Prediccion | Real | Error |
|---------|-----------|------|-------|
| Util total (calibrada) | 13/21 | 13/21 | **0** |
| Generation fiel (calibrada) | 17/21 | 20/21 | +3 (pesimista) |
| Generation mixto (calibrada) | 4/21 | 1/21 | -3 (pesimista) |
| Daninos | 0/21 | 0/21 | 0 |
| Categorica (VERDE) | VERDE | VERDE | **Correcta** |
| Alta confianza | 4/4 | 3/4 | -1 (Q12 fallo) |
| Condicion de falsacion | no activada | no activada | Correcta |

La utilidad calibrada acerto el numero exacto, pero por compensacion: Q12 no mejoro (predicha como alta confianza), Q10 mejoro (predicha como "contenido no existe"). El predictor acerto el total por las razones parcialmente incorrectas. Eso es mejor que en v3 (total incorrecto por razones desconocidas) pero peor que lo ideal (total correcto por las razones correctas).

La generation fue significativamente mejor de lo predicho. La prediccion esperaba degradacion (19→17) por mezcla multi-chunk. La realidad fue mejora (19→20). El predictor sobreestimo el riesgo de dilucion. La leccion de v5.0 ("mas contexto ambiguo empeora generation") no aplico a v6 porque los chunks de top-K tienen coherencia tematica que los de re-search no tenian.

## Latency

| Queries 10-15 | t_ret promedio | t_llm promedio | t_total promedio |
|----------------|---------------|----------------|-----------------|
| v6 | 0.22s | 7.41s | 7.63s |
| v5.1 | 0.08s | 3.0s | 3.1s |

~2.5x v5.1. El prompt es ~3x mas largo (3 chunks vs 1), lo que explica el aumento en t_llm. Retrieval sigue negligible. Latency no es un problema para el eval, pero para produccion seria un factor.

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
| v5.1 categorica | ROJA | ROJA | Correcta |
| v5.1 gen direction | sube | sube | Correcta |
| v5.1 util direction | baja | baja | Correcta |
| v5.1 util magnitud | 11-12 | 9 | -2 a -3 (optimista) |
| **v6 categorica** | **VERDE** | **VERDE** | **Correcta** |
| **v6 util calibrada** | **13** | **13** | **0** |
| **v6 gen calibrada** | **17** | **20** | **+3 (pesimista)** |
| v6 alta confianza | 4/4 | 3/4 | -1 |

El patron historico cambio: por primera vez, el predictor fue pesimista en generation (esperaba degradacion, hubo mejora). La explicacion: aplico la leccion de v5.0 ("mas contexto empeora generation") a v6, pero la leccion no transferia porque los mecanismos eran diferentes (re-search incoherente vs. top-K coherente).

La prediccion de utilidad, históricamente optimista por 2-3 puntos, acerto exacto por primera vez. La decision de ajustar -1 en vez de -2 ("porque el mecanismo cambio") resulto ser la correccion correcta. Pero el acierto incluyo compensacion (Q12 fallo, Q10 gano) — no fue precision pura del modelo causal.

## Cierre de Fase 2

v6 cierra perros en la nieve con numeros: 13/21 util, 0/21 daninos.

La trayectoria de Fase 2:

| Version | Util | Daninos | Fiel | Intervención |
|---------|------|---------|------|-------------|
| v3b | 11/21 | 2/21 | 13/21 | Baseline (3b, top-1) |
| v5.0 | 9/21 | 3/21 | 9/21 | Re-search (fallo) |
| v5.1 | 9/21 | 0/21 | 19/21 | Model swap (3b→8b) |
| **v6** | **13/21** | **0/21** | **20/21** | **Top-K=3** |

Cada version enseno algo distinto:
- v5.0 enseno que mas busquedas ≠ mejor retrieval (migracion de cuello de botella)
- v5.1 enseno que mejor modelo ≠ mejor utilidad (paradoja honestidad-suerte) y revelo el techo distribuido
- v6 enseno que atacar el cuello de botella real produce ganancia real, y revelo modos de exito (inferencia contextual) y fallo (cross-vocabulary) no anticipados

El techo distribuido de v5.1 tenia tres capas: (1) retrieval top-1, (2) markup blindness, (3) literalismo semantico. v6 ataco la capa 1 y bajo la presion lo suficiente para cerrar el criterio. Pero tambien revelo una cuarta capa: **cross-vocabulary transfer failure** — el retrieval no puede conectar vocabulario coloquial de queries con vocabulario tecnico de chunks, y el modelo no compensa esa brecha.

Las 8 queries parciales restantes se dividen en:

| Causa | Queries | Se ataca con |
|-------|---------|-------------|
| Contenido no existe en corpus | Q7, Q11 | Expansion del corpus (fuera de scope) |
| Retrieval miss (embedding) | Q12 | Re-ranking, query expansion, o mejor chunking |
| Literalismo semantico | Q21 | Reformulacion de query o clarificacion al usuario |
| Cross-vocabulary mismatch | Q7 | Clarificacion al usuario |
| Chunks tangenciales insuficientes | Q18, Q19 | Mejor retrieval o corpus |
| Mencion sin how-to | Q14, Q15 | Preprocesamiento de markup + mejor chunking |

Notar que Q7 aparece en dos categorias: el contenido sobre regularizacion no existe explicitamente en el corpus, Y la query usa vocabulario que no matchea el corpus. Ambas causas aplican.

## Implicacion para Fase 3

Fase 2 pregunto: "que componente del pipeline es el cuello de botella?" La respuesta, construida iterativamente de v5.0 a v6: **el cuello de botella es distribuido** — retrieval, markup, literalismo, cross-vocabulary. Ninguna intervencion single-axis resuelve todo, pero atacar el eje mas presionado (retrieval top-1→top-3) produjo la mayor ganancia del proyecto (+4 util).

v6 abrio tres hilos que no son Fase 3 pero son material para v6.1:
1. **Inferencia contextual (Q10):** ¿repetible o ruido? Si es sistematico, implica que top-K tiene un modo de mejora oculto donde chunks debiles se refuerzan mutuamente.
2. **Cross-vocabulary (Q7):** ¿cuarta capa del techo o subclase de literalismo semantico? El tratamiento es el mismo (reformulacion/clarificacion), pero el diagnostico importa.
3. **Q12 retrieval miss:** ¿falla de embeddings o de chunking? Si serialization.rst produce un embedding lejano a "save and load model," es un problema de embedding. Si el chunk relevante esta enterrado en un chunk largo, es un problema de chunking. La distincion determina la intervencion.

Estos tres hilos son retornos decrecientes sobre Fase 2. Fase 3 — agentes con clarificacion al usuario — ataca un eje diferente: en vez de optimizar que chunks trae el sistema, preguntar al usuario que quiere antes de buscar. El trace manual de Q3 y Q18 en fases anteriores mostro que la clarificacion era la herramienta mas prometedora. v6 confirma: las 8 queries parciales restantes necesitan o mejor contenido (fuera de scope) o mejor comprension de la intencion del usuario (Fase 3).

## Archivos

```
v6/
  prediction.md              — Prediccion pre-comprometida (per-query + categorica)
  measure_full_pipeline.py   — Pipeline v6 (top-K=3, formato numerado)
  pipeline_results_v6.jsonl  — Resultados de las 21 queries
  labels_retrieval_v6.csv    — Clasificacion retrieval v6 (top-3 evaluado)
  labels_generation_v6.csv   — Clasificacion generation v6 (f/e/x/i)
  labels_utility_v6.csv      — Clasificacion utilidad v6
  meta_notes.md              — Progresion de la metodologia de prediccion v3→v6
  README.md                  — Este archivo
```
