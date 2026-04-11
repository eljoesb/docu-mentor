# v6 — Prediccion pre-comprometida

**Fecha:** 2026-04-11

## Cambio unico

Top-K retrieval con K=3 en vez de top-1. Todo lo demas identico a v5.1: mismo modelo (llama3.1:8b), mismo prompt (adaptado a plural), mismo corpus, mismo embedding (BGE-base).

## Decision: K=3

K=3, no K=5. Razones:

1. **Las queries mas prometedoras necesitan K=2-3.** La busqueda en el corpus muestra que para Q8, Q12, Q13, Q20, Q21, el contenido correcto existe en archivos distintos al top-1 y deberia rankear en posiciones 2-3. Ninguna requiere K=5 para que el chunk correcto aparezca.

2. **Riesgo de dilucion.** Con K=5 meto ~2500 tokens de contexto al 8b. Tres chunks tangenciales + dos relevantes = mas material para que el modelo mezcle. La leccion de v5.0 es que mas contexto ambiguo empeora la generation, no la mejora. K=3 limita ese riesgo.

3. **Retornos decrecientes.** Si el chunk correcto no esta en top-3, probablemente esta en posicion 10+ (la distribucion de relevancia tiene cola larga). K=5 captura poco que K=3 no capture, pero paga el costo de dilucion.

4. **Espacio para iterar.** Si K=3 no alcanza, K=5 es v6.1 — un cambio trivial. Mejor empezar conservador y subir que empezar agresivo y no saber si el problema es K o el formato.

## Decision: formato del prompt

Contextos numerados: "Context 1: ... Context 2: ... Context 3: ..."

Razon: los numeros le dan al modelo una forma de referenciar chunks especificos ("Based on Context 2...") y señalan que son piezas DISTINTAS de informacion. Esto evita el efecto "sopa" de concatenar con --- donde los chunks se desdibujan en un texto continuo. El modelo necesita DISTINGUIR chunks relevantes de tangenciales, no fusionarlos.

Cambio en el prompt template:
- "context" → "contexts" (plural)
- "If the context doesn't contain" → "If none of the contexts contain"
- El cuerpo lista Context 1/2/3 separados

## Queries que esperado mejoren

### Alta confianza (contenido correcto existe y deberia rankear top-3)

**Q20 (view vs reshape):** Top-1 fue tensor_view.rst (cola del archivo — lista de metodos). La cabeza del mismo archivo explica view semantics, contiguity, storage compartido. Con K=3, otro chunk del mismo archivo deberia aparecer. **Mecanismo:** intra-file, mismo archivo, chunk diferente. p→u.

**Q12 (save/load model):** Top-1 fue aot_inductor.md (C++ inference). notes/serialization.rst tiene exactamente "Saving and loading torch.nn.Modules" con codigo de state_dict. **Mecanismo:** inter-file, el archivo correcto existe y es denso en vocabulario relevante (save, load, state_dict, checkpoint). p→u.

**Q8 (multiple GPUs):** Top-1 fue distributed.md (seccion de funciones deprecadas). El mismo archivo tiene "Basics" section sobre DDP, notes/ddp.rst tiene tutorial completo, notes/cuda.rst tiene best practices multi-GPU. **Mecanismo:** inter-file, contenido abundante en multiples archivos. p→u.

**Q13 (numpy→tensor):** Top-1 fue sparse.rst (ejemplos de torch.tensor para sparse). tensors.rst tiene al PRINCIPIO del archivo `torch.tensor(np.array(...))` y mencion de `torch.as_tensor`. **Mecanismo:** inter-file, chunk correcto al inicio de archivo con vocabulario directo. p→u.

### Confianza media (contenido existe pero mecanismo menos seguro)

**Q21 (learning rate scheduling):** Top-1 fue optim.md (chunk sobre chaining schedulers al final de epochs). El MISMO archivo tiene seccion "How to adjust learning rate" con ExponentialLR completo. **Mecanismo:** intra-file. Pero 8b mostro literalismo semantico en v5.1 ("middle of training" = "middle of epoch") — el riesgo es que 8b rechace AUNQUE el chunk correcto este presente, porque el chunk muestra scheduling per-epoch, no intra-batch. Si 8b ve la seccion completa con titulo "How to adjust learning rate", deberia conectar. **p→u probable pero no seguro.**

**Q15 (custom autograd):** Top-1 fue extending.func.rst (torch.func + autograd.Function avanzado). El archivo hermano notes/extending.rst tiene el tutorial basico con LinearFunction como ejemplo (forward, setup_context, backward). **Mecanismo:** inter-file, archivo hermano con nombre similar. Si BGE rankea extending.rst en top-3 para "implement custom autograd function", 8b puede extraer el tutorial correctamente — ya mostro que puede dar estructura de autograd.Function (Q15 en v5.1 fue mixto con estructura correcta pero detalles inventados). Con el chunk real, deberia ser fiel+util. **p→u posible.**

**Q19 (loss nan):** Top-1 fue faq.rst (accumulating history, no NaN). El mismo archivo tiene otras FAQs que podrian tocar estabilidad numerica, pero no encontre contenido especifico sobre NaN causes. **Mecanismo:** intra-file, incierto. **p→p probable.**

### Baja confianza / top-K no ataca el modo de fallo

**Q14 (flatten before linear):** Top-1 fue named_tensor.md, que MENCIONA `{meth}~Tensor.flatten` pero 8b no lo vio (markup blindness). Con K=3, podria aparecer nn.rst que lista `nn.modules.flatten.Flatten`, pero esta al 88% del archivo (linea 524 de 593) y es un autosummary listing, no prosa. **Top-K no ataca markup blindness.** El chunk que YA tenia 8b era relevante — el problema es que 8b no parsea Sphinx markup. Traer mas chunks con mas markup no arregla eso. **p→p.**

**Q11 (F.linear):** nn.functional.rst tiene `linear` bajo autosummary directive, sin descripcion ni firma. No hay prosa en el corpus que explique que hace F.linear. **El contenido no existe en forma accesible.** Top-K no puede encontrar lo que no esta. **p→p.**

**Q10 (.size vs .shape):** La diferencia entre .size() y .shape no esta documentada en el corpus. .size() no aparece como concepto en ningun archivo. **El contenido no existe.** p→p.

**Q7 (overfitting/memorization):** El corpus no tiene contenido sobre regularizacion, dropout, o data augmentation como tecnicas contra overfitting. "Memorizing training data" no matchea con ningun tema del corpus. **El contenido no existe en la forma que necesita el usuario.** p→p.

**Q18 (weights not changing):** En v5.1, 8b ya MEZCLO con top-1 (diagnostico inventado sobre named_parameters). Con K=3, mas chunks de optim.md llegan — incluyendo training loops con zero_grad/backward/step. **Riesgo alto de mas mezcla.** 8b podria tomar el training loop correcto y combinar con diagnostico inventado, produciendo un mixto mas elaborado. **p→p o regresion a x.**

## Riesgo nuevo: mezcla multi-chunk

Con top-1, el modelo tiene una decision binaria: usar el chunk o rechazar. Con K=3, tiene que DISTINGUIR chunks relevantes de tangenciales. Si un chunk es bueno y dos son tangenciales, el modelo debe usar solo el bueno. Si combina los tres, produce mixto.

La leccion de v5.0 aplica: mas contexto ambiguo empeora generation. La diferencia con v6: en v5.0, el contexto nuevo venia de una RE-BUSQUEDA (query diferente, chunks diferentes). En v6, los 3 chunks vienen de la MISMA busqueda y deberian tener coherencia tematica. El riesgo es menor pero no cero.

Prediccion: 1-2 queries que eran fiel en v5.1 podrian bajar a mixto porque los chunks adicionales les dan material para mezclar. Las candidatas mas vulnerables: queries con top-1 tangencial donde el chunk 2 o 3 es aun MAS tangencial (no mejor).

## Numeros comprometidos

### Por query (utilidad)

| Query | v5.1 | v6 pred | Confianza | Mecanismo |
|-------|------|---------|-----------|-----------|
| Q8 | p | u | Alta | Inter-file: DDP tutorial en corpus |
| Q12 | p | u | Alta | Inter-file: serialization.rst |
| Q13 | p | u | Alta | Inter-file: tensors.rst inicio |
| Q20 | p | u | Alta | Intra-file: tensor_view.rst cabeza |
| Q21 | p | u | Media | Intra-file: optim.md seccion schedulers |
| Q15 | p | u o p | Media | Inter-file: extending.rst tutorial |
| Q19 | p | p | Baja | Incierto si faq.rst tiene NaN content |
| Q14 | p | p | Alta | Markup blindness, top-K no ataca |
| Q11 | p | p | Alta | Contenido no existe en prosa |
| Q10 | p | p | Alta | Contenido no existe |
| Q7 | p | p | Alta | Contenido no existe |
| Q18 | p | p o x | Media | Riesgo de mas mezcla |

### Totales

| Metrica | v5.1 | v6 intuicion | v6 calibrada |
|---------|------|-------------|-------------|
| Generation fiel | 19/21 | 18/21 | 17/21 |
| Generation mixto | 2/21 | 3/21 | 4/21 |
| Generation inventado | 0/21 | 0/21 | 0/21 |
| Utility util | 9/21 | 14/21 | 13/21 |
| Utility parcial | 12/21 | 7/21 | 8/21 |
| Utility danino | 0/21 | 0/21 | 0/21 |

Notar: generation podria EMPEORAR ligeramente (19→17-18 fiel) porque mas chunks dan material para mezclar. El trade-off es generation fiel vs utility util: acepto 1-2 mezclas si producen respuestas utiles.

### Calibracion

Historial del predictor en utilidad:

| Version | Prediccion | Real | Error |
|---------|-----------|------|-------|
| v3 | 14/21 | 11/21 | -3 |
| v5.1 | 11-12 | 9 | -2 a -3 |

Patron: consistentemente optimista por 2-3 puntos en utilidad. Si aplico la correccion: 13 - 2 = 11. Pero esta vez la intervencion ataca el cuello de botella real (retrieval) por primera vez. Las correcciones anteriores fueron sobre intervenciones que no atacaban el cuello de botella. La correccion historica podria no aplicar.

Decision: **no aplico correccion historica completa** porque el mecanismo cambio. Ajusto -1 en vez de -2.

**Calibrada final: 13/21 util.**

## Categorica: cierra perros en la nieve?

Perros en la nieve: utilidad >= 13/21 AND daninos <= 3/21.

- Daninos: 0 (8b no inventa, y top-K no deberia cambiar eso). **Pasa.**
- Util: calibrada 13. **Justo en el umbral.**

## **VERDE**

v6 cierra perros en la nieve. Es la primera intervencion que ataca el cuello de botella #1 del techo distribuido (retrieval top-1 → top-3). El contenido correcto existe en el corpus para al menos 4 queries de alta confianza (Q8, Q12, Q13, Q20). Con esas 4, paso de 9 a 13. Q21 y Q15 son bonus.

**Riesgo principal:** si los chunks correctos no rankean top-3 (mi analisis de corpus dice que si, pero no verifique los embeddings). Si 2+ de las 4 de alta confianza fallan, caigo a 11 y la categorica es ROJA.

**Este commit es la prueba de que la prediccion existio antes que los resultados.**
