# v3 — Pipeline completo BGE + Ollama: Notas de evaluacion

## Dos corridas, un hallazgo metodologico

v3 tuvo dos corridas:
- **v3a** (prompt original): "Use ONLY the following context to answer the question."
- **v3b** (prompt mejorado): agrega "If the context discusses a related topic but doesn't directly answer the question, say so explicitly instead of adapting the context to fit the question."

Los chunks recuperados son **identicos** en ambas corridas (0/21 diferencias, mismo BGE-base, mismo corpus, misma busqueda exacta por coseno). Solo cambio la generacion.

## Hallazgo critico: varianza intra-evaluador

La clasificacion de retrieval sobre los **mismos chunks** dio resultados muy distintos entre sesiones:

| Sesion | Bueno | Tangencial | Malo |
|--------|-------|------------|------|
| v3a (despues de muchas horas) | 15/21 | 3/21 | 3/21 |
| v3b (cabeza fresca) | 7/21 | — | — |
| v2.1 (sesion anterior) | 11/21 | 5/21 | 5/21 |

Varianza: **9 queries sobre 21** cambiaron de clasificacion entre sesiones. Todas en la misma direccion: downgrade (criterio mas estricto en v3b). Esto significa que diferencias menores a ~9 puntos entre dos mediciones podrian ser ruido del clasificador humano, no senal del sistema.

Las 9 queries que cambiaron (todas downgrade, mismos chunks):

| Query | v3a | v3b | Nota |
|-------|-----|-----|------|
| Q6 (best optimizer GANs) | b | t | Chunk de compiler_faq menciona optimizers pero no para GANs |
| Q7 (memorizing training data) | b | t | Chunk de faq.rst sobre acumular historial, no overfitting |
| Q8 (multiple GPUs) | b | t | Chunk de distributed.md solo dice "deprecated" |
| Q10 (.size vs .shape) | b | m | Chunk de export.md usa .shape en otro contexto |
| Q12 (save/load model) | b | t | Chunk de AOT inductor, no save/load estandar |
| Q13 (numpy to tensor) | b | m | Chunk de sparse.rst, nada sobre NumPy |
| Q17 (pytorch vs tf) | t | m | Intro generica de PyTorch, no compara |
| Q18 (weights not changing) | b | t | Chunk de optim.md sobre mantener weights, no diagnostico |
| Q19 (loss nan) | b | t | Chunk de faq.rst sobre acumular loss, no NaN |

Patron: las 9 son queries donde el chunk contiene **vocabulario tematicamente cercano** pero no responde la pregunta. Con cansancio, el evaluador interpreta "habla del tema" como "bueno". Con cabeza fresca, distingue "habla del tema" de "responde la pregunta". Estas 9 queries son los casos borderline ideales para calibrar LLM-as-judge.

**Decision:** usamos los numeros de v3b (cabeza fresca, criterio estricto) como resultado oficial. La corrida v3a queda documentada como dato crudo.

Esto refuerza la necesidad de LLM-as-judge para estabilizar las metricas en versiones futuras.

## Resultados oficiales (v3b — prompt mejorado)

| Eje | Bueno/Fiel/Util | Medio | Malo |
|-----|-----------------|-------|------|
| Retrieval | **7/21** bueno | — | — |
| Generation | **14/21** fiel | 5/21 mixto | 2/21 inventado |
| Utilidad | **11/21** util | 8/21 parcial | 2/21 danino |

### Comparacion con v1

| Eje | v1 | v3b | Delta |
|-----|----|----|-------|
| Retrieval bueno | 3/21 | 7/21 | **+4** |
| Generation fiel | 8/21 | 14/21 | **+6** |
| Generation inventado | 11/21 | 2/21 | **-9** |
| Utilidad util | 10/21 | 11/21 | **+1** |
| Danino | 6/21 | 2/21 | **-4** |

### Comparacion con prediccion

| Eje | Prediccion | Real | Error |
|-----|-----------|------|-------|
| Retrieval | 11/21 | 7/21 | -4 (criterio mas estricto) |
| Generation | 14/21 | 14/21 | **0** (con prompt mejorado) |
| Utilidad | 14/21 | 11/21 | **-3** |

### Prediccion micro-experimento prompt

Prediccion: 3 de 9 inventados pasan a fiel/mixto.
Real: 7 de 9 pasaron. Error: +4 (subestime el efecto del prompt).

Las 2 que resistieron: Q10 (.size/.shape — el LLM invento sobre NumPy) y Q19 (loss NaN — el LLM invento sobre gradientes que explotan). Ambas comparten el patron: el chunk contiene terminos de la pregunta en contexto diferente, y el LLM de 3B no distingue la diferencia semantica a pesar de la instruccion.

## Resultados de v3a (prompt original, dato crudo)

| Eje | Bueno/Fiel/Util | Medio | Malo |
|-----|-----------------|-------|------|
| Retrieval | 15/21 bueno | 3/21 tangencial | 3/21 malo |
| Generation | 7/21 fiel | 5/21 mixto | 9/21 inventado |
| Utilidad | 11/21 util | 8/21 parcial | 2/21 danino |

Estos numeros NO se usan como resultado oficial por la sobreestimacion de retrieval (15 vs 7 sobre mismos chunks). Se preservan como evidencia de varianza intra-evaluador.

## Diagnostico por query: los fallos mas instructivos

### Q7 — "how do I stop my model from memorizing the training data"

**v3a:** Retrieval bueno | Generation mixto | Utilidad danino
**v3b:** Retrieval tangencial | Generation mixto | Utilidad danino

El chunk de faq.rst habla de "don't accumulate history across your training loop" — sobre no acumular variables diferenciables, un problema de memoria/gradientes, NO de overfitting.

El LLM tomo "detach the variable" y lo presento como solucion a overfitting en ambas corridas. El prompt mejorado no arreglo esto — el LLM de 3B no distingue "training loop memory problem" de "overfitting". Es **confidently wrong**: `.detach()` no tiene nada que ver con regularizacion.

**Arco narrativo:** Este es el mismo fallo exacto de v1. En v1, MiniLM encontro el mismo chunk y el LLM dio la misma respuesta incorrecta. Mejorar retrieval no arreglo Q7 porque el fallo nunca fue de retrieval — fue de generation desde el principio. Solo se puede ver esto despues de mejorar retrieval lo suficiente como para descartar esa hipotesis.

### Q10 — "difference between .size() and .shape"

**v3a:** Retrieval bueno | Generation inventado | Utilidad util
**v3b:** Retrieval malo | Generation inventado | Utilidad danino

El chunk de export.md usa `.shape` en contexto de dynamic shapes. El LLM invento diferencias entre `.size()` y `.shape` (no existen — son lo mismo en PyTorch). En v3b invento sobre NumPy en vez de PyTorch.

Nota: la reclasificacion de retrieval (bueno → malo) y utilidad (util → danino) entre v3a y v3b refleja el cambio de criterio, no cambio de chunk. Con criterio estricto, un chunk que menciona `.shape` en contexto de export no responde "cual es la diferencia entre .size() y .shape".

### Q18 — "my model weights are not changing during training"

**v3a:** Retrieval bueno | Generation fiel | Utilidad parcial
**v3b:** Retrieval tangencial | Generation fiel | Utilidad parcial

El chunk de optim.md muestra `torch.save(optimizer.state_dict(), PATH)` en contexto de "weights need to remain unchanged" (intencionalmente).

En v3a el LLM diagnostico: "you're saving the optimizer's state dictionary instead of the model's parameters" — incorrecto pero usando solo info del chunk (fiel).
En v3b el LLM dijo: "the example code snippet doesn't explicitly demonstrate this issue. I don't know." — el prompt mejorado lo hizo mas honesto.

**Patron clave:** retrieval bueno + generation fiel no garantiza utilidad. Los tres ejes son genuinamente independientes. Solo la combinacion "chunk realmente responde" + "LLM es fiel" produce utilidad.

### Q19 — "why is my loss nan"

**v3a:** Retrieval bueno | Generation inventado | Utilidad util
**v3b:** Retrieval tangencial | Generation inventado | Utilidad parcial

El chunk de faq.rst habla de acumular `total_loss += loss` con historial de autograd (problema de memoria). El LLM invento sobre gradientes que explotan/desaparecen — una causa real de loss NaN pero que no esta en el chunk.

Una de las 2 queries donde el prompt mejorado no funciono. El LLM de 3B no obedece la instruccion de honestidad cuando el tema (loss + training) coincide superficialmente.

### Q20 — "view vs reshape"

**v3a:** Retrieval tangencial | Generation fiel | Utilidad danino
**v3b:** Retrieval tangencial | Generation fiel | Utilidad parcial

El chunk menciona que `reshape` puede retornar view o tensor nuevo. En v3a el LLM invento informacion falsa (`.device` para distinguir views). En v3b dijo "I don't know" — mejora directa del prompt.

## El efecto del prompt: una linea que elimino 7/9 invenciones

### Que cambio

Prompt original (v3a):
```
Use ONLY the following context to answer the question.
If the context doesn't contain the answer, say "I don't know based on the provided context."
```

Prompt mejorado (v3b):
```
Use ONLY the following context to answer the question.
If the context doesn't contain the answer, say "I don't know based on the provided context."
If the context discusses a related topic but doesn't directly answer the question,
say so explicitly instead of adapting the context to fit the question.
```

### Impacto medido

| Metrica | v3a | v3b | Delta |
|---------|-----|-----|-------|
| Generation fiel | 7/21 | 14/21 | **+7** |
| Generation inventado | 9/21 | 2/21 | **-7** |
| Utilidad danino | 2/21 | 2/21 | 0 |

La linea nueva elimino 7 de 9 invenciones. Las 2 resistentes (Q10, Q19) comparten un patron: el chunk contiene terminos de la pregunta en contexto semanticamente diferente, y el LLM de 3B no distingue la diferencia. Para esas 2, ni la instruccion de honestidad es suficiente — probablemente necesitan un modelo mas grande o chunks mas especificos.

### Efecto colateral: "no se" honesto no es utilidad

El prompt mejorado convirtio muchas invenciones en "no se" honestos. Eso mejoro generation (inventado → fiel) pero no siempre mejoro utilidad — un "no se" es fiel pero parcial. El sistema dejo de mentir pero no se volvio mas util en total.

Ejemplo: Q12 (save/load model). En v3a el LLM invento sobre `torch::jit::save` (fiel al chunk de AOT inductor). En v3b dijo "I don't know" (fiel, honesto). Utilidad en ambos casos: parcial. Diferente tipo de fallo, mismo resultado para el usuario.

## Los perros en la nieve: parcialmente cerrados

Regla de decision del prediction.md:
- Se cierran si utilidad >= 13/21 y daninos <= 3.

Resultado:
- Utilidad: 11/21. **No alcanza** (necesitaba 13).
- Daninos: 2/21. **Si alcanza** (necesitaba <= 3).

**Conclusion:** el sistema ya no miente (2 daninos vs 6 en v1). Pero no es suficientemente util (11/21 vs 13/21 necesarios). El LLM ahora es honesto — dice "no se" cuando no sabe. Pero ser honesto no es lo mismo que ser util. Para subir utilidad necesitas que el retrieval traiga mejores chunks, no que el LLM sea mas inteligente.

## Leccion principal de v3

El proyecto arranco con la premisa de que retrieval era el cuello de botella. En v2.1 se confirmo: cambiar el modelo de embeddings mejoro retrieval. En v3 se descubrio que habia **dos cuellos de botella simultaneos**: retrieval malo Y generation mala. Arreglar solo uno no alcanzaba.

Se arreglaron con dos intervenciones de costo radicalmente distinto:
- **Retrieval:** BGE-base (modelo de 440MB, cache de embeddings, migracion de modelo). Costo: horas.
- **Generation:** una linea de texto en el prompt. Costo: minutos.

A veces la solucion a un problema de ML es ingenieria pesada; otras veces es una frase en un prompt. Saber cuando aplicar cada una es lo que importa.

## Camino siguiente

### Prioridad 1: LLM-as-judge calibrado

La varianza intra-evaluador (±8 queries) hace insostenible la clasificacion humana manual para versiones futuras. LLM-as-judge con calibracion contra el baseline humano de v3 es el paso natural:
- Usar los datos de v3b como ground truth humano.
- Correr un LLM (Claude) como juez sobre las mismas 21 queries.
- Medir acuerdo inter-evaluador (humano vs LLM).
- Si el acuerdo es alto (>80%), usar LLM-as-judge para todas las versiones futuras.

### Prioridad 2: Mejorar retrieval

Con criterio estricto, retrieval esta en 7/21 bueno. Es el eje con mas potencial de mejora. Opciones:
- **Hybrid search BGE + BM25:** v2 hibrido fallo porque MiniLM era muy debil. Con BGE-base como componente denso, el regimen es diferente. Q13 (numpy → tensor) es el caso canonico donde BM25 aporta senal que embeddings puros no capturan.
- **Chunking mas granular:** Q3, Q8, Q20 podrian beneficiarse de chunks mas cortos o semantic chunking.

### Prioridad 3: Modelo mas grande (si prompt no alcanza)

Q10 y Q19 demuestran que hay un piso de comprension lectora que 3B no alcanza. Si las prioridades 1 y 2 no cierran los perros en la nieve, escalar a llama3.1:8b es el paso natural.
