# Prediccion: v5 — Re-busqueda con query reformulada (Grupo B)

**Fecha:** 2026-04-10

## Que es v5

Re-busqueda adaptativa. Cuando el pipeline detecta que la primera respuesta es insuficiente (parcial / "I don't know"), reformula la query y busca de nuevo. Se evalua sobre las 6 queries del Grupo B (las que dieron utilidad=parcial en v3b) y se monitorean 5 canarios del Grupo C.

## Predicciones por query

---

### Q8: "How to use multiple GPUs in PyTorch"

**Estado actual (v3b):** Retrieval tangencial (distributed.md, chunk sobre multi-GPU functions deprecated) | Generation fiel ("I don't know") | Utilidad parcial.

**Rama del arbol:** DataParallel vs DDP vs FSDP. El sistema debe buscar la guia practica de entrenamiento distribuido, no la nota de deprecacion de las funciones colectivas multi-GPU.

**Query reformulada:** "PyTorch DistributedDataParallel DataParallel multi-GPU training setup tutorial"

**Razonamiento:** El corpus tiene notes/ddp.rst con documentacion extensa de DistributedDataParallel (implementacion interna, forward pass, gradientes). Tambien distributed.md tiene mas chunks ademas de la nota de deprecacion. La reformulacion cambia el ancla semantica de "multiple GPUs" (que matchea la nota de deprecacion) a "DistributedDataParallel" (que matchea el tutorial). Es un cambio real de embedding, no trivial.

**Prediccion: VERDE.** El contenido existe y la reformulacion apunta directamente a el. Alta confianza.

---

### Q11: "what does torch.nn.functional.linear do"

**Estado actual (v3b):** Retrieval tangencial (named_tensor.md, lista de operadores F soportados que NO incluye linear) | Generation fiel ("I don't know") | Utilidad parcial.

**Rama del arbol:** Firma y semantica de F.linear: y = xW^T + b. La diferencia entre el modulo nn.Linear y la funcion funcional F.linear.

**Query reformulada:** "linear transformation input weight bias matrix multiplication torch.nn.functional"

**Razonamiento:** El corpus NO tiene la documentacion de F.linear. nn.functional.rst es solo un autosummary que lista `linear` sin descripcion. La unica mencion de la operacion lineal con `input.mm(weight.t())` esta en notes/extending.rst como ejemplo de autograd.Function — un contexto totalmente distinto (extension de autograd, no API reference). El embedding de "F.linear" no va a matchear un tutorial de autograd extensions. Ademas, la reformulacion es trivialmente obvia: agregar "weight bias transformation" a la query original. Si BGE no encontro el contenido con "torch.nn.functional.linear", tampoco lo va a encontrar con "linear transformation weight bias".

**Prediccion: ROJA.** El contenido no existe en el corpus en forma accesible. No hay reformulacion que encuentre lo que no esta.

---

### Q12: "how to save and load a model for inference"

**Estado actual (v3b):** Retrieval tangencial (torch.compiler_aot_inductor.md, sobre inference en C++) | Generation fiel ("I don't know") | Utilidad parcial.

**Rama del arbol:** torch.save / torch.load / state_dict serialization. El patron canonico de guardar model.state_dict() y recargarlo con load_state_dict().

**Query reformulada:** "torch.save torch.load model state_dict checkpoint serialization"

**Razonamiento:** El corpus tiene notes/serialization.rst con EXACTAMENTE lo que se necesita: seccion "Saving and loading torch.nn.Modules" con codigo de `torch.save(bn.state_dict(), 'bn.pt')` y `new_bn.load_state_dict(bn_state_dict)`. El fallo original fue que "inference" como ancla semantica atrajo el chunk de AOT inductor (que trata sobre inference en C++). La reformulacion reemplaza "inference" por "state_dict/serialization", que es un cambio semantico real. El embedding de "save model state_dict serialization" deberia matchear serialization.rst con alta confianza.

**Prediccion: VERDE.** El contenido existe, es accesible, y la reformulacion apunta directamente. Alta confianza.

---

### Q15: "implement custom autograd function"

**Estado actual (v3b):** Retrieval bueno (notes/extending.func.rst, sobre torch.func + autograd.Function) | Generation mixto (el LLM invento `torch.autograd.register_autograd_function`) | Utilidad parcial.

**Rama del arbol:** El tutorial basico de autograd.Function: subclasear Function, definir forward/backward estaticos, save_for_backward, gradcheck. NO el caso avanzado de torch.func transforms.

**Query reformulada:** "autograd.Function subclass forward backward save_for_backward example tutorial"

**Razonamiento:** La primera busqueda encontro extending.func.rst (extension avanzada para torch.func), que es el archivo equivocado. El archivo correcto es notes/extending.rst, que tiene un tutorial completo con LinearFunction como ejemplo: forward con `input.mm(weight.t())`, setup_context con `save_for_backward`, backward con gradientes. La reformulacion agrega "forward backward save_for_backward" que es vocabulario especifico del tutorial basico. El embedding deberia distinguir entre "extending torch.func with autograd.Function" y "implementing autograd.Function forward backward" — son contextos distintos.

**Prediccion: VERDE.** El corpus tiene el tutorial exacto en un archivo hermano del que ya se encontro. La reformulacion con terminos especificos (forward/backward/save_for_backward) deberia matchear extending.rst. Alta confianza.

---

### Q18: "my model weights are not changing during training"

**Estado actual (v3b):** Retrieval tangencial (optim.md, chunk sobre mantener weights intencionalmente con `torch.save(optimizer.state_dict(), PATH)`) | Generation fiel ("I don't know") | Utilidad parcial.

**Rama del arbol:** Diagnostico del ciclo de entrenamiento: optimizer.zero_grad() → forward → loss → backward → optimizer.step(). Tambien: requires_grad, model.train() vs model.eval(), vanishing gradients.

**Query reformulada:** "optimizer.step zero_grad loss.backward training loop update parameters"

**Razonamiento:** optim.md tiene MULTIPLES ejemplos del training loop correcto (al menos 6 instancias de `optimizer.zero_grad(); loss.backward(); optimizer.step()`). El chunk original matcheo "weights need to remain unchanged" — una coincidencia lexica falsa. La reformulacion cambia de sintoma ("weights not changing") a mecanismo ("optimizer.step zero_grad backward"), que deberia matchear los ejemplos de training loop en optim.md. Sin embargo, hay un riesgo de generation: el LLM de 3B necesita conectar "aqui esta el training loop correcto" con "tus weights no cambian porque probablemente te falta uno de estos pasos". Esa inferencia no esta en el chunk — es diagnostica.

**Prediccion: VERDE.** El contenido existe y la reformulacion es semanticamente significativa. Confianza media-alta. El riesgo esta en generation, no en retrieval.

---

### Q20: "view vs reshape"

**Estado actual (v3b):** Retrieval tangencial (tensor_view.rst, chunk de lista de metodos que menciona "reshape can return view or new tensor") | Generation fiel ("I don't know") | Utilidad parcial.

**Rama del arbol:** Storage compartido, contiguidad, cuando view funciona y cuando no, como reshape decide si copia o no.

**Query reformulada:** "tensor view contiguous storage reshape copy shared data PyTorch"

**Razonamiento:** tensor_view.rst tiene la explicacion completa al principio del archivo: "View tensor shares the same underlying data with its base tensor", ejemplo con `t.view(2, 8)` y `storage().data_ptr()`, y la explicacion de contiguidad con `transpose` y `.contiguous()`. El chunk original era del final del archivo (la lista de metodos). La reformulacion con "contiguous storage shared data" deberia matchear los chunks iniciales del mismo archivo, que son mas explicativos.

**Prediccion: VERDE.** El contenido esta en el mismo archivo que ya se busco, solo en chunks diferentes. Alta confianza.

---

## Resumen de predicciones

| Query | Color | Confianza |
|-------|-------|-----------|
| Q8 (multiple GPUs) | VERDE | Alta |
| Q11 (F.linear) | ROJA | Alta |
| Q12 (save/load model) | VERDE | Alta |
| Q15 (custom autograd) | VERDE | Alta |
| Q18 (weights not changing) | VERDE | Media-alta |
| Q20 (view vs reshape) | VERDE | Alta |

**Composicion:** 5 verdes + 0 amarillas + 1 roja.

## Prediccion total

**Intuicion dice: recupero 5 de 6.**

**Calibracion historica dice: recupero 4 de 6.**

Historial del predictor:

| Version | Prediccion | Real | Error |
|---------|-----------|------|-------|
| v3 retrieval | 11/21 | 7/21 | -4 (optimista) |
| v3 generation | 14/21 | 14/21 | 0 |
| v3 utilidad | 14/21 | 11/21 | -3 (optimista) |
| v4.0 global | 75% | 56% | -19pp (optimista) |
| v4.1 global | 70% | 59% | -11pp (optimista) |

Patron: 4 de 5 predicciones fueron optimistas, con sesgo de -3 a -19pp. Restando 1 al numero intuitivo: 5 - 1 = 4. El error mas probable es que una de las verdes de confianza media-alta (Q18) no se recupere por problemas de generation, no de retrieval.

## Prediccion de canarios (Grupo C)

**Predigo que 1-2 de los 5 canarios se rompen.**

### Dato critico: 5/5 canarios activan el trigger

Todos los canarios (Q4, Q5, Q6, Q16, Q17) responden con "I don't know" en v3b. Si el trigger de re-busqueda es "la respuesta contiene 'I don't know'", los 5 entran a re-busqueda. Esto NO estaba en la prediccion original — asumia que los canarios no activaban el trigger.

### Analisis por canario

| Canario | Query | Riesgo | Razon |
|---------|-------|--------|-------|
| Q4 | capital de France | Bajo | Ningun chunk del corpus es tematicamente cercano. La re-busqueda encontrara otro chunk irrelevante y el LLM volvera a decir "I don't know". |
| Q5 | chocolate chip cookies | Bajo | Mismo razonamiento que Q4. El dominio es tan lejano que no hay chunk que induzca invencion. |
| Q6 | best optimizer for GANs | **Alto** | El corpus tiene contenido sobre optimizers (optim.md, compiler_faq.md). Una re-busqueda con "optimizer GAN training" podria encontrar un chunk de optim.md que menciona Adam/SGD. El LLM de 3B podria tomar eso y decir "Adam is the best optimizer for GANs" — confidently wrong. |
| Q16 | batch insert DB | Medio-bajo | "batch" y "insert" existen en el corpus (DataLoader batching). La re-busqueda podria encontrar otro chunk de data.md, pero el LLM probablemente sigue rechazando porque "database" no aparece en ningun chunk. |
| Q17 | pytorch vs tensorflow | **Medio-alto** | user_guide/index.md dice "PyTorch provides a flexible and efficient platform for deep learning." Una re-busqueda podria encontrar un chunk mas descriptivo sobre PyTorch, y el LLM podria elaborar una comparacion inventada en vez de rechazar. |

### Prediccion

**Q6 se rompe** (confianza media-alta): la re-busqueda encuentra contenido sobre optimizers, el LLM inventa una recomendacion.

**Q17 posiblemente se rompe** (confianza media): depende de si la re-busqueda encuentra un chunk lo suficientemente descriptivo sobre PyTorch como para que el LLM deje de rechazar.

Q4, Q5, Q16 sobreviven.

**Numero comprometido: 1 canario roto** (Q6). Rango: 0-2.

### Implicacion para el diseno

Este hallazgo sugiere que el trigger necesita una clausula de escape para OOD: si la query es detectablemente fuera de dominio Y la respuesta es "I don't know", NO re-buscar. El "I don't know" honesto ante OOD es el comportamiento correcto — la re-busqueda solo lo puede empeorar.

## Nota metodologica

Hice trampa parcial: verifique la existencia de archivos clave en el corpus (serialization.rst, ddp.rst, extending.rst, tensor_view.rst) antes de escribir las predicciones. Esto hace que las predicciones de "existe el contenido" sean mas confiables que las de v3 y v4, pero NO reduce la incertidumbre sobre si BGE realmente matchea la query reformulada con esos chunks. La prediccion sigue siendo sobre embeddings + generation, no solo sobre existencia de archivos.

## Numeros comprometidos (pre-commit)

- **Grupo B:** recupero 4 de 6 (intuicion 5, calibracion historica 5-1=4). Composicion: 5 verdes, 1 roja.
- **Grupo C:** 1 canario roto de 5 (Q6). Rango: 0-2.
- **Este commit es la prueba de que la prediccion existio antes que los resultados.**
