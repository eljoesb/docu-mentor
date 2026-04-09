# v2 — BM25 Baseline: Observaciones crudas

## Setup
- Corpus: 2912 chunks (mismo chunking que v1, copiado textual)
- Tokenización: lowercase + split por espacios (lo tonto)
- Index + búsqueda de 21 queries: < 0.2s total. Sin caché necesario.

## Resultado principal: 16/21 chunks diferentes

BM25 y embeddings coinciden en el chunk ganador en solo 5 de 21 queries.
Pero ojo: la comparación es por archivo fuente, no por chunk exacto.
En Q8 (multiple GPUs), ambos apuntan a `notes/cuda.rst` pero a chunks
distintos: embeddings encuentra el de `CUDA_VISIBLE_DEVICES`, BM25
encuentra uno sobre TF32 tensor cores. En Q18 y Q21, ambos apuntan a
`optim.md` pero a chunks distintos dentro del mismo archivo.

Solo Q9 (leaf tensor) y Q17 (pytorch vs tensorflow) coinciden en el
chunk exacto.

## Dónde gana cada uno

### Embeddings gana en queries conceptuales/semánticas
- **Q1** "what is autograd": embeddings → autograd.rst (definición perfecta).
  BM25 → distributed_autograd.rst (contexto distribuido, no la definición).
  BM25 matcheó "autograd" en el archivo equivocado porque distributed_autograd
  tiene más repeticiones de la palabra.
- **Q2** "how do I move a tensor to GPU": embeddings → tensor_attributes.rst
  (sobre devices y cómo mover tensores). BM25 → governance.rst (!). Fallo total
  de BM25 — la query no tiene keywords que matcheen la doc correcta. La respuesta
  habla de `.to(device)` y `torch.device`, no usa las palabras "move" ni "GPU"
  de la forma que BM25 las busca.
- **Q7** "stop my model from memorizing training data": embeddings → faq.rst
  (sobre no acumular historial). BM25 → dynamic_shapes_troubleshooting. Ni uno
  es sobre overfitting/regularización, pero embeddings al menos encontró algo
  sobre training loops.
- **Q11** "what does torch.nn.functional.linear do": embeddings → extending.rst
  (sobre implementar un módulo Linear). BM25 → compiler_dynamo_deepdive.md
  (sobre C++ bindings, nada que ver). Score BM25: 9.56 — el más bajo. Este es
  el caso predicho de tokenización: "torch.nn.functional.linear" es un solo
  token y no matchea nada bien.
- **Q19** "why is my loss nan": embeddings → autograd.rst con el ejemplo exacto
  de NaN por división por cero. BM25 → compiler_faq ("why is my code crashing").
  BM25 matcheó "why" + "is" + "my" en la pregunta equivocada.
- **Q20** "view vs reshape": embeddings → tensor_view.rst (sobre view ops,
  contiguidad). BM25 → torch.rst (lista de nombres de funciones). Embeddings
  captura la semántica de "comparar dos conceptos".

### BM25 gana en queries con keywords específicos
- **Q3** "how to use DataLoader with multiple workers": BM25 → randomness.rst
  con código real de `DataLoader(num_workers=...)`. Embeddings → data.md con
  notas sobre serialización en Windows. BM25 matcheó "DataLoader", "workers",
  "num_workers" directamente.
- **Q12** "how to save and load a model for inference": BM25 → serialization.rst
  ("how you can save and load PyTorch tensors and module states"). Embeddings →
  hub.md sobre download_url_to_file. BM25 matcheó "save", "load" exacto.
- **Q15** "implement custom autograd function": BM25 → extending.rst ("When to
  use... implement a custom function if you want to perform computations...").
  Embeddings → autograd.rst (intro general de autograd). BM25 matcheó "custom",
  "function", "autograd" juntos.

### Ambos decentes o ambos malos
- **Q8, Q9, Q17, Q21**: ambos encuentran chunks razonables (o el mismo).
- **Q4, Q5, Q16** (out-of-domain): ambos devuelven basura, como se espera.
  Ningún retriever debería encontrar algo útil para "chocolate chip cookies"
  en docs de PyTorch.
- **Q6** (opinion): ambos tangenciales. Bien.
- **Q10, Q14**: ambos fallan — ni "difference between .size() and .shape" ni
  "flatten tensor before linear layer" tienen buena cobertura en el corpus.

## Patrón claro

| Tipo de query | Gana |
|---|---|
| Conceptual ("what is X") | Embeddings |
| Jargon mismatch (el usuario no usa el término técnico) | Embeddings |
| Keyword-rich ("save and load a model") | BM25 |
| API names con puntos ("torch.nn.functional.linear") | Ninguno (tokenización rota) |
| Out-of-domain | Empate (ambos basura, bien) |

## Observación sobre tokenización (Q11)

La query 11 ("what does torch.nn.functional.linear do") tiene el score BM25
más bajo (9.56) de todas las queries que esperan respuesta. Confirmado: el
token "torch.nn.functional.linear" como unidad no matchea bien. Un tokenizer
que rompa por puntos mejoraría este caso, pero necesitaría medir si introduce
ruido en otros.

## Observación sobre BM25 y queries "humanas"

BM25 falla brutalmente en Q2 ("move tensor to GPU" → governance.rst). Esto es
porque la pregunta usa lenguaje coloquial y la respuesta usa API names
(`.to(device)`). Embeddings captura la equivalencia semántica entre "move to
GPU" y "device=cuda". BM25 no puede hacer eso.

Las queries de jargon_mismatch (Q7, Q18, Q19) son parecidas: el usuario
describe el síntoma ("my loss is nan", "weights not changing") y la doc usa
terminología técnica ("division by zero in backward", "optimizer state_dict").

## Conclusión para hybrid search

Esto es exactamente el caso para combinar ambos:
- BM25 gana cuando las keywords del usuario aparecen literalmente en el chunk correcto
- Embeddings gana cuando hay gap semántico entre la pregunta y la respuesta
- No se complementan al azar — se complementan *sistemáticamente* en tipos de query distintos

Un hybrid search con RRF debería capturar lo mejor de ambos: en Q12 (save/load),
el chunk de serialization.rst subiría en el ranking combinado. En Q1 (autograd),
el chunk de autograd.rst subiría. En Q2, BM25 no contribuye nada útil pero
tampoco daña si el ranking combina correctamente.

---

## Entrega: respuestas concretas

### ¿En cuántas queries cambió el chunk ganador?

16 de 21. Pero eso sobreestima la divergencia real: la comparación es por
archivo fuente. De las 5 que dicen "mismo archivo", solo Q9 y Q17 son el
mismo chunk exacto. Q8, Q18 y Q21 apuntan al mismo .rst/.md pero a
párrafos distintos dentro del archivo. La divergencia real es más como 19/21.

### 3 queries examinadas a ojo

**Q12 — "how to save and load a model for inference" (BM25 debería brillar)**

Embeddings eligió un chunk de `hub.md` que habla de `download_url_to_file` y
`load_state_dict_from_url` — funciones de torch.hub, no el flujo estándar de
save/load. El chunk empieza con `autofunction:: download_url_to_file` y luego
pregunta "how can you find out what you can do with the model?" — tangencial.

BM25 eligió un chunk de `notes/serialization.rst` que abre con: "This note
describes how you can save and load PyTorch tensors and module states in
Python, and how to serialize Python modules so they can be loaded in C++."
Luego menciona `torch.save` y `torch.load` directamente.

**Veredicto: BM25 es claramente mejor.** El match literal de "save", "load",
"model" lo llevó al chunk correcto. Embeddings se confundió con la similitud
semántica entre "load a model" y "load from hub".

**Q7 — "how do I stop my model from memorizing the training data" (BM25 debería fallar)**

Embeddings eligió un chunk de `notes/faq.rst`: "Don't accumulate history
across your training loop. By default, computations involving variables that
require gradients will keep history." No es sobre overfitting/regularización,
pero al menos es sobre un error común en training loops que causa que el
modelo se comporte mal.

BM25 eligió un chunk de `dynamic_shapes_troubleshooting_guardon_errors.md`:
"Do I know one path will always be taken?" con un ejemplo de `torch._check`.
Nada que ver con memorización, overfitting, ni training. BM25 matcheó
probablemente en "model" y "my" — palabras de alta frecuencia y bajo IDF que
no dicen nada sobre la intención de la query.

**Veredicto: BM25 es peor.** La query usa lenguaje coloquial ("memorizing")
que no existe en la documentación técnica. Embeddings al menos capturó la
intención general de "algo anda mal en mi training". BM25 devolvió ruido.

**Q11 — "what does torch.nn.functional.linear do" (BM25 debería brillar en API names)**

Embeddings eligió un chunk de `notes/extending.rst` que habla sobre
implementar un módulo Linear custom. Menciona `Linear` en contexto y es
razonablemente útil aunque no describe qué hace `F.linear` específicamente.

BM25 eligió un chunk de `compiler_dynamo_deepdive.md` que habla sobre
tracing de Python a C++ y dice "What can a tracer do when it finds an
operation that it does not understand?" — completamente irrelevante. Score
BM25: 9.56, el más bajo de todas las queries que esperan respuesta.

**Veredicto: BM25 es peor, y es el caso más revelador.** Se suponía que
BM25 debía brillar acá — es una query con un nombre de API literal. Pero
"torch.nn.functional.linear" como un solo token (lowercase + split por
espacios no rompe por puntos) no matchea nada. Si el tokenizer rompiera
por puntos, los tokens serían ["what", "does", "torch", "nn", "functional",
"linear", "do"] y "linear" + "functional" + "nn" tendrían chances de matchear
chunks sobre módulos lineales. Este es el caso más fuerte para probar un
segundo tokenizer.

### ¿Algún patrón inesperado?

Sí: **BM25 es más vulnerable a la frecuencia de stopwords de lo que
esperaba.**

En Q19 ("why is my loss nan"), BM25 eligió el chunk de compiler_faq que
abre con "Why is my code crashing?" — matcheó en "why", "is", "my" que son
3 de las 5 palabras de la query. Las palabras con señal real ("loss", "nan")
quedaron diluidas. BM25 teóricamente baja el peso de stopwords vía IDF, pero
"my" no es tan frecuente en documentación técnica como en lenguaje natural,
así que su IDF no es tan bajo como uno esperaría. El resultado es que BM25
se dejó llevar por la similitud superficial de estructura de frase ("why is
my X") en vez del contenido técnico.

Otro patrón inesperado: **Q2 ("move tensor to GPU") → governance.rst**.
Esperaba que BM25 encontrara algo irrelevante pero al menos técnico. Que
haya aterrizado en el FAQ de governance de la comunidad PyTorch es un fallo
más grave de lo que anticipé. Significa que ningún chunk técnico tiene buena
cobertura de las palabras "move", "tensor", "GPU" juntas — la documentación
usa `.to()`, `device=`, y `cuda` en vez de "move to GPU". Este gap léxico
es precisamente donde embeddings brilla y BM25 no tiene forma de competir.

---

## Bloque 1.5 — Smart tokenizer (rompe por puntos/guiones/paréntesis)

### Setup

Segundo tokenizer: `re.sub(r"[.\-_()/\\\[\]{}'\"\`]", " ", text.lower()).split()`
Rompe `torch.nn.functional.linear` → `["torch", "nn", "functional", "linear"]`.
Ambos tokenizers corren sobre el mismo corpus de 2912 chunks. Indexación: 0.09s.

### Resultado: 9/21 queries cambiaron de chunk entre dumb y smart

No todas para mejor. Hay un tradeoff real.

### Predicciones vs realidad

**Q11 — CONFIRMADA.** Score saltó de 9.56 → 20.12 (más del doble). Chunk
cambió de compiler_dynamo_deepdive (irrelevante) a fx.md, que tiene un
ejemplo con `self.linear = torch.nn.Linear(512, 512)`. No es la respuesta
perfecta (describe nn.Linear, no nn.functional.linear), pero es mucho más
relevante que lo que daba el tokenizer tonto. La hipótesis era correcta:
romper por puntos libera los tokens "nn", "functional", "linear" y permite
matchear chunks sobre módulos lineales.

**Q2 — CONTROL CONFIRMADO.** Movió de governance.rst a rpc.md, pero sigue
siendo basura. Score apenas cambió (16.79 → 17.07). Como se predijo: el
problema de Q2 es gap semántico ("move to GPU" vs ".to(device)"), no
tokenización. Ningún tokenizer arregla esto.

**Q20 — GANANCIA NO PREDICHA.** Smart tokenizer ahora coincide con
embeddings: ambos eligen tensor_view.rst. Score de 8.87 → 14.02. El chunk
habla de reshape, flatten, view y contiguous. Romper por puntos liberó
"reshape" y "view" de referencias compuestas como `:meth:\`~torch.Tensor.reshape\``
donde el tokenizer tonto no separaba los componentes.

**Q14 — GANANCIA.** Smart encuentra nn.rst que lista literalmente
`nn.modules.flatten.Flatten` y `nn.modules.flatten.Unflatten`. Dumb
encontraba distributed.fsdp (irrelevante). Embeddings encontraba sparse.rst
(irrelevante). Smart tokenizer gana a los otros dos.

### La regresión: Q12

**Q12 — REGRESIÓN.** Smart tokenizer perdió el chunk bueno. Dumb encontraba
serialization.rst ("how you can save and load PyTorch tensors and module
states") — respuesta directa. Smart encuentra autograd.rst sobre inference
mode — tangencial.

¿Por qué? El smart tokenizer rompe todas las referencias a `torch.save`,
`torch.load`, `model.state_dict()` en tokens individuales: "torch", "save",
"load", "model", "state", "dict". Esos tokens aparecen en muchos más chunks
ahora (cualquier chunk que mencione `torch.algo` contribuye "torch"). El
token "save" pierde especificidad relativa. El IDF de los tokens comunes
baja. Serialization.rst ya no domina.

Esto es el tradeoff fundamental: **romper nombres de API mejora queries sobre
API names pero diluye queries sobre acciones ("save", "load") donde el
tokenizer tonto matcheaba mejor.**

### Tabla de ganadores y perdedores

| Query | Dumb | Smart | Veredicto |
|---|---|---|---|
| Q11 (functional.linear) | basura | tangencial | Smart gana |
| Q14 (flatten + linear) | basura | nn.rst (directo) | Smart gana |
| Q20 (view vs reshape) | lista de funciones | tensor_view.rst | Smart gana |
| Q12 (save/load model) | serialization.rst (directo) | autograd/inference | **Smart pierde** |
| Q2 (move to GPU) | governance (basura) | rpc (basura) | Empate (ambos mal) |
| Q3 (DataLoader workers) | randomness.rst | faq.rst | Lateral |
| Q13 (numpy to tensor) | compiler_faq (name-drop) | extending.rst | Lateral (ambos tangenciales) |
| Q15 (custom autograd) | extending.rst | extending.func.rst | Lateral |
| Q16 (database, OOD) | governance.rst | data.md | Lateral (ambos OOD) |

Score: 3 ganancias claras, 1 regresión clara (Q12), 5 laterales.
(Q13 parecía regresión pero tras verificación manual es lateral — ver
sección "Entrega Bloque 1.5" abajo.)

### La regresión de Q12 importa

No es un detalle menor. Q12 era uno de los 3 casos donde BM25-dumb
claramente superaba a embeddings. Al arreglar el tokenizer para API names,
se rompió el caso donde keyword matching simple funcionaba perfecto.

Esto significa que el smart tokenizer no es estrictamente superior al dumb.
Es mejor en un eje (API names) y peor en otro (acciones con palabras
comunes). Para el Bloque 2, hay que decidir:

1. Usar smart tokenizer y aceptar la regresión de Q12 (3 mejoras vs 1
   regresión, balance neto positivo).
2. Usar dumb tokenizer y renunciar a Q11/Q14/Q20.
3. Usar RRF de tres sistemas: embeddings + BM25-dumb + BM25-smart.

La opción 1 parece la más práctica. La opción 3 es interesante pero agrega
complejidad sin saber si el beneficio marginal lo justifica.

### Observación sobre governance.rst

Con el smart tokenizer, Q16 (batch insert database) dejó de converger a
governance.rst — ahora cae en data.md. Q2 también dejó governance.rst (ahora
cae en rpc.md). Q5 (cookies) sigue en governance.rst. Esto sugiere que
governance.rst era un atractor para el tokenizer tonto por tener muchas
palabras comunes ("how", "do", "I", "would", "like") en su texto de FAQ.
El smart tokenizer rompe ese efecto parcialmente al expandir el vocabulario
de tokens técnicos que compiten.

---

## Entrega Bloque 1.5

### 1. ¿Q11, Q13, Q15 mejoraron con smart tokenization?

**Q11 — Sí, mejoró.** Score de 9.56 → 20.12. El chunk pasó de
compiler_dynamo_deepdive (sobre tracing C++, completamente irrelevante) a
fx.md (ejemplo con `self.linear = torch.nn.Linear(512, 512)`). El chunk
nuevo no es la respuesta perfecta — describe nn.Linear (el módulo), no
nn.functional.linear (la función) — pero al menos está en el tema de
capas lineales. De "basura" a "tangencial". La hipótesis se confirmó.

**Q13 — Lateral, no la regresión que pensaba.** Dumb encontraba
compiler_faq.md que menciona `torch.from_numpy` — pero como name-drop
incidental. El chunk es sobre si NumPy funciona con `torch.compile`, no
sobre cómo convertir un array. No hay ejemplo de uso ni explicación de
semántica. Smart encontró extending.rst sobre el protocolo
`__array_function__` — meta-información sobre cómo PyTorch extiende NumPy.
Ambos son tangenciales: dumb name-dropea la función correcta en contexto
equivocado, smart habla del ecosistema correcto sin la función correcta.
Ni uno responde "cómo convierto un numpy array a tensor".

**Q15 — Lateral, ambos buenos.** Dumb → extending.rst ("implement a custom
function if you want to perform computations... with the autograd engine").
Smart → extending.func.rst ("So you'd like to use torch.autograd.Function
with torch.func transforms"). Ambos son directamente relevantes. Smart
encontró un archivo más específico (extending.func vs extending), pero la
calidad del chunk es comparable. No es una mejora ni una regresión.

### 2. ¿Q2 se mantuvo sin mejora?

**Confirmado.** Smart tokenizer movió Q2 de governance.rst a rpc.md. Ambos
son basura para "how do I move a tensor to GPU". Score apenas cambió
(16.79 → 17.07). El problema de Q2 es léxico-semántico: el usuario dice
"move to GPU", la doc dice `.to(device)` y `cuda`. Ningún tokenizer
resuelve eso — hace falta comprensión semántica, que es lo que embeddings
aporta.

### 3. Conteo agregado

**9/21 queries difieren entre dumb y smart.**

De esas 9:
- 3 smart cualitativamente mejor a ojo: Q11, Q14, Q20
- 1 smart cualitativamente peor: Q12
- 5 lateral (ni mejor ni peor de forma clara): Q2, Q3, Q13, Q15, Q16

(Q13 originalmente la conté como regresión de smart, pero al verificar el
chunk de dumb a mano, el `torch.from_numpy` era un name-drop incidental en
un chunk sobre torch.compile, no una guía de conversión. Ambos tangenciales.)

Balance neto: +2 (3 mejoras - 1 regresión). El smart tokenizer no es una
mejora universal pero el balance es más favorable de lo que parecía. El
único caso donde genuinamente pierde es Q12 (save/load → dilución de IDF).

### 4. Predicción para Bloque 2: RRF (embeddings + BM25-smart)

Baseline actual: 3/21 queries con retrieval "bueno" (Q1, Q9, Q19 según
la evaluación manual de v1).

**Predicción: 7/21.**

Razonamiento:
- Los 3 buenos existentes se mantienen: RRF no debería romperlos porque
  embeddings ya los rankea #1 y ese voto pesa en la fusión.
- Q14 sube a bueno: BM25-smart encuentra nn.rst con `Flatten`/`Unflatten`
  listados — directamente relevante. Embeddings no lo tiene. RRF lo
  surfacea.
- Q15 sube a bueno: BM25-smart encuentra extending.func.rst sobre
  `torch.autograd.Function` — respuesta directa. Embeddings tenía
  autograd.rst genérico. RRF lo promueve.
- Q20 sube a bueno: ambos sistemas convergen en tensor_view.rst con el
  smart tokenizer. El chunk habla de reshape, view, flatten, contiguous.
  RRF refuerza.
- Q21 posible: ambos apuntan a optim.md con contenido de schedulers. Era
  "tangencial" en v1, pero RRF podría surfacear un chunk mejor si está en
  el top-k de algún sistema.

Riesgos a la baja:
- RRF combina rankings, no garantiza que el top-1 después de fusión sea
  el mejor de cada sistema. Si embeddings tiene el chunk correcto en #1
  y BM25 lo tiene en #50, el chunk de BM25 #1 (basura) puede competir.
- Q12 perdió su chunk bueno (serialization.rst) al pasar a smart. Si
  BM25-dumb hubiera estado en la mezcla, Q12 también subiría. Con smart,
  probablemente no.

El 7 es optimista-pero-defendible. Si estuviera forzado a dar un rango,
diría 5-8, con 7 como punto central.

---

## Bloque 2 — RRF: embeddings + BM25-smart (k=60, top_n=50)

### Tabla comparativa de 4 columnas

(Output de measure_hybrid.py — ver consola para tabla completa)

```
ID CAT               EMB_SRC                   BM25S_SRC                  HYBRID_SRC                  D  S TYPE
 1 core_concept      notes/autograd.rst        rpc/distributed_autograd   rpc/distributed_autograd    4  3 compromise
 2 how_to            tensor_attributes.rst     rpc.md                     notes/mps.rst               9  2 compromise
 3 how_to            data.md                   notes/faq.rst              data.md                     1  2 compromise
 4 out_of_domain     notes/autograd.rst        fx.md                      fx.md                      24  2 compromise
 5 out_of_domain     sparse.rst                community/governance.rst   package.md                 41  2 compromise
 6 opinion           notes/modules.rst         notes/autograd.rst         notes/modules.rst           7 10 compromise
 7 jargon_mismatch   notes/faq.rst             dynamic_shapes_trouble..   notes/get_start_xpu.rst     6  1 compromise
 8 how_to            notes/cuda.rst            notes/cuda.rst             notes/cuda.rst              0  0 consensus
 9 core_concept      notes/autograd.rst        notes/autograd.rst         notes/autograd.rst          0  0 consensus
10 core_concept      compiler_dynamic_shapes   compiler_faq.md            export/programming_model    8  1 compromise
11 core_concept      notes/extending.rst       fx.md                      fx.md                      15  0 sparse_wins
12 how_to            hub.md                    notes/autograd.rst         hub.md                      0  8 dense_wins
13 how_to            tensor_view.rst           notes/extending.rst        tensors.rst                14  2 compromise
14 how_to            sparse.rst                nn.rst                     named_tensor.md             1 12 compromise
15 how_to            notes/autograd.rst        extending.func.rst         extending.func.rst          9  0 sparse_wins
16 out_of_domain     sparse.rst                data.md                    data.md                     3  0 sparse_wins
17 opinion           pytorch_main_components   pytorch_main_components    pytorch_main_components     0  0 consensus
18 jargon_mismatch   optim.md                  optim.md                   optim.md                    7  1 compromise
19 jargon_mismatch   notes/autograd.rst        compiler_faq.md            notes/faq.rst               1  2 compromise
20 ambiguous         tensor_view.rst           tensor_view.rst            tensor_view.rst             4  0 sparse_wins
21 ambiguous         optim.md                  optim.md                   optim.md                    1  2 compromise
```

CHANGE_TYPE: consensus=3, compromise=13, sparse_wins=4, dense_wins=1.

### Análisis de 5 queries

**Q11 (torch.nn.functional.linear) — sparse_wins, D:15 S:0**

Hybrid eligió fx.md, el top-1 de BM25-smart. Chunk: ejemplo de fx tracing
con `self.linear = torch.nn.Linear(512, 512)`. Es sobre nn.Linear (módulo),
no nn.functional.linear (función). Embeddings tenía extending.rst (sobre
implementar un módulo Linear custom) en posición 15 — demasiado lejos para
competir en RRF.

Veredicto: tangencial. Ni peor ni mejor que embeddings solo. Hybrid no
aportó acá — simplemente dejó ganar a BM25-smart.

**Q7 (memorizing training data) — compromise, D:6 S:1**

Hybrid eligió notes/get_start_xpu.rst — un getting started de XPU con
código de torch.compile para inference. Completamente irrelevante para
overfitting/memorización. Embeddings tenía faq.rst (D:6, sobre no acumular
historial en training loops) que era tangencial pero al menos hablaba de
training. BM25 tenía dynamic_shapes_troubleshooting (S:1, sobre branching).

RRF encontró un chunk que era #6 en embeddings y #1 en BM25 — el get_start
xpu fue un "decent in both" que le ganó al "tangential in one". Peor que
embeddings solo.

Veredicto: malo. Regresión.

**Q13 (numpy to tensor) — compromise, D:14 S:2**

Hybrid eligió tensors.rst. El chunk muestra literalmente:
`torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))` y luego advierte que
torch.tensor siempre copia datos, y sugiere `torch.as_tensor` para evitar
copias. Esto responde la pregunta directamente: muestra la conversión
funcionando y da contexto sobre performance.

Ni embeddings (tensor_view.rst, tangencial) ni BM25-smart (extending.rst,
tangencial) tenían esto como top-1. RRF lo surfaceó porque estaba en
posición 14 de embeddings y posición 2 de BM25-smart — un chunk que ningún
sistema individual hubiera elegido, pero que es el mejor de los tres.

**Veredicto: bueno. Este es el caso genuino de RRF agregando valor.**

**Q2 (move tensor to GPU) — compromise, D:9 S:2**

Hybrid eligió notes/mps.rst que muestra:
```
mps_device = torch.device("mps")
x = torch.ones(5, device=mps_device)
model.to(mps_device)
```
El patrón `.to(device)` es correcto pero para MPS (Apple Silicon), no CUDA.
Embeddings tenía tensor_attributes.rst (device=cuda, cómo se crean tensores
en un device), que era tangencial pero sobre CUDA. La diferencia es menor:
ambos muestran el patrón de device placement, uno para CUDA y otro para MPS.

Veredicto: tangencial. Lateral vs embeddings — mismo nivel, distinto GPU.

**Q18 (weights not changing) — compromise, D:7 S:1**

Hybrid eligió optim.md, mismo archivo que ambos sistemas individuales pero
un chunk sobre MoE expert duplication y "loading model weights and optimizer
states". Menciona "model weights" pero en contexto de duplicar un modelo,
no de diagnosticar por qué los pesos no cambian durante training. No ayuda
a un usuario que está debuggeando un training loop roto.

Veredicto: tangencial. No mejoró vs embeddings.

### Conteo de retrieval — criterio estricto

Aplico el mismo criterio que v1: "bueno" = el chunk responde directamente
la pregunta. No sobrevalorar name-drops ni matches superficiales.

| Categoría | Queries | Count |
|---|---|---|
| Bueno | Q9, Q13 | 2 |
| Tangencial | Q2, Q3, Q6, Q8, Q11, Q12, Q15, Q17, Q18, Q19, Q20, Q21 | 12 |
| Malo | Q1, Q4, Q5, Q7, Q10, Q14, Q16 | 7 |

**Resultado: 2/21 bueno.**

### Comparación con predicción

| Métrica | Predicción | Resultado |
|---|---|---|
| Bueno | 7/21 | 2/21 |

**Me equivoqué por 5 queries enteras.** Dirección: mucho más pesimista de
lo que predije.

Errores específicos de mi predicción:

**Predije que se mantendrían y se perdieron:**
- Q1: predije que RRF mantendría el bueno de embeddings. RRF lo perdió.
  La definición de autograd (D:0 en embeddings) fue superada por un chunk
  de distributed_autograd (D:4, S:3) porque el compromiso "decente en ambos"
  sumó más score RRF que el "perfecto en uno solo".
- Q19: mismo patrón. El ejemplo de NaN (D:0 en embeddings) fue superado por
  faq.rst (D:1, S:2). El chunk de faq.rst es sobre acumular loss en training,
  tangencial. El chunk de autograd.rst con el ejemplo de NaN era bueno.

**Predije que subirían a bueno y no subieron:**
- Q14: predije que nn.rst (Flatten) subiría. RRF eligió named_tensor.md en
  su lugar — un compromiso (D:1, S:12) que no ayuda.
- Q15: predije que extending.func.rst subiría. RRF sí lo eligió (sparse_wins,
  S:0) pero a ojo es tangencial, no bueno — habla de autograd.Function
  con torch.func transforms, no el tutorial básico.
- Q20: predije refuerzo por consenso en tensor_view.rst. RRF sí lo eligió
  (sparse_wins, D:4, S:0) pero el chunk es sobre qué ops devuelven views vs
  copias, no una comparación directa de view vs reshape. Tangencial.
- Q21: predije posible upgrade. RRF eligió un chunk de optim.md sobre
  SWA/EMA scheduling, no sobre el uso básico de schedulers.

**No predije y fue ganancia:**
- Q13: compromise encontró tensors.rst con conversión numpy-to-tensor.
  Ni embeddings ni BM25-smart lo tenían como top-1. RRF genuinamente
  surfaceó un chunk mejor. El único caso.

### Diagnóstico: por qué RRF empeoró

El patrón es claro y el mecanismo es mecánico:

**13/21 queries son "compromise" — el chunk ganador no era #1 de ningún
sistema.** Con k=60, la diferencia de score entre rank 0 y rank 5 es
mínima (1/60 vs 1/65 ≈ 3%). Un chunk en posición 4+3 en ambos sistemas
(score: 1/64 + 1/63 = 0.0312) le gana a un chunk en posición 0 en uno
solo (score: 1/60 = 0.0167). **El "decente en ambos" siempre le gana al
"perfecto en uno" con k=60.**

Esto mató Q1 y Q19: embeddings tenía el chunk perfecto en posición 0,
pero BM25-smart no lo veía (probablemente fuera de su top-50). Un chunk
mediocre que estaba en top-5 de ambos acumuló más score RRF.

Las 3 consensus (Q8, Q9, Q17) son los únicos casos donde RRF no daña:
ambos sistemas ya estaban de acuerdo, entonces el chunk ganador suma
score de ambos y domina. Pero son solo 3 de 21.

### Qué implica esto para v2

RRF con k=60 y top_n=50 **no funciona** como retrieval strategy para este
corpus y estos dos sistemas. No solo no llega a 10/21, sino que retrocede
de 3/21 a 2/21.

Posibles direcciones (sin implementar aún):
1. **Bajar k** — con k=1, posición 0 da score 1/1=1.0 y posición 4 da
   score 1/5=0.2. Ahora "perfecto en uno" (1.0) le gana a "decente en
   ambos" (0.2+0.2=0.4). Esto preservaría los bueno de embeddings.
2. **Ponderar sistemas** — dar más peso a embeddings (que tiene los 3
   bueno originales) y menos a BM25 (que aporta señal complementaria pero
   no debe dominar).
3. **Conditional fusion** — usar BM25 solo cuando embeddings tiene baja
   confianza (score < threshold), no siempre.
4. **Aceptar que el cuello de botella no es retrieval fusion** sino
   chunking y cobertura del corpus.
