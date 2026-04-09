# DocuMentor

Asistente de consulta sobre documentación técnica. Dada una pregunta en lenguaje natural sobre una librería, devuelve la sección relevante de los docs.

Proyecto guía del roadmap de transición a AI/ML Engineer. Evoluciona por versiones, donde cada versión nace del fracaso concreto de la anterior. La intención no es construir la mejor versión de una; es **sentir el dolor específico** que justifica el siguiente concepto del roadmap.

**Corpus actual:** documentación oficial de PyTorch (`docs/source` del repo `pytorch/pytorch`), 248 archivos `.rst` y `.md`.

---

## Version 0 — Retrieval léxico y semántico sobre docs de PyTorch

Objetivo de la v0: construir el pipeline más tonto posible que reciba una pregunta y devuelva un chunk relevante. Sin LLM, sin frameworks, sin bases de datos vectoriales. Solo Python, `scikit-learn` y `sentence-transformers`.

El recorrido atravesó cuatro iteraciones, cada una disparada por un fallo concreto de la anterior.

### Iteración 1 — Ranking por conteo de palabras

**Qué se construyó:** chunking por párrafos (split por `\n\n`) y ranking contando cuántas palabras de la query aparecen en cada chunk.

**Stats del corpus:** 248 archivos → 10,970 chunks.

**Queries de prueba:**

- `what is autograd` → `notes/extending.rst` (fragmento sobre extender autograd, no sobre qué es)
- `tensor cuda gpu` → `sparse.rst` (nota sobre compatibilidad con NVIDIA)
- `how to use DataLoader with multiple workers` → `rpc.md` (contenido de RPC sin relación con DataLoader)

**Fallo diagnosticado:** las palabras comunes ("to", "is", "the") dominan el ranking. Los chunks largos ganan injustamente porque acumulan más coincidencias. El ganador no es el más relevante, es el que tiene más palabras genéricas.

**Concepto que esto destapó:** hace falta ponderar por rareza (IDF) y normalizar por longitud.

### Iteración 2 — TF-IDF

**Qué cambió:** reemplazo del ranking tonto por `TfidfVectorizer` de scikit-learn + `cosine_similarity`. El chunking se mantuvo igual.

**Queries de prueba (mismas tres):**

- `what is autograd` → `func.whirlwind_tour.md` → solo el texto `"## What is torch.func?"`
- `tensor cuda gpu` → `get_start_xpu.rst` → dos líneas de código sueltas
- `how to use DataLoader with multiple workers` → `elastic/multiprocessing.rst` → solo el header `"Starting Multiple Workers"`

**Fallo diagnosticado:** el péndulo se invirtió. Antes los chunks largos ganaban; ahora los chunks ultra-cortos ganan porque su densidad léxica es perfecta cuando hay match. TF-IDF no es el problema — el problema es el corpus. Al inspeccionar la distribución:

min: 1, max: 611
chunks con menos de 10 palabras: 5483 (50%)
chunks con menos de 5 palabras: 2730 (25%)

La mitad del corpus son fragmentos basura: headers sueltos, líneas de código huérfanas, items de lista de una palabra. Ningún algoritmo de ranking puede recuperar información útil de un corpus donde el item promedio es un título sin cuerpo.

**Concepto que esto destapó:** la estrategia de chunking importa tanto o más que el algoritmo de búsqueda.

**Lección transferible:** antes de tunear un algoritmo, inspecciona la distribución de los datos que recibe. El 80% de los problemas de ML en producción se ven mirando histogramas y outliers, no leyendo papers.

### Iteración 3 — Chunking por fusión de párrafos cortos

**Qué cambió:** en vez de filtrar los párrafos cortos, se acumulan hasta alcanzar un tamaño mínimo. El bucle recorre los párrafos en orden, mantiene un buffer acumulado, y cierra el chunk cuando supera las 50 palabras.

**Stats del corpus:** 248 archivos → 2,912 chunks (de 10,970). Min 9, max 611. Solo 46 chunks por debajo de 50 palabras (residuos del último buffer al final de cada archivo — bug conocido, no crítico).

**Queries de prueba:**

- `what is autograd` → `notes/extending.rst`, fragmento introductorio de la sección "Extending torch.autograd"
- `tensor cuda gpu` → `get_start_xpu.rst`, sección "Minimum Code Change" con ejemplo de `tensor.to("cuda")` y `.to("xpu")`
- `how to use DataLoader with multiple workers` → `notes/randomness.rst`, párrafo sobre DataLoader y reseed de workers

**Diagnóstico:** mejora clara en DataLoader (pasó de RPC irrelevante a contenido realmente sobre DataLoader). Mejora ambigua en `tensor cuda gpu` (contiene las tres palabras y código real, aunque sea del contexto de migración a Intel XPU). **Sin mejora estructural** en `what is autograd` — el ganador sigue siendo `extending.rst` en vez del archivo que define qué es autograd. Más contexto alrededor, mismo sesgo de fondo.

**Fallo diagnosticado:** TF-IDF no entiende que `"what is X"` es una pregunta que pide una definición. Compara palabras, no intenciones. Arreglar el chunking no resuelve esto — es un techo estructural de la búsqueda léxica.

**Concepto que esto destapó:** hace falta un sistema que represente el **significado**, no solo las palabras.

### Iteración 4 — Embeddings semánticos

**Qué cambió:** reemplazo del `TfidfVectorizer` por el modelo pre-entrenado `all-MiniLM-L6-v2` de `sentence-transformers`. Cada chunk y cada query se convierten en vectores densos de 384 dimensiones. Ranking sigue siendo `cosine_similarity`.

**Queries de prueba (mismas tres):**

| Query                                         | TF-IDF (iter. 3)                                           | Embeddings (iter. 4)                                                                                                         |
| --------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `what is autograd`                            | `notes/extending.rst` — intro a "Extending torch.autograd" | `notes/autograd.rst` — _"Autograd is a reverse automatic differentiation system. Conceptually, autograd records a graph..."_ |
| `tensor cuda gpu`                             | `get_start_xpu.rst` — migración CUDA → XPU                 | `distributed.md` — párrafo sobre unpickling de tensores y activación de contexto CUDA en GPU                                 |
| `how to use DataLoader with multiple workers` | `notes/randomness.rst` — reseed de workers                 | (pendiente de correr)                                                                                                        |

**Diagnóstico:**

El caso de `what is autograd` muestra la diferencia fundamental. TF-IDF eligió el chunk que repite "autograd" más veces (`extending.rst`). Embeddings eligió el chunk que **define** qué es autograd, aunque la palabra aparezca menos veces, porque el modelo entiende que _"X is a Y"_ es la forma canónica de responder a _"what is X"_. El chunk ganador de embeddings ni siquiera tiene las palabras "what" o "is" en el encabezado — fue elegido por significado, no por léxico.

**Costo observado:** primera corrida 22.4s, corridas subsiguientes ~9.9s (con modelo cacheado). Cada query recalcula los embeddings de los 2,912 chunks desde cero. Inviable como producto real.

**Concepto que esto destapa para v1:** hace falta calcular los embeddings **una sola vez**, persistirlos en disco, y cargarlos para cada query. Eso es un **vector store** (FAISS, ChromaDB, etc.).

### Resumen del arco de v0

En una sola sesión, el pipeline recorrió los primeros 30 años de Information Retrieval:

1. **Grep tonto** → matchear palabras es básicamente azar en cuanto el corpus crece.
2. **TF-IDF** → ponderar por rareza ayuda, pero premia densidad léxica y choca con el corpus basura.
3. **Chunking por fusión** → la estructura del corpus importa tanto como el algoritmo.
4. **Embeddings semánticos** → el significado vive en un espacio distinto al de las palabras.

Cada salto fue forzado por un fallo concreto del paso anterior, no por una decisión de "agregar una feature". Ese es el método que define el resto del roadmap.

**Archivos resultantes:**

- `finder.py` — pipeline con chunking por fusión + TF-IDF (iteración 3).
- `finder_embeddings.py` — mismo pipeline con embeddings semánticos (iteración 4).

**Bugs conocidos de v0, no resueltos intencionalmente:**

- 46 chunks residuales de menos de 50 palabras al final de cada archivo. Inofensivos en la práctica; dejados como recordatorio de que todo corpus tiene ruido estructural.
- No hay caché de embeddings. Cada query reencodea todo. Este es el dolor que justifica v1.

---

## Version 1 — Vector store + generación (en progreso)

Objetivo: convertir DocuMentor de un **retriever** en un verdadero sistema **RAG** (Retrieval Augmented Generation).

Dos cambios planeados, en orden:

1. **Vector store persistente.** Calcular los embeddings una sola vez, guardarlos en disco, cargarlos por query. Candidatos: FAISS (más rápido, sin servidor) o ChromaDB (más features, un poco más pesado). Meta: bajar la latencia por query de ~10s a <1s.

2. **Generación con LLM.** Hasta ahora, DocuMentor devuelve el chunk crudo. v1 le pasa el chunk a un LLM con un prompt tipo _"responde la pregunta del usuario usando solo este contexto"_, y devuelve la respuesta sintetizada. Eso es literalmente la "G" de RAG.

Fallo esperado que va a disparar v2: el LLM va a alucinar cuando el chunk recuperado no contenga la respuesta, o va a responder con confianza cosas que contradicen el corpus. Ahí nacen **evaluation** y **guardrails** (Fases 4 y 5 del roadmap).
