# DocuMentor

Asistente de consulta sobre documentación técnica. Dada una pregunta en lenguaje natural sobre una librería, devuelve una respuesta sintetizada usando la documentación oficial como fuente de verdad.

Proyecto guía del roadmap de transición a AI/ML Engineer. Evoluciona por versiones, donde cada versión nace del fracaso concreto de la anterior. La intención no es construir la mejor versión de una; es **sentir el dolor específico** que justifica el siguiente concepto del roadmap, y medir el sistema antes de cambiarlo.

**Corpus actual:** documentación oficial de PyTorch (`docs/source` del repo `pytorch/pytorch`), 248 archivos `.rst` y `.md`.

**Stack v1:** Python + scikit-learn + sentence-transformers + Ollama (llama3.2:3b local).

---

## Version 0 — Retrieval léxico y semántico sobre docs de PyTorch

Objetivo de la v0: construir el pipeline más tonto posible que reciba una pregunta y devuelva un chunk relevante. Sin LLM, sin frameworks, sin bases de datos vectoriales. Solo Python, `scikit-learn` y `sentence-transformers`.

El recorrido atravesó cuatro iteraciones, cada una disparada por un fallo concreto de la anterior.

### Iteración 1 — Ranking por conteo de palabras

**Qué se construyó:** chunking por párrafos (split por `\n\n`) y ranking contando cuántas palabras de la query aparecen en cada chunk.

**Stats del corpus:** 248 archivos → 10,970 chunks.

**Fallo diagnosticado:** las palabras comunes ("to", "is", "the") dominan el ranking. Los chunks largos ganan injustamente porque acumulan más coincidencias. El ganador no es el más relevante, es el que tiene más palabras genéricas.

**Concepto que esto destapó:** hace falta ponderar por rareza (IDF) y normalizar por longitud.

### Iteración 2 — TF-IDF

**Qué cambió:** reemplazo del ranking tonto por `TfidfVectorizer` de scikit-learn + `cosine_similarity`. El chunking se mantuvo igual.

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

**Fallo diagnosticado:** TF-IDF no entiende que `"what is X"` es una pregunta que pide una definición. Compara palabras, no intenciones. Arreglar el chunking no resuelve esto — es un techo estructural de la búsqueda léxica.

**Concepto que esto destapó:** hace falta un sistema que represente el **significado**, no solo las palabras.

### Iteración 4 — Embeddings semánticos

**Qué cambió:** reemplazo del `TfidfVectorizer` por el modelo pre-entrenado `all-MiniLM-L6-v2` de `sentence-transformers`. Cada chunk y cada query se convierten en vectores densos de 384 dimensiones. Ranking sigue siendo `cosine_similarity`.

**Diagnóstico:** el caso de `what is autograd` mostró la diferencia fundamental. TF-IDF eligió el chunk que repite "autograd" más veces (`extending.rst`). Embeddings eligió el chunk que **define** qué es autograd (`notes/autograd.rst`), aunque la palabra aparezca menos veces, porque el modelo entiende que _"X is a Y"_ es la forma canónica de responder a _"what is X"_. El chunk ganador de embeddings fue elegido por significado, no por léxico.

**Costo observado:** primera corrida 22.4s, corridas subsiguientes ~9.9s. Cada query recalculaba los embeddings de los 2,912 chunks desde cero. Inviable como producto real — el siguiente dolor estructural.

### Resumen del arco de v0

En una sola sesión, el pipeline recorrió los primeros 30 años de Information Retrieval:

1. **Grep tonto** → matchear palabras es básicamente azar en cuanto el corpus crece.
2. **TF-IDF** → ponderar por rareza ayuda, pero premia densidad léxica y choca con el corpus basura.
3. **Chunking por fusión** → la estructura del corpus importa tanto como el algoritmo.
4. **Embeddings semánticos** → el significado vive en un espacio distinto al de las palabras.

Cada salto fue forzado por un fallo concreto del paso anterior, no por una decisión de "agregar una feature".

---

## Version 1 — Pipeline RAG completo + evaluación sistemática

Objetivo de v1: convertir DocuMentor de un retriever en un sistema RAG completo (Retrieval Augmented Generation), y **medir sistemáticamente su comportamiento** antes de intentar mejorarlo.

v1 se dividió en tres bloques: caché persistente, generación con LLM local, y evaluación formal.

### Bloque 1 — Vector store persistente (caché de embeddings)

**Problema que resolvió:** cada corrida de v0 recalculaba los 2,912 embeddings desde cero, tardando ~10 segundos por query. Inviable.

**Solución:** cachear los embeddings en disco con `pickle`, junto con un hash MD5 del corpus para invalidación automática. Si el hash cambia (porque se agregaron archivos, cambió el chunking, etc.), el caché se regenera sin intervención manual.

**Resultados:**

| Ruta           | Tiempo embeddings | Tiempo query |
| -------------- | ----------------- | ------------ |
| Caché frío     | 14.15s            | 0.06s        |
| Caché caliente | 0.00s             | 2.84s        |

**Observación inesperada:** con el caché caliente, el cuello de botella residual de ~2.8s **no era cargar embeddings** — era cargar el modelo `SentenceTransformer` en memoria para encodear la query. Medición precisa: 2.96s de carga del modelo, 0.17s de trabajo útil. El 95% del costo en caché caliente era startup del modelo, no trabajo real.

**Decisión:** ignorar la optimización del startup por ahora. En cuanto entre Ollama en el bloque 2, el LLM va a dominar la latencia (5-15s por query) y los ~3s del startup se vuelven ruido irrelevante. Optimización prematura evitada.

**Lección transferible:** cuando una métrica no mejora como esperabas, rastrea el costo residual antes de asumir que el fix no funcionó. El 50% de performance engineering es saber qué está medido y qué no.

### Bloque 2 — Generación con Ollama (llama3.2:3b local)

**Problema que resolvió:** v0 devolvía el chunk crudo. El usuario tenía que leer texto técnico en bruto. Un RAG real sintetiza la respuesta usando el contexto recuperado.

**Solución:** después del retrieval, el chunk top-1 se inyecta en un prompt estándar ("_Use ONLY the following context to answer..._") y se envía a Ollama corriendo localmente. La respuesta del LLM reemplaza al chunk como output.

**Decisión de stack:** Ollama + llama3.2:3b en vez de una API cloud. Razones:

- Costo marginal cero por iteración (permite experimentar sin fricción económica).
- Coherencia con el propósito del proyecto (asistente sobre documentación local).
- Restricciones de hardware explícitas desde el inicio, preparando terreno para Fase 6 del roadmap (Edge AI).
- Modelo chico (3B parámetros) deliberadamente para sentir los trade-offs de tamaño.

**Observación sobre comportamientos cold/hot de Ollama:** la primera llamada al LLM tardaba 16s (cold start, carga del modelo en memoria). Llamadas subsiguientes ~2s (hot). Ollama descarga el modelo de memoria después de ~5 minutos sin uso. Este patrón cold/hot define cómo se comporta el sistema en CLI (siempre cold) vs. servidor persistente (siempre hot).

### Bloque 3 — Evaluación sistemática de v1

Después de probar el sistema a ojo con siete queries iniciales, quedó claro que los juicios subjetivos no eran suficientes. El caso decisivo fue la query `how do I stop my model from memorizing the training data` — una pregunta sobre overfitting en jerga no técnica. El sistema recuperó un chunk sobre `accumulate history` (memory leak) y el LLM respondió con confianza que la solución era llamar a `.detach()`. Técnicamente fluido, completamente equivocado, imposible de detectar sin leer el chunk.

Ese fallo — _confidently wrong_ — forzó una pregunta más seria: **¿cuántas de las respuestas aparentemente buenas del sistema están aciertando por las razones equivocadas?**. Para responder eso hacía falta medir sistemáticamente.

#### Construcción del dataset de evaluación

Se construyó un dataset manual de **21 queries etiquetadas** en formato JSONL (`eval_dataset.jsonl`), distribuidas por categoría:

- `core_concept`: 4 queries (definiciones, "what is X")
- `how_to`: 7 queries (tareas procedurales)
- `out_of_domain`: 3 queries (fuera del corpus, el sistema debe rechazar)
- `opinion`: 2 queries (pide recomendaciones, los docs no opinan)
- `jargon_mismatch`: 3 queries (síntomas en lenguaje no técnico)
- `ambiguous`: 2 queries (respuesta depende del contexto no dado)

**Proceso de curación:** 7 queries fueron baseline inventadas durante el testing inicial de v1. Las otras 14 fueron extraídas de hilos reales de Stack Overflow y PyTorch Discuss, filtradas con un checklist estricto:

1. ¿Es sobre PyTorch puro, sin mezcla con librerías externas (Lightning, HuggingFace, etc.)?
2. ¿La respuesta aceptada usa APIs nativas de PyTorch documentadas?
3. ¿La pregunta tiene sentido sin ver código específico del usuario?
4. ¿La respuesta es factual o es opinión?

La tasa de aceptación al dataset fue aproximadamente 1 de cada 3-5 hilos mirados. La mayoría fue descartada porque mezclaban con Lightning o porque la respuesta requería conocimiento fuera de los docs.

**Decisión crítica de diseño:** no se usó ningún LLM para generar queries ni etiquetas. Toda la curación fue manual. La razón es que usar un LLM para construir un eval contra el mismo LLM (o un LLM similar) introduce _evaluation contamination_ — el dataset resultante está sesgado hacia queries que el LLM puede responder, y el eval muestra resultados falsamente optimistas. Este es uno de los modos de falla más comunes de evaluación de RAG en la literatura reciente y se evitó deliberadamente.

#### Metodología: predicción antes de medir

Antes de correr el eval, se hizo una predicción explícita: _"creo que 14 de 21 queries van a dar resultado bueno"_, con clasificación por query en verdes/amarillas/rojas e hipótesis específicas y falsables para los casos rojos.

La razón de predecir antes de medir: la diferencia entre expectativa y resultado es donde vive el aprendizaje calibrado. Sin predicción, los números del eval se racionalizan post-hoc y no enseñan nada sobre dónde el modelo mental del sistema está mal calibrado.

#### Script de evaluación

Se construyó `evaluate.py` que carga el dataset, corre cada query contra el pipeline completo (retrieval + generation), y guarda los resultados en `eval_results.jsonl` con: chunk recuperado, score de retrieval, respuesta del LLM, tiempos por etapa. Escritura incremental (append por query) para no perder datos si el script crashea a mitad de corrida.

**Decisión explícita:** el script **no automatiza el juicio**. No compara la respuesta del LLM contra ninguna respuesta esperada. No calcula accuracy. Solo instrumenta y guarda datos crudos. El juicio lo hace un humano leyendo cada resultado con un método estructurado. Automatizar el juicio requiere un juez (LLM-as-judge), y eso introduce un nuevo eje de errores no calibrado — es trabajo para Fase 4 real del roadmap, no para este paso.

#### Resultados del eval: el sistema falla donde más importa

Las 21 queries se analizaron manualmente con un método de tres ejes: **Retrieval** (¿el chunk contenía la información?), **Generation** (¿el LLM fue fiel al chunk?), **Utilidad final** (¿el usuario quedaría bien servido?).

**Conteos agregados:**

Retrieval: 3 bueno / 8 tangencial / 10 malo (14% bueno)
Generation: 8 fiel / 2 mixto / 11 inventado (38% fiel)
Utilidad: 10 útil / 5 parcial / 6 dañino (48% útil)

**Tres hallazgos estructurales:**

**Hallazgo 1: el sistema funciona por la razón equivocada.**

El retrieval acierta solo el 14% del tiempo, el LLM es fiel al contexto solo el 38% del tiempo, y aún así la utilidad final es del 48%. La pregunta obvia: ¿cómo es posible?

Respuesta: porque llama3.2:3b conoce PyTorch razonablemente bien desde su entrenamiento. Cuando el retrieval entrega un chunk malo, el LLM rescata la respuesta desde su memoria interna (documentación de PyTorch es abundante en internet) e inventa una respuesta útil **a pesar de** que el chunk sea basura. En las queries donde el LLM no tiene ese rescate (temas menos cubiertos o más específicos), el mismo patrón de retrieval roto produce respuestas dañinas.

La consecuencia práctica: **el eval no está midiendo la calidad del sistema RAG. Está midiendo la intersección entre el sistema y el conocimiento pre-entrenado del LLM.** Si mañana se cambiara el corpus a documentación interna de una empresa (que ningún LLM vio durante entrenamiento), la tasa de utilidad caería probablemente al 15-20%, no al 48%. La nieve desaparecería y quedaría el lobo real.

Este es un caso textbook del problema de los _"perros en la nieve"_: el sistema aparenta funcionar por las métricas de caja negra, pero internamente está funcionando por una señal espuria (conocimiento del LLM) en vez de por la señal que se supone está midiendo (calidad del retrieval).

**Hallazgo 2: el retrieval es el cuello de botella, no el LLM.**

Solo 3 de 21 chunks fueron buenos. Reemplazar llama3.2:3b por un modelo más grande (llama3.1:8b, GPT-4, Claude) probablemente subiría la utilidad del 48% al 65-70%, pero no por mejor retrieval — por mejor rescate desde entrenamiento. El retrieval seguiría fallando 86% del tiempo, ahora solo más escondido bajo un LLM más capaz.

Este insight contradice la intuición popular de _"un LLM mejor arregla todo"_. No es cierto. Un LLM mejor esconde mejor los problemas, pero el ceiling estructural del sistema lo pone el retrieval.

**Hallazgo 3: el sistema es bueno en lo fácil y malo en lo difícil.**

Por categoría, la tasa de utilidad fue:

- `out_of_domain`: 100% (3/3) — el sistema rechaza correctamente casos obvios.
- `core_concept`: 50% (2/4) — queries simples con chunks específicos, 50/50.
- `how_to`: 57% (4/7) — tareas procedurales, el LLM rescata lo que el retrieval no encuentra.
- `opinion`: 50% (1/2) — una rechazada bien, otra el LLM se lanzó a opinar (query 17 `pytorch vs tensorflow`).
- `jargon_mismatch`: 33% (1/3) — el modo de falla más grave y el más común en producción real.
- `ambiguous`: 0% (0/2) — el sistema no sabe pedir clarificación, inventa una interpretación.

Las categorías donde el sistema funciona mejor (`out_of_domain`) son las menos comunes en producción. Las categorías donde fracasa (`jargon_mismatch`, `ambiguous`) son exactamente las que dominan el tráfico real de usuarios.

#### Calibración de la predicción vs. realidad

Predicción: 14 de 21 buenas. Realidad: 10 de 21 útiles. Sobrestimación de 4 queries (~19%).

Descomposición del error de predicción:

- **Verdes (8 predichas):** 6 de 8 útiles. Calibración razonable, una sorpresa real (query 11 `torch.nn.functional.linear` falló porque el retriever trajo un chunk sobre extender módulos, no sobre la función).
- **Amarillas (9 predichas):** 3 de 9 útiles. **Este fue el error mayor.** Las amarillas fueron usadas como "no sé qué va a pasar" y resultaron estar sistemáticamente sesgadas hacia malos resultados.
- **Rojas (4 predichas):** 3 de 4 fallaron como se predijo. Las hipótesis específicas fueron correctas. Solo la query 15 (custom autograd function) se salvó por rescate del LLM desde entrenamiento, no por retrieval bueno.

**Lección:** la categoría "amarilla" en predicciones es un refugio cómodo para casos donde se quiere evitar comprometerse. En sistemas con baseline imperfecto, las amarillas tienden a resolverse mal más que bien. La calibración mejora forzándose a elegir verde o roja cuando se duda, y reservando amarilla solo para casos genuinamente equiprobables.

### Artefactos producidos en v1

- `documentor.py` — pipeline RAG completo, refactorizado en funciones importables (`load_or_build_chunks`, `load_or_build_embeddings`, `answer_query`).
- `evaluate.py` — script de evaluación batch que corre el dataset contra el pipeline y guarda resultados en JSONL.
- `eval_dataset.jsonl` — 21 queries etiquetadas (7 baseline inventadas + 14 curadas de Stack Overflow/Discuss).
- `eval_results.jsonl` — resultados crudos de la corrida de eval: query, chunk, score, respuesta LLM, tiempos.
- `NOTES.md` — análisis manual query por query con método de tres ejes (retrieval/generation/utilidad). Documento vivo, refleja el proceso de pensamiento en este punto del proyecto.
- `embeddings_cache.pkl` — embeddings pre-calculados + hash del corpus para invalidación automática.

### Cierre de v1

v1 cumplió sus dos objetivos declarados:

1. **Pipeline RAG funcional end-to-end** con caché persistente, retrieval semántico, y generación local.
2. **Infraestructura de evaluación sistemática** que permite medir el sistema antes de cambiarlo.

v1 no cumplió un objetivo **no declarado** que la mayoría de implementaciones de RAG asumen silenciosamente: producir un sistema confiable. El sistema no es confiable. Lo que el eval reveló es que acierta la mitad del tiempo por las razones equivocadas, y falla de formas peligrosas (confidently wrong) en los casos más comunes de uso real.

**Lo importante no es que v1 tenga 48% de utilidad. Lo importante es que ahora se sabe por qué.**

### Lo que v1 destapó para v2

El cuello de botella estructural es **retrieval**. Atacar generación o guardrails primero sería mover los síntomas sin tocar la causa. v2 ataca retrieval directamente, con un objetivo medible:

> **Objetivo cuantitativo de v2:** subir `Retrieval: bueno` de 3/21 (14%) a al menos 10/21 (48%) contra el mismo dataset, sin cambiar el LLM, sin cambiar el prompt, sin cambiar el dataset. Si la utilidad final sube sin que suba el retrieval, el sistema sigue aciertando por la razón equivocada y el cambio no cuenta.

Esta restricción — _mejorar solo retrieval, remedir contra el mismo eval_ — es lo que convierte el eval de v1 en una **herramienta de validación**, no solo un reporte. Cualquier cambio estructural a v2 se va a justificar contra estos números, no contra impresiones.
