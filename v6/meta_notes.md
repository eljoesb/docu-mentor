# Meta-notas: progresion de la metodologia de prediccion

## v3 — Predecir numeros

La primera prediccion fue puramente cuantitativa: "14/21 queries van a ser utiles." Sin modelo de POR QUE mejorarian. Sin mecanismos. Sin condiciones de falsacion. El predictor era un estimador de punto.

Resultado: 14 predicha, 11 real. Error -3 (optimista). La direccion era obvia (primer sistema con retrieval → algo funciona), pero la magnitud estaba inflada. No habia forma de saber si el error era de retrieval, generation, o ambos.

Leccion de la prediccion: los numeros solos no enseñan nada. Acertar al numero correcto por las razones incorrectas no es calibracion — es suerte.

Leccion del analisis post-hoc: v3 enseno que el sistema funcionaba por las razones equivocadas. El analisis cualitativo de las 21 respuestas revelo que varias queries eran utiles por invenciones correctas por suerte, no por retrieval bueno — descubrimiento que llevo al concepto de "perros en la nieve" y al criterio de evaluacion del proyecto. Ese hallazgo no vino del gap prediccion-resultado sino del escrutinio detallado de los resultados.

## v4 — Predecir porcentajes (mismo problema)

v4.0 y v4.1 predijeron "75% global" y "70% global." El formato cambio (porcentaje vs conteo) pero el problema persistio: predicciones sin mecanismos, sin falsacion, sin descomposicion.

Resultado: 56% y 59% real. Siempre optimista. Ningun insight del gap sobre QUE fallo.

Leccion de la prediccion: cambiar la metrica no resuelve la falta de modelo causal.

Leccion del analisis post-hoc: v4 enseno que las reglas de evaluacion explicitas se aplican como leyes — el sistema que parecia funcionar "bien" en v3 se desmorono cuando se midio con rigor. La disciplina de medir en vez de estimar fue el hallazgo de v4, independiente de la prediccion.

## v5.0 — Predecir con subgrupos (primer intento de mecanismo)

v5.0 predijo por subgrupos: "Grupo A se mantiene, Grupo B mejora, canarios se mantienen." La estructura era mas rica — tres buckets con predicciones separadas. Pero los mecanismos eran superficiales: "re-search mejora retrieval" sin analizar POR QUE fallaria o cuando.

Resultado: Grupo B = 0/6 (prediccion: 4/6). Catastrofe. El mecanismo era incorrecto: re-search no mejoraba retrieval porque las queries reformuladas no matcheaban el vocabulario del corpus. La prediccion identifico los subgrupos pero no anticipo el modo de fallo.

Leccion: subgrupos sin modelo causal producen predicciones con estructura pero sin contenido predictivo.

## v5.1 — Predecir mecanismos (el cambio real)

v5.1 fue el punto de inflexion. La prediccion incluyo una tesis causal:

> "En un sistema RAG con retrieval imperfecto, mejorar la obediencia del modelo al prompt reduce invenciones Y aciertos por suerte simultaneamente; la utilidad neta depende de cual de los dos efectos domina."

Por primera vez, la prediccion decia no solo QUE pasaria sino POR QUE. Y incluia un trade-off explicito (mas honestidad = menos inventos PERO TAMBIEN menos aciertos por suerte) con una prediccion sobre cual efecto dominaria.

Resultado: el mecanismo fue correcto. La paradoja se confirmo en Q13 (perdio invencion correcta por suerte) y Q14 (markup blindness, nuevo modo de fallo). La direccion fue correcta en todos los ejes (generation sube, utility baja, categorica ROJA). Los numeros estuvieron mal (11-12 predichos, 9 real).

Leccion: los numeros pueden estar mal si el mecanismo esta bien — y el mecanismo enseña mas que los numeros. Descubrir la paradoja fue mas valioso que acertar el conteo.

Hallazgos emergentes que la prediccion no anticipo:
- **Markup blindness** (Q14): el 8b no parsea `{meth}~Tensor.flatten` como mencion funcional
- **Literalismo semantico** (Q21): el 8b interpreta "middle of training" como "middle of epoch"
- **Techo distribuido**: el cuello de botella no es el modelo sino retrieval + markup + literalismo

Estos hallazgos — surgidos del gap entre prediccion y resultado — son la razon por la que la metodologia funciona. Sin prediccion pre-comprometida, Q14 y Q21 se habrian clasificado como "parcial" sin investigar la causa raiz.

## v6 — Predecir trade-offs multi-objetivo con condiciones de falsacion

v6 llevo la metodologia un paso mas:

1. **Trade-off explicito de tres ramas**: la prediccion anticipo que mas contexto (top-K=3) produciria un trade-off generation-vs-utility, y categorizo tres ramas posibles (buena: +util >> -fiel; neutra: +util ≈ -fiel; mala: +util ≤ -fiel). Esto es mas que un mecanismo — es un espacio de resultados posibles con la prediccion sobre cual rama se realizaria.

2. **Per-query con mecanismo**: cada query tenia una prediccion con mecanismo especifico (intra-file, inter-file, markup blindness no atacada, contenido no existe). No solo "mejora" o "no mejora" sino "mejora porque tensors.rst tiene torch.as_tensor al principio del archivo y deberia rankear top-3 para esta query."

3. **Lista explicita de queries que NO mejoran y POR QUE**: Q14 (markup blindness, top-K no ataca), Q11 (contenido no existe en prosa), Q10 (contenido no existe), Q7 (contenido no existe). Comprometerse a lo que NO funciona es mas informativo que comprometerse a lo que funciona.

4. **Calibracion historica aplicada**: el predictor reconocio su sesgo optimista (-2 a -3 en utilidad) y ajusto -1 en vez de -2, justificando la diferencia (primera intervencion que ataca el cuello de botella real).

5. **Condicion de falsacion**: "Si 2+ de las 4 de alta confianza fallan, caigo a 11 y la categorica es ROJA." Esto convierte la prediccion en apuesta falsable.

Resultado: 13/21 util (prediccion calibrada: 13/21). Categorica VERDE confirmada. 3/4 alta confianza correctas (Q8 ✓, Q13 ✓, Q20 ✓, Q12 ✗). La condicion de falsacion no se activo (solo 1 de 4 fallo). Q10 mejoro sin estar predicha (compenso Q12). Generation mejoro (19→20 fiel) en vez de empeorar como temido — la "mezcla multi-chunk" fue over-predicted.

El error especifico: Q12 (serialization.rst no rankeo top-3). La prediccion dijo "contenido correcto existe y deberia rankear top-3" pero no verifico los embeddings. El riesgo estaba documentado ("si los chunks correctos no rankean top-3") pero la confianza era "alta" cuando debio ser "media."

La sorpresa: Q10 (.size vs .shape) mejoro via inferencia contextual — el modelo vio ambos usados y dedujo equivalencia. Esto no era predecible porque la prediccion decia "contenido no existe" (correcto — no existe explicacion explicita) pero el modelo pudo inferir la respuesta desde uso implicito. Modo de mejora no anticipado: inferencia desde uso contextual, no desde explicacion directa.

## La progresion

| Version | Tipo de prediccion | Que enseno el gap | Que enseno el analisis post-hoc |
|---------|--------------------|-------------------|-------------------------------|
| v3 | Numero de punto | — | Perros en la nieve: utilidad por suerte, no por retrieval |
| v4 | Porcentaje de punto | — | Las reglas explicitas se aplican como leyes |
| v5.0 | Subgrupos sin mecanismo | El mecanismo importa mas que la estructura | Migracion de cuello de botella |
| v5.1 | Mecanismo causal + trade-off | Paradoja honestidad/suerte; modos de fallo emergentes; techo distribuido | Tres modos de fallo bajo el mismo sintoma |
| v6 | Trade-off multi-objetivo + falsacion | Inferencia contextual como modo de mejora; mezcla multi-chunk over-predicted | Cross-vocabulary transfer failure; cuarta capa del techo |

Las versiones tempranas enseñaron via analisis post-hoc; las tardias via prediccion pre-comprometida. Ambos son validos. El segundo es mas eficiente porque fuerza a articular el modelo mental ANTES de medir, y el gap entre prediccion y resultado señala exactamente donde el modelo es incorrecto. El primero requiere escrutinio abierto sin hipotesis, que es mas lento pero descubre cosas que ninguna prediccion habria buscado.

La meta-leccion: **la prediccion es una herramienta de aprendizaje, no de adivinacion.** Lo que importa no es acertar el numero sino descubrir, en el gap entre prediccion y resultado, que modelo mental del sistema estaba incorrecto y por que. Cada version actualizo el modelo mental mas de lo que actualizo los numeros.

El predictor paso de ser un estimador de punto a ser un modelo causal falsable. Eso es progreso real en la capacidad de razonar sobre el sistema — independiente de si los numeros caen dentro del intervalo predicho.
