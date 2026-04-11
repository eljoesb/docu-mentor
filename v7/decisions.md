# v7 — Decisiones de diseno

## Contexto

v6 cerro perros en la nieve (13/21 util, 0 daninos). De las 8 queries parciales, el analisis identifico 3 candidatas donde clarificacion al usuario podria desbloquear utilidad.

## Queries candidatas para v7

| Query | Clase de clarificacion | Fuerza | Mecanismo |
|-------|----------------------|--------|-----------|
| Q21 (learning rate mid-training) | Desambiguacion semantica | Fuerte | Chunks YA tienen la respuesta. LLM rechazo por literalismo ("middle of training" ≠ "between epochs"). Clarificacion de una linea desbloquea sin re-retrieval. |
| Q18 (weights not changing) | Diagnostico/triage | Fuerte | Query de debugging. Clarificacion tipo "estas llamando zero_grad/backward/step en orden?" estrecha el scope a contenido recuperable. Clase distinta a Q21: no es "que quisiste decir" sino "cual de estas causas es la tuya". |
| Q7 (stop memorizing) | Reformulacion de vocabulario | Moderada (asterisco) | Jargon mismatch: "memorizing" ≠ "overfitting/regularization". La clarificacion util requiere que el modelo entienda el mismatch antes de preguntar — si entiende eso, ya resolvio la mitad del problema. Candidata pero probablemente no mejora con clarificacion post-retrieval porque los chunks no dan senal de que preguntar. |

### Queries descartadas (no candidatas)

- Q11 (F.linear), Q12 (save/load), Q14 (flatten) — gap de corpus, query clara
- Q19 (loss NaN) — gap de corpus + especulacion post-rechazo
- Q15 (custom autograd) — chunks relevantes pero fallo de generation, no de ambiguedad (ver v6/ideas_futuras.md)

## Decision 1: Evaluacion hibrida

**Aceptada: opcion (a) primaria + opcion (c) secundaria sobre las 3 candidatas.**

**Metrica primaria (a):** Correr v7 contra las 21 queries normales. Exito = el agente pide clarificacion en las 3 candidatas (Q21, Q18, Q7) y NO pide en las 18 restantes. Mide discriminacion del triage.

**Metrica secundaria (c):** Para las 3 candidatas, escribir manualmente un user_context plausible y correr un segundo pase donde el agente usa ese contexto. Exito = esas queries pasan de parcial a util. Mide si la clarificacion produce respuestas mejores.

**Ventaja:** Separa dos fallas diagnosticables:
- Falla en primaria = no detecto cuando pedir clarificacion
- Falla en secundaria = detecto pero no uso bien la respuesta

**Sesgo documentado:** Los user_context son escritos por mi con conocimiento del corpus, lo que sesga los resultados hacia arriba. v7 mide el caso favorable, no el caso real.

## Decision 2: Trigger post-retrieval

**Aceptada: el agente pide clarificacion DESPUES del primer retrieval, no antes.**

Razones:
1. Cambio minimo sobre v6 — agrega un paso al final del pipeline, no redisena el principio
2. El agente tiene mas informacion para hacer una clarificacion especifica (los chunks como senal)
3. Las regresiones son mas controlables — solo se activa cuando v6 ya estaba fallando
4. Es mas testeable — puedo inspeccionar los chunks que vio el agente y entender por que decidio (o no) pedir clarificacion

Para Q21 las dos opciones funcionan. Para Q18 post-retrieval es mejor (chunks tangenciales dan material para preguntar sobre training loop). Para Q7 pre-retrieval seria mejor (chunks nunca ayudan), pero Q7 es la candidata debil.

## Decision 3: Evaluacion de Q7

**Q7 se mantiene como candidata con asterisco, no como candidata fuerte.**

El escenario realista es: Q7 probablemente no mejora con clarificacion post-retrieval porque los chunks no le dan al agente la senal de que preguntar. Si mejora, es por capacidad emergente del 8b, no por diseno del pipeline.

## Techo revisado de v7

| Escenario | Util | Delta vs v6 | Queries que cambian |
|-----------|------|-------------|-------------------|
| Realista | 15/21 | +2 | Q21 + Q18 |
| Optimista | 16/21 | +3 | Q21 + Q18 + Q7 |
| Maximo teorico | 16/21 | +3 | Fijado por corpus (5 queries sin contenido) |

El target de la prediccion de v7 deberia usar 15 como base, no 16.

**Riesgo de regresion:** Si v7 introduce falsos positivos (pide clarificacion cuando no hace falta), el techo baja porque los falsos positivos rompen queries que ya funcionaban. v7 no puede llegar a 18 o 20 — el techo esta fijado en 16 por el corpus.

## Decision 3: Scope de v7.0 — A+B+C sin D

**Aceptada: v7.0 = Pieza A (detector) + Pieza B (generador de pregunta) + Pieza C (receptor de respuesta). Sin Pieza D (re-retrieval).**

### Piezas incluidas en v7.0

- **Pieza A — Detector de "necesita clarificacion":** Mira la respuesta de v6 (post-retrieval + generation) y decide si pedir clarificacion.
- **Pieza B — Generador de pregunta:** Compone una pregunta especifica basada en la query original y los chunks recuperados.
- **Pieza C — Receptor de respuesta:** Acepta la respuesta del usuario y re-genera con los chunks originales + contexto del usuario. NO hace re-retrieval.

### Pieza D — Re-retrieval con query enriquecida (diferida a v7.1)

Cuando el usuario responde la clarificacion, v7.1 re-busca con query nueva (original + respuesta del usuario). Motivacion: Q18 necesita chunks distintos (los originales son tangenciales), y Q7 necesita vocabulario distinto ("regularization" en vez de "memorizing"). Pero re-retrieval es el cambio que fallo en v5.0 (re-search) — aunque con senal externa del usuario es conceptualmente distinto, merece medicion aislada.

### Razones para v7.0 sin D

1. **Q21 sola justifica v7.0.** Chunks correctos, solo necesita desambiguar. Si funciona, 13→14 util con experimento limpio.
2. **Analogia con v5.0 pesa.** "Conceptualmente distinto" es lo que se penso de v5.0 tambien. Disciplina de medir aislado.
3. **Diagnostico limpio.** Sin D, las unicas variables son detector, generador de pregunta, y manejo de respuesta. Si Q21 no mejora, una de esas tres fallo — testeable. Con D incluida, cuarta variable y diagnostico difuso.
4. **El tamaño del cambio no predice la leccion.** v5.1 fue "un solo cambio" (swap de modelo) y produjo el hallazgo mas importante del proyecto.

### Techo revisado para v7.0 (sin D)

| Escenario | Util | Delta vs v6 | Nota |
|-----------|------|-------------|------|
| Realista | 14/21 | +1 | Solo Q21 (chunks ya correctos, clarificacion desbloquea) |
| Optimista | 14/21 | +1 | Q18 y Q7 probablemente no mejoran sin re-retrieval |
| Con v7.1 (+D) | 15-16/21 | +2 a +3 | Q18 con re-retrieval, Q7 con asterisco |

**Target de prediccion v7.0: 14/21.** Mas modesto, mas honesto, mas facil de cumplir o fallar limpiamente.

### Diseno del detector (Pieza A) para v7.0

Heuristica simple, no clasificador:

**"Chunks con scores altos + respuesta dice 'I don't know' → candidato a clarificacion."**

- Si chunks tienen scores altos Y el LLM rechaza → algo se perdio en traduccion → clarificar
- Si chunks tienen scores bajos Y el LLM rechaza → gap de corpus → NO clarificar

Esto captura Q21 (chunks de optim.md con scores 0.65-0.66, LLM rechaza por literalismo) y deberia filtrar Q11/Q12 (chunks tangenciales con scores mas bajos).

Limitacion conocida: Q15 tambien tiene chunks con scores altos (~0.74) y respuesta vaga. Pero Q15 no dice "I don't know" — dice "Yes, you can." El detector heuristico no se activa en Q15 porque la condicion es "rechazo explicito + chunks buenos", no "respuesta vaga + chunks buenos". Esa distincion es suficiente para v7.0.

### Analisis de viabilidad del detector heuristico (pre-prediccion)

**Resultado: la heuristica de score + rechazo NO discrimina candidatas de no-candidatas.**

Distribucion de top-1 scores para queries parciales en v6:

| Query | Grupo | top1 | Rechaza? |
|-------|-------|------|----------|
| Q11 | NO-CAND | 0.7565 | Si |
| Q14 | NO-CAND | 0.7239 | Si |
| Q18 | CANDIDATA | 0.7187 | Si |
| Q12 | NO-CAND | 0.7115 | Si |
| Q21 | CANDIDATA | 0.6587 | Si |
| Q7 | CANDIDATA | 0.6488 | Si |
| Q19 | NO-CAND | 0.6341 | Si |

Los scores se solapan completamente. No hay umbral que separe los dos grupos. La condicion AND (score alto + rechazo) se cumple en ambos.

**Causa:** Los scores reflejan similitud tematica query-chunk, no si la respuesta es desbloqueable con clarificacion. Q11 tiene score alto porque los chunks mencionan torch.nn.functional — tematicamente cercanos pero sin contenido especifico. Q21 tiene score medio porque los chunks de optim.md son sobre schedulers — tematicamente cercano y con el contenido que si existe.

**Implicacion para v7.0:** El detector necesita una señal distinta a score + rechazo. Opciones a evaluar:
1. **Analisis semantico de la respuesta del LLM:** Distinguir "no se porque el contenido no esta" (Q11, Q12) de "no se porque la query es ambigua" (Q21). El LLM de v6 ya da pistas: Q21 dice "only at the end of each epoch" (señal de literalismo), Q11 dice "doesn't explicitly describe" (señal de ausencia).
2. **Segundo LLM call como clasificador:** Pasar query + chunks + respuesta a un prompt que pregunte "¿esta respuesta fallo por ambiguedad de la query o por falta de contenido?"
3. **Regla basada en la respuesta de v6:** Si la respuesta menciona contenido parcialmente relevante (Q21: "schedulers like SWALR"), es candidata. Si dice "the context doesn't mention X" (Q11), no.
4. **Aceptar falsos positivos y medirlos.** Si el detector tiene 3 verdaderos positivos y 2-3 falsos positivos, la metrica primaria reporta precision y recall por separado.

**Nota de integridad:** Este analisis se hizo sobre los datos de v6 (las mismas 21 queries contra las que se medira v7). El umbral fijo de 0.65 propuesto originalmente habria sido overfitting al benchmark. El hallazgo de que los scores no discriminan evita ese error pero introduce otro problema: cualquier detector mas sofisticado (opciones 1-3) tambien se diseña mirando estos datos. La disciplina correcta es: disenar el detector con una regla justificable a priori, documentar la regla antes del eval, y medir contra las 21 queries sin ajustar post-hoc.

## Decision 4: pendiente

Prediccion de v7. A definir.
