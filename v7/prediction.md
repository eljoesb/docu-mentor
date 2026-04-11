# v7.0 — Prediccion pre-comprometida

**Fecha:** 2026-04-11

## Cambio unico

Introducir ciclo de clarificacion post-retrieval: despues de que v6 genera una respuesta de rechazo ("I don't know"), un detector decide si pedir clarificacion al usuario. Si si, genera una pregunta; si el usuario responde, re-genera con los mismos chunks + contexto del usuario. Sin re-retrieval (Pieza D diferida a v7.1).

## Arquitectura del detector (Pieza A) — comprometida antes del eval

**Trigger:** La respuesta inicial contiene rechazo explicito ("I don't know based on the provided context" o similar).

**Clasificador:** Segundo LLM call (mismo llama3.1:8b) con el siguiente prompt fijo:

```
Given the following query, retrieved contexts, and system response, classify why the system could not answer:

(A) DISAMBIGUATION: The contexts contain information related to the query, but the query might mean something different from what the system interpreted. A clarification question to the user could help produce a useful answer from the SAME contexts already retrieved.

(B) ABSENCE: The contexts simply don't contain the information needed to answer the query, even if the topic is partially related. Clarification would not help because the specific information is missing from the documentation.

(C) OUT_OF_DOMAIN: The query is about a topic completely outside the scope of the technical documentation. The rejection is correct and no clarification is needed.

Query: {query}
Contexts: {contexts}
Response: {response}

Classification (A, B, or C):
```

**Regla:** Si clasificacion = A → generar pregunta de clarificacion (Pieza B). Si B o C → retornar respuesta original sin cambios.

**Nota de integridad:** Este prompt de clasificacion esta fijado antes del eval. No se ajusta post-hoc. Si produce falsos positivos o negativos, eso se documenta en el README post-eval como dato del detector, no se corrige y re-mide.

## User context simulados — comprometidos antes del eval

Los siguientes textos se usan en la metrica secundaria (segundo pase con clarificacion simulada). Estan fijados antes del eval.

**Q21 — "how to update learning rate in middle of training"**
```
user_context: "I want to use a learning rate scheduler that adjusts the rate between training epochs, not manually change it mid-batch."
```
Justificacion: es lo que un usuario diria si le preguntan "do you mean between epochs or mid-batch?" Los chunks de optim.md ya tienen scheduler.step() con ExponentialLR.

**Q18 — "my model weights are not changing during training"**
```
user_context: "My training loop computes loss and calls loss.backward(), but I'm not calling optimizer.step() afterward."
```
Justificacion: respuesta diagnostica a una pregunta de triage. Sesgo documentado: yo se que optimizer.step() faltante es la causa mas comun. Un usuario real podria dar una respuesta mas vaga.

**Q7 — "how do I stop my model from memorizing the training data"**
```
user_context: "Yes, I want techniques to prevent overfitting like dropout or weight decay."
```
Justificacion: confirma que memorizing = overfitting. Sesgo documentado: requiere que el modelo ya haya entendido el jargon mismatch para formular la pregunta que produce esta respuesta. El user_context es favorable.

**Sesgo general:** Los tres user_context estan escritos con conocimiento del corpus y de que chunks existen. Miden el caso favorable, no el caso real. Documentado en v7/decisions.md.

## Componente 1 — Metrica primaria (discriminacion del triage)

### Queries que activan el trigger (rechazo en respuesta): 12 de 21

| Query | Util en v6 | Rechazo? | Va al clasificador? |
|-------|-----------|----------|-------------------|
| Q1-Q3, Q8-Q10, Q13, Q20 | u | No | No (8 queries) |
| Q15 | p | No ("Yes, you can") | No |
| Q4, Q5, Q6, Q16, Q17 | u (rechazo correcto) | Si | Si (5 queries) |
| Q7, Q11, Q12, Q14, Q18, Q19, Q21 | p | Si | Si (7 queries) |

### Prediccion del clasificador por query

**Verdaderos positivos (pide clarificacion correctamente):**

| Query | Confianza | Senal en la respuesta |
|-------|-----------|----------------------|
| Q21 | Alta | Respuesta describe mismatch especifico: "only at the end of each epoch or using specific schedulers like SWALR" vs "middle of training". Pattern claro de literalismo semantico — chunks tienen contenido cercano que el modelo interpreto demasiado estrecho. |

**Falsos negativos (no pide clarificacion pero deberia):**

| Query | Confianza | Por que falla |
|-------|-----------|---------------|
| Q18 | Media | Respuesta dice "none of them directly address the issue." Suena como ausencia, no disambiguation. Los chunks son sobre optimizer weight preservation / weight init / averaged model — tematicamente tangenciales. El clasificador ve ausencia donde hay oportunidad diagnostica. |
| Q7 | Alta | Respuesta dice "don't directly address memorizing training data." Suena como ausencia. Los chunks son sobre gradient history y eval/train mode — no dan senal de que el usuario quiere regularizacion. El clasificador no puede inventar la conexion memorizing→overfitting. |

**Verdaderos negativos (no pide clarificacion correctamente):**

| Query | Confianza | Razon |
|-------|-----------|-------|
| Q4 (France) | Alta | Claramente fuera de dominio. Clasificador dice C. |
| Q5 (cookies) | Alta | Claramente fuera de dominio. Clasificador dice C. |
| Q16 (database) | Alta | Dominio diferente (base de datos, no ML). Clasificador dice C o B. |
| Q11 (F.linear) | Alta | Respuesta dice "doesn't explicitly describe the functionality." Ausencia clara. Clasificador dice B. |
| Q12 (save/load) | Alta | Respuesta dice "do not explicitly cover how to save and load." Ausencia clara. Clasificador dice B. |
| Q19 (loss NaN) | Media-alta | Respuesta dice "do not address the specific issue of NaN." Ausencia. Ademas Q19 especula post-rechazo, lo cual refuerza que el modelo NO ve disambiguation sino ausencia + tentacion de adivinar. Clasificador dice B. |

**Riesgo de falsos positivos (pide clarificacion cuando no deberia):**

| Query | Riesgo | Util en v6? | Por que podria fallar |
|-------|--------|-------------|---------------------|
| Q6 (GANs optimizer) | Medio | Si (u) | Respuesta menciona "training runs, inference optimizers, mobile training" — chunks tienen contenido de optimizacion. Clasificador podria interpretar "chunks have optimizer info, user wants GANs specifically → disambiguation." Si se activa, Q6 pasa de u→p = REGRESION. |
| Q17 (pytorch vs tf) | Bajo-medio | Si (u) | Respuesta menciona "general information about PyTorch, ecosystem." Chunks tienen info general de PyTorch. Clasificador podria ver "clarify what aspect of comparison?" Pero la pregunta del usuario es clara — quiere una comparacion que no esta en la docs. |
| Q14 (flatten) | Medio | No (p) | Respuesta menciona "discuss flatten and unflatten but not for linear layers." Chunks tienen la palabra "flatten" pero no el how-to. Clasificador podria ver disambiguation. Pero Q14 ya era parcial → no es regresion en utility. |

### Numeros comprometidos — Componente 1

| Metrica | Prediccion |
|---------|-----------|
| Verdaderos positivos | 1 (Q21) |
| Falsos negativos | 2 (Q18, Q7) |
| Verdaderos negativos | 9 (Q4, Q5, Q11, Q12, Q16, Q19 + Q1-Q3, Q8-Q10, Q13, Q15, Q20 no pasan trigger) |
| Falsos positivos | 0-1 (Q6 es el mas riesgoso) |
| Precision | 1/1 o 1/2 (si Q6 o Q14 se activan) |
| Recall | 1/3 (solo Q21 de las 3 candidatas) |

## Componente 2 — Metrica secundaria (utilidad post-clarificacion)

Para las queries donde el detector acierta (pide clarificacion en candidata):

| Query | Chunks originales | User context | Mejora a util? | Confianza |
|-------|-------------------|-------------|---------------|-----------|
| Q21 | optim.md: scheduler.step(), ExponentialLR, epoch loop | "scheduler between epochs, not mid-batch" | **Si** | Alta. Los chunks YA TIENEN la respuesta. El user_context desambigua "middle of training" → "between epochs." Re-generation con la misma prompt + contexto del usuario deberia producir respuesta util basada en Context 1 (scheduler.step() despues del optimizer) y Context 3 (scheduler epoch loop). |
| Q18 (si se activa — improbable) | optim.md: weight preservation, weight init, averaged model | "not calling optimizer.step()" | **No** | Alta. Los chunks no mencionan optimizer.step() como paso del training loop. Sin re-retrieval, agregar el user_context al prompt no produce nueva informacion — el modelo sabe que falta step() pero los chunks no confirman. |
| Q7 (si se activa — improbable) | faq.rst: gradient history, autograd.rst: eval/train | "prevent overfitting with dropout/weight decay" | **No** | Alta. Los chunks no tienen contenido sobre dropout ni weight decay. Sin re-retrieval, el user_context orienta al modelo pero no le da material para responder. |

### Numero comprometido — Componente 2

Queries que mejoran de p→u con clarificacion: **1 (Q21)**.

## Componente 3 — Regresiones

### Queries utiles en riesgo de regresion

5 queries utiles activan el trigger (rechazo correcto de out-of-domain):

| Query | Score top-1 | Riesgo FP | Impacto si FP |
|-------|-------------|-----------|---------------|
| Q4 (France) | 0.3877 | Muy bajo | u→p (-1) |
| Q5 (cookies) | 0.5127 | Muy bajo | u→p (-1) |
| Q6 (GANs) | 0.6602 | Medio | u→p (-1) |
| Q16 (database) | 0.6557 | Bajo | u→p (-1) |
| Q17 (pytorch vs tf) | 0.7648 | Bajo-medio | u→p (-1) |

### Prediccion de regresiones

**Escenario mas probable: 0 regresiones.** El clasificador maneja correctamente Q4/Q5/Q16 (fuera de dominio obvio) y Q17 (pregunta clara que no se desambigua). Q6 es el riesgo real pero la respuesta describe "optimizer types" no presentes, que suena mas a ausencia que a disambiguation.

**Peor caso: 1 regresion (Q6).** Si el clasificador ve Q6 como "chunks have optimizer content, user wants specific optimizer" → disambiguation falsa. Impacto: -1 en utility.

**Numero comprometido: 0 regresiones.** Si Q6 regresiona, es informacion sobre los limites del clasificador 8b para distinguir "tema cercano" de "desambiguacion."

## Componente 4 — Categorica

### Numeros finales comprometidos

| Metrica | v6 | v7.0 pred | Delta |
|---------|-----|-----------|-------|
| Utility util | 13/21 | **14/21** | +1 |
| Utility parcial | 8/21 | 7/21 | -1 |
| Utility danino | 0/21 | 0/21 | 0 |
| Generation fiel | 20/21 | 20/21 | 0 |
| Generation mixto | 1/21 | 1/21 | 0 |

### Calibracion

| Version | Prediccion | Real | Error |
|---------|-----------|------|-------|
| v3 | 14/21 | 11/21 | -3 (optimista) |
| v5.1 | 11-12 | 9 | -2 a -3 (optimista) |
| v6 | 13/21 | 13/21 | 0 |

Patron: predictor historizamente optimista por 2-3 puntos. v6 acerto exacto pero por compensacion.

Para v7.0 el cambio predicho es +1. Aplicar correccion historica de -2 llevaria a 12/21, que es peor que v6 (13) y significaria que el ciclo de clarificacion introduce regresiones netas. No creo que eso sea probable: las 13 queries utiles sin rechazo no se ven afectadas, y las 5 con rechazo correcto deberian clasificarse como B o C.

**No aplico correccion historica** porque: (1) el cambio es de +1, no +4 como en v3; (2) el mecanismo es bien entendido (Q21 tiene los chunks, solo falta desambiguar); (3) el riesgo principal es de regresion por falsos positivos, no de optimismo en mejora.

**Prediccion calibrada final: 14/21 util.**

### VERDE

v7.0 mejora el criterio de perros en la nieve: 14/21 util (>= 13), 0 daninos (<= 3).

**Condicion de falsacion:** Si Q21 no mejora (detector no la detecta O re-generation falla con el user_context), utility queda en 13/21 y la categorica es VERDE TECNICO (empata v6 pero el experimento fallo en su objetivo).

**Condicion de alarma:** Si hay 1+ regresion en queries utiles (falso positivo en Q4/Q5/Q6/Q16/Q17), utility baja a 12-13/21 y la categorica es ROJA (el ciclo de clarificacion hizo mas daño que beneficio).

## Resumen de compromisos pre-eval

| Compromiso | Valor |
|------------|-------|
| Prompt del clasificador | Fijado arriba (A/B/C) |
| User context Q21 | "scheduler between epochs, not mid-batch" |
| User context Q18 | "not calling optimizer.step()" |
| User context Q7 | "prevent overfitting with dropout/weight decay" |
| Umbral de score | No aplica (el detector no usa scores, usa clasificador LLM) |
| Target utility | 14/21 |
| Categorica | VERDE |
| Queries que mejoran | Q21 (unica) |
| Regresiones esperadas | 0 |

**Este commit es la prueba de que la prediccion existio antes que los resultados.**
