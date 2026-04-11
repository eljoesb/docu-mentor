# v5.1 — Prediccion pre-comprometida

**Fecha:** 2026-04-10

## Cambio

llama3.2:3b → llama3.1:8b en generation. Pipeline identico a v3b: misma retrieval (BGE-base), mismo prompt, mismo corpus. Sin retry, sin agentes, sin capas nuevas.

## Senal empirica

8b paso el test de Q7 (el fallo catastrofico): rechazo honestamente ("I don't know... the context only discusses avoiding accumulation of history during training loops... doesn't address memorization"). 3b fallo esta misma query en v1, v3a y v3b (mezclaba "accumulate history" con "memorizing"). La senal: 8b obedece la instruccion de rechazo cuando el chunk no responde la pregunta.

## Bucket 1: v3b mixto → fiel (6 queries)

Queries donde 3b mezclo info del chunk con info inventada (Q3, Q7, Q13, Q14, Q15, Q21).

Mecanismo esperado: 8b obedece mejor "use ONLY the following context" y "say so explicitly instead of adapting the context." En vez de mezclar, rechaza honestamente o extrae solo lo relevante.

De las 6: cuatro tienen retrieval bueno (Q3, Q14, Q15, Q21), una tangencial (Q7), una malo (Q13). Q7 es la que mas probablemente mejore — el test manual ya confirmo que 8b rechaza honestamente donde 3b mezclaba. Con retrieval bueno, 8b deberia extraer correctamente sin inventar → fiel. Con retrieval malo/tangencial, 8b deberia rechazar → fiel.

**Complicacion:** Q13 (numpy→tensor) en v3b era gen=mixto pero util=util — el 3b invento `torch.tensor()` (correcto) desde un chunk de sparse.rst (irrelevante). Con 8b, la invencion desaparece → generation mejora (fiel) pero utilidad baja (parcial). Mejorar la honestidad del modelo destruye aciertos por suerte.

- Intuicion: 6/6 suben a fiel (Q7 confirmado por test)
- Calibrada (sesgo historico optimista, ajusto -1): **5/6 suben a fiel**

## Bucket 2: v3b inventado → fiel o mixto (2 queries: Q10, Q19)

Queries donde 3b ignoro el chunk y fabrico la respuesta completa.

- Q10 (.size vs .shape): chunk de export.md, completamente irrelevante. 8b deberia rechazar facilmente → fiel.
- Q19 (loss nan): chunk de faq.rst, tematicamente cercano (FAQ de PyTorch sobre numerics/debugging). Riesgo: 8b mezcla en vez de inventar puro, mejorando a mixto pero no a fiel.

- Intuicion: 2/2 mejoran (Q10 → fiel, Q19 → fiel o mixto)
- Calibrada: **1/2 mejora**

## Bucket 3: v3b fiel se mantiene fiel (13 queries)

Queries donde 3b fue fiel al chunk: respondio correctamente con retrieval bueno, o rechazo honestamente con retrieval malo/tangencial.

Riesgo principal: 8b mas capaz → podria elaborar mas y cruzar la linea a mixto en chunks tangenciales. Un modelo que "conecta puntos" entre el chunk y su conocimiento previo mezcla en vez de rechazar.

Contrapeso: el test de Q7 muestra que 8b respeta la restriccion del prompt.

- Intuicion: 13/13 se mantienen fiel
- Calibrada: **12/13 se mantienen fiel**

## Prediccion categorica: cierra perros en la nieve?

Perros en la nieve: utilidad >= 13/21 AND daninos <= 3/21.

### Daninos

v3b: 2 (Q7, Q10).
- Q7: 8b rechaza honestamente (test confirmado) → danino → parcial
- Q10: 8b deberia rechazar chunk irrelevante (export.md) → danino → parcial

Prediccion: **0-1 daninos. Pasa el umbral.**

### Util

v3b: 11/21. Para 13, necesito +2 neto.

**Fuentes de mejora:**
- Q21 (lr scheduling): chunk de optim.md tiene el ejemplo exacto de ExponentialLR + scheduler.step(). 3b mezclo (invento chaining syntax, claim falso sobre cosine annealing). 8b deberia extraer fielmente → parcial → util. (+1)
- Q15 (custom autograd): chunk de extending.func.rst es sobre torch.func + autograd.Function. Si 8b extrae lo correcto sin inventar register_autograd_function → parcial → util. Pero el chunk es el caso avanzado (torch.func), no el tutorial basico — riesgo de que 8b rechace por falta de respuesta directa. (+0.5)

**Fuentes de perdida:**
- Q13 (numpy→tensor): pierde acierto por suerte. 3b invento torch.tensor() (correcto) desde sparse.rst (irrelevante). 8b rechaza → util → parcial. (-1)
- Posible regresion en 1 fiel (bucket 3, -1 calibracion). (-0.5)

**Balance neto:** +1.5 - 1.5 = 0. Rango: 11-12 util.

Para llegar a 13, necesitaria que 8b extraiga significativamente mejor de TODOS los chunks tangenciales con potencial, sin perder nada. Eso contradice la logica: un modelo que obedece mejor "use ONLY the context" rechaza chunks tangenciales en vez de extraer agresivamente de ellos.

**Paradoja:** mejorar la obediencia al prompt reduce invenciones PERO TAMBIEN reduce extracciones utiles de chunks tangenciales y aciertos por suerte. El 8b es mas honesto, pero honestidad con retrieval 7/21 bueno no alcanza para 13 util.

## Numeros comprometidos

| Bucket | Intuicion | Calibrada |
|--------|-----------|-----------|
| Mixto → fiel (6) | 6/6 | 5/6 |
| Inventado → mejora (2) | 2/2 | 1/2 |
| Fiel se mantiene (13) | 13/13 | 12/13 |
| Util total | 12 | 11-12 |
| Daninos | 0-1 | 0-1 |

## **ROJA**

v5.1 no cierra perros en la nieve. El swap a 8b mejora generation (menos inventos, mejores rechazos, mas fidelidad) pero no cierra utilidad. El cuello de botella de retrieval (7/21 bueno) pone un techo que el modelo no puede romper sin violar "use ONLY the context." Daninos pasa, util no llega.

**Este commit es la prueba de que la prediccion existio antes que los resultados.**
