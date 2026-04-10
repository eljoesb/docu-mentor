# Predicción: v3 — Pipeline completo BGE + Ollama

**Fecha:** 2026-04-10

## Hipótesis global

En v2.1 mejoramos retrieval de 3/21 a 11/21 bueno cambiando MiniLM por BGE-base. La pregunta central de v3 es: **¿esa mejora en retrieval se traduce en mejor utilidad final?**

En v1 el sistema tenía 10/21 útil a pesar de solo 3/21 retrieval bueno — el LLM rescataba queries usando conocimiento pre-entrenado. El riesgo es que la mejora de retrieval sea invisible en utilidad porque el LLM ya rescataba esos casos. Los "perros en la nieve" se cierran si la utilidad sube significativamente respecto a v1, confirmando que retrieval importa para el pipeline completo.

## Baseline de referencia (v1)

| Eje | Bueno/Fiel/Útil | Medio | Malo |
|-----|------------------|-------|------|
| Retrieval | 3/21 (Q1, Q9, Q19) | 8/21 tangencial | 10/21 malo |
| Generation | 8/21 fiel (Q1, Q4, Q5, Q6, Q8, Q11, Q16, Q19) | 2/21 mixto | 11/21 inventado |
| Utilidad | 10/21 útil (Q1, Q4, Q5, Q6, Q9, Q12, Q13, Q15, Q16, Q19) | 5/21 parcial | 6/21 dañino |

## Predicción de v3 — Tres números comprometidos

### Eje 1: Retrieval — **11/21 bueno**

Predicción directa: los mismos 11 de v2.1 (Q1, Q2, Q7, Q9, Q10, Q11, Q12, Q14, Q15, Q18, Q21). El pipeline local usa el mismo modelo BGE-base, el mismo chunking, y búsqueda exacta por coseno (no ivfflat). No hay razón para que difiera.

Rango: 10–12. Un +/- 1 es posible si el orden de desempate entre chunks cambia por diferencias de precisión numérica, pero sería ruido.

### Eje 2: Generation — **14/21 fiel**

Razonamiento:
- Los 11 chunks buenos dan al LLM material correcto para trabajar. De esos 11, predigo **9 fieles** — en 2 casos el LLM podría embellecer o agregar info que no está en el chunk.
- Los 5 OOD/opinion (Q4, Q5, Q6, Q16, Q17): en v1 el LLM dijo "no sé" para Q4, Q5, Q6, Q16. Q17 inventó. Predigo **4/5 fieles** (el LLM sigue diciendo "no sé" ante chunks irrelevantes para OOD).
- Los 5 restantes (Q3-tangencial, Q8-tangencial, Q13-malo, Q19-tangencial, Q20-tangencial): predigo **1/5 fiel** — chunks tangenciales/malos tienden a inducir invención.

Total: 9 + 4 + 1 = **14/21 fiel**.

Rango: 12–16.

### Eje 3: Utilidad — **14/21 útil**

Razonamiento:
- De los 11 retrieval buenos con ~9 generation fieles: **9 útiles** directos. Los 2 donde el LLM embellece podrían ser parciales, no dañinos.
- OOD/opinion (5 queries): **4 útiles** — "no sé" ante pregunta fuera de dominio es respuesta útil.
- Los 5 restantes: **1 útil** por rescate del LLM con conocimiento pre-entrenado (como en v1 con Q12, Q13).

Total: 9 + 4 + 1 = **14/21 útil**.

Rango: 12–16.

## Hipótesis específicas (query-level)

### H1: Q7 (memorizing training data) — Retrieval bueno → Generation fiel → Útil

En v1 fue el caso más dañino: chunk de accumulating gradients, LLM recomendó `.detach()` para overfitting. En v2.1 BGE encontró `faq.rst` con contenido sobre overfitting/regularización. Predigo que con un chunk bueno, el LLM responde con la información del chunk en vez de inventar. **Los tres ejes en verde.**

Si falla generation a pesar de chunk bueno, es señal de que el LLM de 3B no sabe cuándo confiar en el chunk vs. su conocimiento pre-entrenado.

### H2: Q10 (.size vs .shape) — Retrieval bueno → Generation fiel → Útil

En v1 fue dañino: inventó que `.shape` es "deprecated". En v2.1 BGE encontró un chunk relevante. Si el chunk contiene la respuesta correcta, el LLM no debería inventar. **Los tres ejes en verde.**

Si generation inventa a pesar de chunk bueno, mismo diagnóstico que H1.

### H3: Q13 (numpy to tensor) — Retrieval malo → Generation inventado → Útil por rescate

En v2.1 retrieval fue malo (sparse.rst). El LLM de 3B sabe cómo convertir numpy a tensor (es conocimiento ubicuo). Predigo **malo / inventado / útil** — el mismo patrón de rescate que v1 donde el LLM sabía la respuesta sin necesitar el chunk.

Si retrieval mejora (improbable sin cambios), toda la hipótesis se reescribe.

## Regla de decisión

### Los perros en la nieve se cierran si:
- Utilidad ≥ 13/21 (mejora de +3 respecto a v1's 10/21)
- Y dañinos ≤ 3/21 (reducción de los 6 dañinos de v1)
- Interpretación: retrieval bueno se traduce en utilidad, el pipeline funciona end-to-end.

### Hay un problema nuevo si:
- Utilidad ≤ 11/21 a pesar de 11/21 retrieval bueno
- Interpretación: el LLM no sabe usar los buenos chunks. El cuello de botella ya no es retrieval sino generation (modelo demasiado chico, prompt mal diseñado, o el chunk necesita más contexto).

### Resultado ambiguo:
- Utilidad 12/21 — mejora modesta. No refuta la hipótesis pero no la confirma fuertemente. Habría que analizar query por query si los chunks buenos se traducen en utilidad o no.
