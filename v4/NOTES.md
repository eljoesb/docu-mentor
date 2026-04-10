# v4 — LLM-as-judge: Notas de evaluacion

## Dos corridas, un hallazgo sobre prompts de juez

v4 tuvo dos corridas:
- **v4.0** (prompt base): criterios abstractos sin ejemplos ni matices
- **v4.1** (prompt iterado): criterios explicitos + few-shot examples, calibrados contra los desacuerdos de v4.0

Mismo modelo (GPT-4o-mini), misma temperature (0), mismo max_tokens (1). Solo cambio el texto del prompt.

## Resultados numericos

### v4.0 — Prompt base

| Eje | Acuerdo | % |
|-----|---------|---|
| Retrieval | 11/21 | 52% |
| Generation | 13/21 | 62% |
| Utility | 11/21 | 52% |
| **Global** | **35/63** | **56%** |

### v4.1 — Prompt iterado

| Eje | Acuerdo | % | Delta vs v4.0 |
|-----|---------|---|---------------|
| Retrieval | 10/21 | 48% | **-4pp** |
| Generation | 16/21 | 76% | **+14pp** |
| Utility | 11/21 | 52% | **0pp** |
| **Global** | **37/63** | **59%** | **+3pp** |

### Comparacion con predicciones

| Prediccion | Esperado | Real | Error |
|------------|----------|------|-------|
| v4.0: acuerdo global | 47/63 (75%) | 35/63 (56%) | **-19pp** (optimista) |
| v4.1: acuerdo global | 44/63 (70%) | 37/63 (59%) | **-11pp** (optimista) |
| v4.1: eje que mas mejora | Utilidad (+4) | Generation (+3) | **Orden invertido** |
| v4.1: eje que menos mejora | Generation (+2) | Retrieval (-1) | **Orden invertido** |

El predictor mejoro en magnitud (error bajo de 19pp a 11pp) pero fallo completamente en el orden de ejes. Predijo utilidad > retrieval > generation; paso generation > utility = retrieval.

## Analisis de desacuerdos v4.1

### Retrieval: 11 desacuerdos (empeoro)

| Query | Humano | Juez | Patron |
|-------|--------|------|--------|
| Q6 (optimizer GANs) | tangencial | malo | t→m |
| Q7 (memorizing data) | tangencial | malo | t→m |
| Q8 (multiple GPUs) | tangencial | malo | t→m |
| Q9 (leaf tensor) | bueno | tangencial | b→t |
| Q11 (nn.functional.linear) | tangencial | malo | t→m |
| Q12 (save/load model) | tangencial | malo | t→m |
| Q14 (flatten tensor) | bueno | malo | b→m |
| Q15 (custom autograd) | bueno | tangencial | b→t |
| Q17 (pytorch vs tf) | malo | tangencial | m→t |
| Q18 (weights not changing) | tangencial | malo | t→m |
| Q19 (loss nan) | tangencial | malo | t→m |

**Patron dominante:** 7 de 11 desacuerdos son humano=tangencial → juez=malo. La regla de "overlap superficial = malo" hizo que el juez bajara todos los tangenciales a malo. El juez no distingue entre "irrelevante" y "relacionado pero no responde" — para el, si no responde, es malo.

**Diagnostico:** la regla fue demasiado agresiva. Le pedimos que clasifique overlap superficial como malo, y aplico la regla literalmente a TODO lo que no responde directamente, incluyendo chunks genuinamente tangenciales. La categoria "tangencial" quedo casi vacia (solo Q9, Q15, Q17, Q20 en el juez vs 8 en el humano).

**Leccion:** las reglas explicitas en prompts de juez se aplican como leyes, no como matices. Una regla pensada para 2-3 edge cases se aplico a 7+ casos. La solucion no es escribir reglas mas complejas — es dar ejemplos de los tres niveles (bueno, tangencial, malo) para que el modelo infiera el rango.

### Generation: 5 desacuerdos (mejoro mucho)

| Query | Humano | Juez | Patron |
|-------|--------|------|--------|
| Q3 (DataLoader workers) | mixto | fiel | x→f |
| Q7 (memorizing data) | mixto | fiel | x→f |
| Q10 (.size vs .shape) | inventado | mixto | i→x |
| Q13 (numpy to tensor) | mixto | inventado | x→i |
| Q19 (loss nan) | inventado | mixto | i→x |

**Patron:** la regla "no se honesto = fiel" funciono perfectamente. Los 8 desacuerdos de v4.0 se redujeron a 5. Los restantes son genuinos borderline en la frontera fiel/mixto y mixto/inventado.

**Q3 y Q7 (x→f):** el juez clasifica como fiel respuestas que el humano ve como mixtas. En ambos casos el LLM elabora sobre el chunk sin inventar datos nuevos — la diferencia es si "elaboracion" cuenta como mezcla. Desacuerdo de criterio, no de juicio.

**Q10 y Q19 (i→x):** el juez baja un nivel (inventado → mixto) donde el humano puso inventado. El juez ve que la respuesta mezcla info del chunk con invenciones, y clasifica como mixto. El humano ve que las invenciones son la parte dominante y clasifica como inventado. Desacuerdo de peso relativo.

**Q13 (x→i):** unico caso donde el juez es mas estricto que el humano. El humano vio mezcla (chunk de sparse + respuesta sobre numpy); el juez vio invencion pura.

### Utility: 10 desacuerdos (no se movio)

| Query | Humano | Juez | Patron |
|-------|--------|------|--------|
| Q2 (tensor to GPU) | util | parcial | u→p |
| Q7 (memorizing data) | danino | parcial | d→p |
| Q8 (multiple GPUs) | parcial | util | p→u |
| Q10 (.size/.shape) | danino | parcial | d→p |
| Q11 (nn.functional.linear) | parcial | util | p→u |
| Q12 (save/load model) | parcial | util | p→u |
| Q13 (numpy to tensor) | util | parcial | u→p |
| Q15 (custom autograd) | parcial | util | p→u |
| Q20 (view vs reshape) | parcial | util | p→u |
| Q21 (learning rate) | parcial | danino | p→d |

**Patron:** los desacuerdos cambiaron de forma pero la cantidad total no (10 en v4.0, 10 en v4.1). Los OOD que antes daban "danino" ahora dan bien (Q4 y Q5 corregidos), pero aparecieron desacuerdos nuevos.

**5 de tipo p→u:** el juez es mas generoso que el humano en la frontera parcial/util. Q8, Q11, Q12, Q15, Q20 — el juez ve respuestas que contienen informacion correcta y las clasifica como utiles. El humano ve que la informacion es incompleta o no responde exactamente lo preguntado y clasifica como parcial.

**2 de tipo u→p:** Q2 y Q13 — el juez es mas estricto que el humano. El humano ve una respuesta que resuelve la pregunta; el juez encuentra algo insuficiente.

**2 de tipo d→p:** Q7 y Q10 — el juez suaviza de danino a parcial. El humano ve respuestas confidently wrong; el juez ve respuestas que contienen algo de informacion y las baja solo a parcial.

**Q21 (p→d):** unico caso donde el juez es mas severo. El humano ve parcial; el juez ve danino. Este es nuevo en v4.1.

**Conclusion:** la frontera parcial/util es la mas borrosa de todo el sistema. 7 de 10 desacuerdos involucran esa frontera. No se resuelve con reglas — necesita ejemplos calibrados por query.

## El hallazgo central de v4

### Las reglas explicitas se aplican como leyes

El prompt de retrieval v4.1 pedia: "Si hay overlap superficial sin match semantico, clasifica como malo." Intencion: corregir 2-3 edge cases. Resultado: el juez aplico la regla a 7+ casos, eliminando casi toda la categoria "tangencial."

Esto es un patron general de prompts de juez:
- **Criterios abstractos** → el juez los interpreta con su propio sesgo (v4.0)
- **Reglas explicitas** → el juez las aplica rigidamente, mas alla de la intencion (v4.1)
- **Ejemplos few-shot** → el juez infiere el rango aceptable (lo que falta probar)

La solucion correcta para v4.2 no es mas reglas sino mas ejemplos. Tomar 2-3 queries de cada nivel (bueno/tangencial/malo) y mostrarselas al juez para que calibre.

### El predictor sobreestima y no predice el orden

Dos predicciones consecutivas optimistas:
- v4.0: +19pp arriba del real
- v4.1: +11pp arriba del real

El predictor mejora en magnitud pero no en estructura. Predijo el orden de ejes exactamente al reves en v4.1. Hipotesis: el predictor razona sobre "que desacuerdos ataco el cambio" sin modelar los desacuerdos nuevos que el cambio introduce. En retrieval, la regla corrigio 0-1 desacuerdos pero introdujo 0+ nuevos. El predictor no anticipo eso.

### Generation es el eje mas tratable por prompt engineering

Generation paso de 62% a 76% con una sola regla. Es el eje mas objetivo ("uso info del chunk o no?") y el que mejor responde a instrucciones explicitas. El juez ahora tiene 76% de acuerdo — entra en la zona "primer filtro + humano revisa desacuerdos".

Retrieval y utilidad siguen en <55% — probablemente no se resuelven solo con prompt engineering. Opciones:
1. Few-shot examples calibrados (v4.2)
2. Aceptar que estos ejes necesitan humano siempre
3. Colapsar las categorias (bueno/malo sin tangencial; util/danino sin parcial)

## Cambios al prompt entre v4.0 y v4.1

### Retrieval — agregado despues del criterio base

```
Important: If the chunk contains terms or words from the question but in a
semantically different context (e.g., "detach" used for memory management
vs "detach" for preventing overfitting), classify as m (bad), not t (tangential).
Surface word overlap without semantic match is bad retrieval.
```

Efecto: empeoro el acuerdo. El juez bajo casi todos los tangenciales a malo.

### Generation — agregado despues de las definiciones

```
Note: A response that says "I don't know based on the provided context" is
faithful (f), not mixed or invented. Honest refusal counts as faithful.
```

Efecto: mejoro el acuerdo +14pp. La regla mas exitosa de las tres.

### Utilidad — agregado despues de las definiciones + few-shot

```
Important rules:
- If the question is clearly out-of-domain (e.g., asking about cooking, geography,
  or unrelated topics) and the response honestly says "I don't know", classify as
  u (useful). Honest refusal of OOD questions is the correct behavior.
- If the response contains specific technical claims that sound plausible but are
  factually incorrect, classify as d (harmful), even if the response seems helpful.

Examples:
- Question: "what is the capital of France" / Response: "I don't know based on
  context" → u (useful, correct OOD rejection)
- Question: "difference between .size() and .shape" / Response: invents a
  difference that doesn't exist in PyTorch → d (harmful, confidently wrong)
```

Efecto: neutro. Corrigio los OOD pero introdujo desacuerdos nuevos en la frontera parcial/util.
