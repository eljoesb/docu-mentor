# Prediccion: v4.1 — LLM-as-judge con prompt iterado

**Fecha:** 2026-04-10

## Contexto

v4 original obtuvo **35/63 (56%)** de acuerdo — 19 puntos por debajo de la prediccion de 75%. El diagnostico revelo que el problema no es el juez sino el prompt: no codificaba los matices que el humano aplica implicitamente. Se iteraron los tres prompts con criterios explicitos y few-shot examples.

### Cambios aplicados al prompt
1. **Retrieval:** criterio explicito de que overlap superficial de palabras sin match semantico = malo (no tangencial)
2. **Generation:** "no se" honesto = fiel (no mixto ni inventado)
3. **Utilidad:** OOD + "no se" honesto = util (no danino); invenciones plausibles = danino. Dos few-shot examples.

## Resultado de v4 (baseline)

| Eje | Acuerdo | % |
|---|---|---|
| Retrieval | 11/21 | 52% |
| Generation | 13/21 | 62% |
| Utility | 11/21 | 52% |
| **Global** | **35/63** | **56%** |

## Prediccion de acuerdo global v4.1

**44/63** (70%).

Razonamiento: la prediccion anterior fallo por sobreestimar en 19 puntos. Corrijo en dos direcciones: (1) bajo el anclaje base, (2) estimo mejora conservadora por los cambios al prompt. Los cambios atacan directamente ~15 de los 28 desacuerdos (los OOD en utilidad, los overlap superficiales en retrieval, los "no se" en generation). Espero corregir ~9 de esos 15 (60% de fix rate — no todos porque el juez puede interpretar distinto incluso con instrucciones). 35 + 9 = 44.

Rango: 40-48 (63%-76%).

## Prediccion por eje

**Utilidad: mayor mejora (~15/21, 71%, +4).** Los cambios son los mas directos aqui. Los 3 desacuerdos OOD (Q4, Q5, Q16 donde "no se" fue clasificado como danino) deberian corregirse con el criterio explicito + few-shot. Tambien espero que el criterio de "invenciones plausibles = danino" corrija Q10 y Q19 donde el juez paso por alto errores sutiles.

**Retrieval: mejora moderada (~14/21, 67%, +3).** El criterio de overlap superficial deberia corregir 2-3 casos donde el juez fue demasiado generoso con chunks que tenian palabras coincidentes pero semantica distinta. Pero la frontera bueno/tangencial sigue siendo borrosa y ahi no agregamos ejemplos.

**Generation: mejora minima (~15/21, 71%, +2).** Los desacuerdos en generation eran mayormente ruido en la frontera fiel/mixto, no errores sistematicos. El cambio de "no se = fiel" corrige 1-2 casos pero la mayoria del ruido no se resuelve con un criterio explicito.

## Calibracion del predictor

| Prediccion | Resultado | Error |
|---|---|---|
| v4: 47/63 (75%) | 35/63 (56%) | -19pp (optimista) |
| v4.1: 44/63 (70%) | ? | ? |

Leccion de v4: tiendo a sobreestimar el acuerdo entre evaluadores. La prediccion v4.1 incorpora esta correccion bajando el anclaje y usando fix rate conservador (60% en vez de asumir que todos los desacuerdos atacados se corrigen).

## Regla de decision (sin cambios)

- **>=80%:** juez reemplaza humano
- **70-79%:** juez como primer filtro, humano revisa desacuerdos
- **<70%:** seguir iterando o aceptar que el eje requiere humano
