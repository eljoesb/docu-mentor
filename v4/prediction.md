# Prediccion: v4 — LLM-as-judge (GPT-4o-mini)

**Fecha:** 2026-04-10

## Que es v4

Meta-evaluacion. El sistema bajo prueba no es el pipeline RAG sino el juez. Se corre GPT-4o-mini como clasificador sobre los mismos datos de v3b (21 queries x 3 ejes) con los mismos criterios, sin ver las clasificaciones humanas. Se mide acuerdo.

## Prediccion de acuerdo global

**47/63** (75%).

Razonamiento: de las 63 clasificaciones, ~30 son casos claros donde cualquier evaluador razonable coincide (los OOD son malo/fiel/util, los chunks que responden directamente son bueno/fiel/util). Las ~33 restantes son borderline — chunks tangenciales, respuestas mixtas, utilidad parcial. En esos, espero ~50% de acuerdo.

Rango: 42-52 (67%-83%).

## Prediccion por eje

**Generation: mayor acuerdo (~18/21, 86%).** Es el eje mas objetivo. "El LLM uso solo info del chunk o invento?" es verificable comparando chunk vs respuesta. Los casos claros son muchos: los "I don't know" son fiel, los que agregan codigo inventado son mixto, los que ignoran el chunk son inventado. Los unicos desacuerdos esperados son en los 5 "mixto" donde la linea entre "fiel con elaboracion" y "mezcla invención" es borrosa.

**Retrieval: acuerdo medio (~15/21, 71%).** Los extremos son claros (Q1 autograd→bueno, Q4 capital de Francia→malo). Pero las 9 queries borderline que oscilaron entre v3a y v3b son exactamente las que el LLM tambien va a juzgar distinto. GPT-4o-mini probablemente sea mas consistente que yo entre sesiones, pero no necesariamente mas correcto.

**Utilidad: menor acuerdo (~14/21, 67%).** Es el eje mas subjetivo. "Parcial" es un bucket enorme — incluye "no se" honestos (Q8, Q11, Q12), respuestas con errores menores (Q15, Q21), y diagnosticos vagos (Q18, Q19). GPT-4o-mini podria clasificar varios de estos como "util" (porque la respuesta no dana) o como "danino" (porque tiene errores). La frontera parcial/util y parcial/danino es donde espero mas desacuerdo.

## Prediccion de sesgo

GPT-4o-mini sera **mas estricto** que yo en retrieval y generation, pero **mas laxo** en utilidad.

- Retrieval mas estricto: GPT-4o-mini va a leer el chunk literalmente y preguntar "esto responde la pregunta?" sin el sesgo humano de "bueno, habla del tema, le pongo bueno." Probablemente clasifique como tangencial o malo varias que yo puse como bueno.
- Generation mas estricto: va a detectar invenciones sutiles que yo deje pasar como "mixto" en vez de "inventado."
- Utilidad mas laxo: va a ser mas generoso con respuestas parciales porque un LLM tiende a valorar "la respuesta contiene informacion correcta" mas que "un usuario real quedaria satisfecho." Los humanos somos mas exigentes sobre utilidad practica.

## Regla de decision

**>=80% de acuerdo global** (>=50/63) para confiar en el juez sin clasificacion humana en versiones futuras.

Si el acuerdo es:
- **>=80% (>=50/63):** LLM-as-judge reemplaza clasificacion humana. Se usa para todas las versiones futuras con spot-checks humanos ocasionales.
- **70-79% (44-49/63):** LLM-as-judge como primer filtro, humano revisa solo los desacuerdos. Reduce trabajo humano de 63 clasificaciones a ~15.
- **<70% (<44/63):** LLM-as-judge no es confiable. Investigar por que (prompt del juez mal calibrado, eje problematico, o la tarea es inherentemente ambigua). Ajustar prompt del juez y re-probar antes de descartar.
