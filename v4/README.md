# v4 — Meta-evaluacion: LLM-as-judge (GPT-4o-mini)

## Que es v4

v4 no evalua el pipeline RAG — evalua al evaluador. Se corre GPT-4o-mini como juez automatico sobre los mismos 21 queries x 3 ejes de v3b, sin ver las clasificaciones humanas. Se mide acuerdo humano-juez para determinar si LLM-as-judge puede reemplazar o asistir la clasificacion manual.

Motivacion: en v3 se descubrio varianza intra-evaluador de ±8 queries sobre los mismos datos. La clasificacion manual es insostenible para versiones futuras.

## Cambios respecto a v3

| Componente | v3 | v4 |
|------------|----|----|
| Sistema evaluado | Pipeline RAG (BGE + Ollama) | El juez automatico |
| Datos | 21 queries nuevas end-to-end | Mismos datos de v3b |
| Evaluador | Humano (3 pasadas ciegas) | GPT-4o-mini (3 pasadas independientes) |
| Metrica | Retrieval/generation/utilidad | Acuerdo humano vs juez |

## Dos corridas: v4.0 y v4.1

### v4.0 — Prompt base

Prompt con criterios abstractos (bueno/tangencial/malo, fiel/mixto/inventado, util/parcial/danino) sin ejemplos ni matices explicitos.

### v4.1 — Prompt iterado

Tres cambios basados en el diagnostico de desacuerdos de v4.0:
1. **Retrieval:** regla de overlap superficial sin match semantico = malo
2. **Generation:** "no se" honesto = fiel
3. **Utilidad:** OOD + "no se" = util; invenciones plausibles = danino + 2 few-shot examples

## Resultados

### Acuerdo humano-juez

| Eje | v4.0 | v4.1 | Delta |
|-----|------|------|-------|
| Retrieval | 11/21 (52%) | 10/21 (48%) | **-4pp** |
| Generation | 13/21 (62%) | 16/21 (76%) | **+14pp** |
| Utility | 11/21 (52%) | 11/21 (52%) | **0pp** |
| **Global** | **35/63 (56%)** | **37/63 (59%)** | **+3pp** |

### Vs regla de decision

| Umbral | Requerido | v4.0 | v4.1 | Status |
|--------|-----------|------|------|--------|
| Juez reemplaza humano | >=80% | 56% | 59% | No alcanza |
| Juez como primer filtro | 70-79% | 56% | 59% | No alcanza |
| Requiere iteracion | <70% | 56% | 59% | **Aqui estamos** |

### Vs predicciones

| Prediccion | Esperado | Real | Error |
|------------|----------|------|-------|
| v4.0 global | 75% | 56% | -19pp |
| v4.1 global | 70% | 59% | -11pp |
| v4.1 eje que mas mejora | Utilidad | Generation | Invertido |
| v4.1 eje que menos mejora | Generation | Retrieval | Invertido |

## Tres hallazgos

### 1. Las reglas explicitas se aplican como leyes, no como matices

La regla de retrieval ("overlap superficial = malo") estaba pensada para 2-3 edge cases. El juez la aplico a 7+ casos, eliminando casi toda la categoria "tangencial." Paso de 8 tangenciales (humano) a 4 (juez).

Patron general: cuando un humano escribe un criterio, lo aplica con intuicion y contexto. Cuando un LLM recibe ese mismo criterio como instruccion, lo aplica rigidamente. Codificar criterios explicitos siempre los hace mas rigidos que la aplicacion humana original.

La solucion no es mas reglas — es mas ejemplos. Few-shot permite al modelo inferir el rango aceptable en vez de aplicar una ley binaria.

### 2. Generation es el eje mas tratable por prompt engineering

Una sola regla ("no se honesto = fiel") subio generation de 62% a 76%. Es el eje mas objetivo: "uso info del chunk o no?" tiene una respuesta verificable comparando textos. Los 5 desacuerdos restantes son genuinos borderline (frontera fiel/mixto y mixto/inventado).

Generation es el unico eje que entra en zona de "primer filtro" (76% > 70%). Retrieval y utilidad siguen en ~50%.

### 3. El predictor sobreestima sistematicamente el acuerdo

Dos predicciones consecutivas optimistas (-19pp y -11pp). El predictor razona sobre "que desacuerdos corrige el cambio" sin modelar los desacuerdos nuevos que el cambio introduce. En retrieval, la regla no solo no corrigio los desacuerdos existentes — creo nuevos.

Calibracion acumulada del predictor:

| Version | Prediccion | Real | Error | Tipo |
|---------|-----------|------|-------|------|
| v3 retrieval | 11/21 | 7/21 | -4 | Optimista |
| v3 generation | 14/21 | 14/21 | 0 | Exacto |
| v3 utilidad | 14/21 | 11/21 | -3 | Optimista |
| v4.0 global | 75% | 56% | -19pp | Optimista |
| v4.1 global | 70% | 59% | -11pp | Optimista |

Patron: el predictor siempre sobreestima, nunca subestima. El sesgo es consistente.

## Metodologia

### Pipeline del juez

1. Carga los 21 records de `v3/pipeline_results_v3b.jsonl`
2. Tres pasadas independientes (retrieval, generation, utilidad), cada una con su prompt
3. GPT-4o-mini, temperature=0, max_tokens=1
4. Output: un CSV por eje con (query_id, query, source, label)

### Comparacion

`compare.py` lee pares de CSVs (humano vs juez) y produce:
- Acuerdo por eje y global
- Matriz de confusion por eje
- Lista de desacuerdos especificos

### Evaluacion ciega

La estructura de tres pasadas se mantiene identica a v3: retrieval ve query + chunk, generation ve chunk + respuesta, utilidad ve query + respuesta. El juez nunca ve las clasificaciones humanas.

## Archivos

```
v4/
  prediction.md          — Prediccion pre-comprometida v4.0
  prediction_1.md        — Prediccion pre-comprometida v4.1
  judge.py               — Script del juez (version v4.1 con prompt iterado)
  compare.py             — Comparacion humano vs juez
  judge_retrieval.csv    — Clasificaciones del juez v4.1 (retrieval)
  judge_generation.csv   — Clasificaciones del juez v4.1 (generation)
  judge_utility.csv      — Clasificaciones del juez v4.1 (utilidad)
  NOTES.md               — Analisis detallado de desacuerdos y diagnosticos
  README.md              — Este archivo
```

## Siguiente paso

### Opcion A: v4.2 — Few-shot examples calibrados

Reemplazar las reglas explicitas (que se aplican como leyes) por ejemplos few-shot (que permiten inferir rango). Tomar 2-3 queries de cada nivel por eje, directamente de los datos de v3b. El juez ve el ejemplo clasificado y calibra.

Esto deberia ayudar especialmente en retrieval (recuperar la categoria tangencial) y utilidad (calibrar la frontera parcial/util).

### Opcion B: Aceptar limitaciones por eje

- **Generation (76%):** listo para usar como primer filtro. Humano revisa solo los 5 desacuerdos.
- **Retrieval y utilidad (~50%):** no son tratables solo con prompt. Opciones: colapsar categorias (binario bueno/malo y util/danino), aceptar evaluacion humana, o pasar a un modelo mas capaz (GPT-4o en vez de mini).

### Opcion C: Simplificar las escalas

Colapsar de 3 niveles a 2: retrieval bueno/malo (sin tangencial), utilidad util/danino (sin parcial). "Tangencial" y "parcial" son las categorias que causan la mayoria de desacuerdos. Eliminarlas fuerza tanto al humano como al juez a tomar decisiones binarias, lo que probablemente suba el acuerdo. Costo: se pierde granularidad.
