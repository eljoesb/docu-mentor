# v5 — Notas

## Decisión de diseño: query original en segunda generation

En `run()`, la segunda generation llama `generate(query, r2["chunk_text"])` con la **query original**, no con la reformulada. La reformulación es solo un truco de retrieval — un ancla semántica distinta para encontrar un chunk mejor. Pero el usuario preguntó una cosa y quiere respuesta a esa cosa, no a la reformulación.

Si se pasara la query reformulada al generate, el LLM podría responder "about DistributedDataParallel internal mechanics" cuando el usuario preguntó "how to use multiple GPUs". Separar los dos roles de la query — como ancla de búsqueda vs como pregunta a responder — es clave.

## Observaciones de la corrida v5.0

### Trigger activó 7/21 queries

| Activó retry | No activó retry |
|---|---|
| Q4, Q5, Q6, Q8, Q11, Q17, Q20 | Q1-Q3, Q7, Q9, Q10, Q12-Q16, Q18, Q19, Q21 |

### Problema: 3 queries del Grupo B no activaron retry

Q12 (save/load model), Q15 (custom autograd), Q18 (weights not changing) — las tres son Grupo B (utilidad=parcial en v3b) pero sus respuestas en v5 no contienen "I don't know". El trigger literal no las captura porque el LLM da respuestas parciales/mixtas en vez de rechazar. Esto significa que v5 no puede mejorarlas — el mecanismo de re-búsqueda ni siquiera se activa.

### Canarios que activaron retry: 4 de 5

Q4, Q5, Q6, Q17 activaron retry. Q16 no. Esto confirma el análisis pre-commit: los canarios OOD dicen "I don't know" y activan el trigger. La predicción de 1-2 canarios rotos depende de qué hizo el LLM con el segundo chunk.

### Reformulaciones de 3B son ruidosas

El modelo de 3B inventó APIs inexistentes en las reformulaciones (`pytorch.optimolvers.gan`, `torch.core.data.dataset.Dataset.__getitem__`). Las reformulaciones útiles son las que usan nombres reales de PyTorch (Q8: `DataParallel, DistributedDataParallel`; Q20: `torch.Tensor.view vs torch.Tensor.reshape`).

## Resultados de clasificación v5.0 (tres ejes)

### Conteos

| Eje | Positivo | Medio | Negativo |
|-----|----------|-------|----------|
| Retrieval | 7/21 bueno | 9/21 tangencial | 5/21 malo |
| Generation | 9/21 fiel | 1 empty + 9 mixto | 2/21 inventado |
| Utility | 9/21 util | 9/21 parcial | 3/21 danino |

### Comparación v3b → v5

| Eje | v3b | v5 | Delta |
|-----|-----|----|-------|
| Retrieval bueno | 7/21 | 7/21 | 0 |
| Generation fiel | 14/21 | 9/21 | **-5** |
| Generation inventado | 2/21 | 2/21 | 0 |
| Utility util | 11/21 | 9/21 | **-2** |
| Utility danino | 2/21 | 3/21 | **+1** |

### Categoría empty y la distinción con fiel honesto

Solo Q12 (save/load model): "Save the model by exporting from your preferred framework." Respuesta vacua que ni usa el chunk ni inventa claims técnicos específicos. Un "I don't know" disfrazado de respuesta.

**Distinción clave fiel-honesto vs empty-vacuo:** Q11 y Q20 también "no responden", pero son fiel, no empty. La diferencia:

- **Q11 (fiel):** "I don't know. The context discusses the Linear module but not F.linear." — Meta-información precisa sobre el chunk. Describe qué tiene y qué le falta.
- **Q20 (fiel):** "The context discusses reshape, reshape_as, flatten but doesn't provide info about distinguishing view from reshape." — Mismo patrón: referencia explícita al contenido del chunk.
- **Q12 (empty):** "Save the model by exporting from your preferred framework." — No describe el chunk, no se basa en el chunk, no rechaza usando el chunk. Solo existe al lado del chunk.

**Heurística para detector automático de empty:** un rechazo fiel suele contener strings como "doesn't", "context", "don't", "provided" — porque está hablando SOBRE el chunk. Un empty no contiene ninguna referencia al chunk. Esto podría ser un detector simple para v5.1: si la respuesta no contiene "I don't know" Y no referencia el contexto/chunk Y es genérica, marcar como empty.

### Tres modos de fallo de generation que el detector "I don't know" no captura

| Modo | Ejemplo | Daño | Detectable por |
|------|---------|------|----------------|
| Empty (confidently nothing) | Q12: consejo genérico vacuo | Frustración | Respuesta corta + ausencia de API names del chunk |
| Mixto (estructura real + detalles inventados) | Q15: autograd.Function correcto, reset_grad inventado | Bug silencioso en código copiado | Comparar claims específicos con chunk |
| Inventado puro | Q10: inventa diferencia NumPy .size()/.shape | Error técnico directo | Comparar claims con chunk (nada matchea) |

## Lección central de v5.0: la intervención empeoró el sistema

v5.0 diseñó una optimización (re-búsqueda) para el cuello de botella de v3 (retrieval malo). Pero entre v3b y v5.0 el cuello de botella se movió: el prompt mejorado de v3b ya había convertido muchos rechazos en "I don't know" honestos (generation fiel). La re-búsqueda atacó un problema parcialmente resuelto y empeoró el problema activo.

**Mecanismo concreto:** las 7 queries que activaron retry encontraron chunks distintos. En algunos casos (Q8: chunk de AMP+DDP, Q17: chunk de main components) el chunk nuevo era más ambiguo que el original. El LLM de 3B, ante material ambiguo, mezcla en vez de rechazar — y eso bajó generation fiel de 14 a 9. Los 5 puntos perdidos en fidelidad son queries donde v3b decía "I don't know" y v5 mezcla.

**Principio general:** cuando el cuello de botella se mueve, las optimizaciones diseñadas para el cuello anterior empeoran el sistema, no lo dejan igual. No es neutralidad — es regresión activa. Porque la optimización cambia el contexto en el que opera el componente que ahora es crítico. En v5.0, la re-búsqueda le dio al LLM más material ambiguo para mezclar. Neutralidad es la excepción, no la regla.

## Implicaciones para v5.1

El detector "I don't know" opera en la capa equivocada. Los tres modos de fallo de generation (empty, mixto, inventado) requieren detectores en la capa de groundedness: comparar la respuesta con el chunk y verificar si los claims están soportados. Eso es faithfulness verification, no string matching.

Prioridad sugerida: atacar empty primero (el más barato de detectar — respuesta corta + ausencia de términos del chunk), dejar mixto/inventado para v5.2+ (requieren algo parecido a LLM-as-judge).
