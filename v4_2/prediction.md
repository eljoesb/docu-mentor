# Prediccion: v4.2 — Hybrid search BGE + BM25

**Fecha:** 2026-04-10

## Que es v4.2

Retrieval hibrido. Se combina BGE-base (denso) con BM25-smart (sparse) via Reciprocal Rank Fusion ponderado. El objetivo es rescatar queries donde BGE pierde por falta de especificidad nominal (Q13: "numpy", Q19: "nan") sin romper las 7 que BGE ya acierta.

## Baseline

**BGE puro (v3b, criterio estricto): 7/21 bueno.**

Los 7 buenos: Q1 (autograd), Q2 (GPU), Q3 (DataLoader), Q9 (leaf tensor), Q14 (flatten), Q15 (custom autograd), Q21 (learning rate).

BM25-smart solo: ~2-3/21 bueno (estimado de v2).

## Configuracion de RRF

**Opcion elegida: (b) Weighted RRF, k=60, alpha_dense=0.7, alpha_sparse=0.3.**

Formula: `score(d) = 0.7 * 1/(60 + rank_dense(d)) + 0.3 * 1/(60 + rank_sparse(d))`

Por que no (a) — k=60 sin ponderacion:
Ya fallo en v2. Con k=60 sin pesos, RRF premia "decente en ambos" sobre "perfecto en uno". El chunk de sparse.rst (BGE rank 0 para Q13) empata con chunks mediocres que BM25 sube. La disparidad BGE >> BM25 requiere darle mas voto al componente fuerte.

Por que no (c) — cascada (BGE primero, BM25 como rescate):
BGE es confidently wrong en Q13 (score 0.7508, el mas alto de todos). Un threshold de confianza no se activaria para el caso canonico que queremos rescatar. La cascada solo rescata queries donde BGE duda; Q13 no es una de ellas.

Por que (b):
Alpha=0.7 protege los buenos de BGE: para que BM25 desplace al ganador de BGE, el chunk alternativo necesita estar en el top ~10 de BGE Y en el top ~3 de BM25 simultaneamente. Eso filtra la mayoria del ruido de BM25. Al mismo tiempo, cuando BGE esta cerca del chunk correcto (rank 3-5) y BM25 lo rankea alto (keywords raros como "numpy" o "nan"), la combinacion puede flipear el resultado.

## Prediccion 1: Retrieval bueno con hybrid

**7/21.** Rango: 6-9.

El punto central es identico al baseline. Suena derrotista pero es la prediccion honesta dado:

1. BM25 sigue siendo un componente debil (2-3/21). Weighted RRF con un componente debil agrega mas ruido que senal en la mayoria de las queries.
2. Las ganancias potenciales (Q13, Q19, Q12) son individualmente de baja probabilidad (~20-50% cada una). El valor esperado de ganancias es ~1.
3. Las perdidas potenciales (breakage de buenos existentes por interferencia de BM25) son bajas pero no nulas. Valor esperado ~0.3.
4. Neto esperado: +0.7, que se redondea a 0 despues de correccion por sesgo optimista.

### Mecanica de las ganancias potenciales

**Q19 (loss nan) — el mejor candidato de rescate (~50%).**
BGE rank 1: faq.rst (training loop problem, tangencial). BGE rank 4: autograd.rst (div-by-zero produce NaN, bueno). Margen: 0.018. BM25 daria senal fuerte a autograd.rst: "nan" es token raro con IDF altisimo, y el chunk muestra `Results in [inf, 1]` y `x.grad` con NaN literalmente. Con alpha=0.7, faq.rst (BGE rank 0, BM25 rank ~15) queda en ~0.0157 y autograd.rst (BGE rank 3, BM25 rank ~1) queda en ~0.0161. El flip es mecanicamente plausible.

**Q13 (numpy to tensor) — el caso canonico (~35%).**
BGE rank 1: sparse.rst (malo, patron semantico "convert X to tensor" sin X=numpy). BM25 rank alto para tensors.rst: "numpy" tiene IDF alto y el chunk tiene `torch.tensor(np.array(...))` literalmente. Pero el obstaculo es el rank de tensors.rst en BGE: si esta en rank 10+ (probable dado que BGE no conecta "numpy" con tensors.rst), la ponderacion alpha=0.7 no deja que BM25 lo rescate. En MiniLM era rank 14; BGE podria ser mejor (rank 8-12?) pero no lo suficiente.

**Q12 (save/load model) — rescate marginal (~20%).**
BGE va a AOT inductor (tangencial). Serialization.rst abre con "how you can save and load". BM25-dumb ya lo encontraba en v2. Si serialization.rst esta en BGE top 10 Y BM25 top 3, el flip es posible. Pero es una cadena de dos condiciones inciertas.

### Mecanica de las perdidas potenciales

Los 7 buenos son queries donde los keywords del query aparecen naturalmente en el chunk correcto: "autograd" en autograd.rst, "GPU"/"cuda" en cuda.rst, "DataLoader"/"workers" en data.md, "leaf tensor" en autograd.rst, "flatten"/"linear" en named_tensor.md, "autograd function" en extending.func.rst, "learning rate" en optim.md. BM25 refuerza estas coincidencias, no las compite.

El unico riesgo es Q14 (flatten before linear): named_tensor.md no es el target natural por keywords, y BM25 podria promover un chunk de nn.rst que lista "Flatten" como API pero no explica el patron. Probabilidad de breakage: ~15%.

**Buenos que se rompen: 0.** Rango: 0-1.

## Prediccion 2: Q13 especificamente

**Sigue malo.** Probabilidad de mejorar a tangencial o bueno: ~35%.

El argumento mecanico a favor del rescate es solido (BM25 matchea "numpy" literalmente en tensors.rst). El argumento en contra es mas fuerte: BGE probablemente tiene tensors.rst en rank 10-15 para Q13, y con alpha=0.7 eso es demasiado lejos para que BM25 rank 0 lo rescate. La aritmetica de RRF ponderado muestra que el chunk alternativo necesita estar en BGE top ~8 para que BM25 lo flipee, y no tengo evidencia de que tensors.rst llegue a ese rango.

Si Q13 se corrige, seria la senal mas fuerte de que hybrid funciona. Si no se corrige, confirma que el problema de especificidad nominal requiere un modelo denso mas capaz (BGE-large? GTE? Instructor?) en vez de un parche sparse.

## Prediccion 3: Buenos que se rompen

**0/7.** Rango: 0-1.

Los 7 buenos actuales son casos de alineacion densa-sparse: el chunk correcto tiene tanto match semantico (BGE lo encuentra) como match lexico (BM25 lo refuerza). Weighted RRF solo rompe un bueno cuando BM25 promueve agresivamente un chunk con keywords pero sin semantica correcta, Y ese chunk esta cerca en el ranking de BGE. Para los 7 actuales, eso no pasa.

## Regla de decision

**>=9/21: hybrid funciona, lo subimos al producto.** Significa que BM25 rescato al menos 2 queries sin romper las existentes. El costo de mantener el indice BM25 se justifica.

**7-8/21: hybrid es un wash. BGE puro queda como definitivo para retrieval.** Significa que BM25 no agrego senal neta. Simplificar el sistema: un componente menos que mantener, una fuente de ruido menos. La mejora de retrieval pasa por mejor modelo denso (BGE-large, fine-tuning) o mejor chunking, no por ensembling con sparse.

**<=6/21: hybrid empeora. BGE puro definitivo, y ademas confirma que el regimen de v2 ("naive ensembling hurts") aplica incluso con ponderacion.** Weighted RRF no es suficiente para proteger un componente fuerte de un componente debil. La disparidad BM25 (2-3/21) vs BGE (7/21) es todavia demasiado grande para que fusion funcione.

## Calibracion del predictor

Cinco predicciones consecutivas optimistas:

| Prediccion | Esperado | Real | Error |
|------------|----------|------|-------|
| v2.1 retrieval | 9/21 | 11/21 | +2 (pero buckets internos cancelados) |
| v3 retrieval | 11/21 | 7/21 | **-4** |
| v3 utilidad | 14/21 | 11/21 | **-3** |
| v4.0 acuerdo | 75% | 56% | **-19pp** |
| v4.1 acuerdo | 70% | 59% | **-11pp** |

El sesgo tiene causa especifica: sobreestimo el impacto positivo de los cambios. Cada vez asumo que la modificacion mejora las cosas; el resultado real es que mejora menos o no mejora. La correccion no es restar puntos mecanicamente sino cambiar el prior: **asumo que el cambio no mejora nada (null hypothesis) y solo predigo mejora cuando el argumento mecanico es irrefutable.** Para v4.2, ningun argumento de rescate es irrefutable — todos dependen de rankings desconocidos en BGE. Por eso mi punto central es 7/21 (sin cambio), no 8 o 9.
