# v2.1 — Notas de análisis

## Diagnóstico: Q13 — "convert numpy array to pytorch tensor"

**Resultado:** BGE fue a sparse.rst (score 0.7508). Clasificado como malo. MiniLM (v1) iba a tensor_view.rst (también malo). Hybrid (v2) iba a tensors.rst (bueno — el ejemplo literal `torch.tensor(np.array(...))`).

**El chunk ganador de BGE:**

```
>>> a = torch.tensor([[0, 2.], [3, 0]])
    >>> a.to_sparse()
    tensor(indices=tensor([[0, 1], [1, 0]]),
           values=tensor([2., 3.]),
           size=(2, 2), nnz=2, layout=torch.sparse_coo)

PyTorch currently supports COO, CSR, CSC, BSR, BSC...
```

**H1 (mecánica — matching por keywords) refutada.** `grep` confirmó cero menciones de "numpy", "np.array", o "from_numpy" en sparse.rst. No hay match léxico posible.

**H2 (representacional — matching por patrón semántico) confirmada.** BGE matcheó el patrón "convert X to Y tensor": el chunk muestra `torch.tensor(...)` (creación de tensor) + `a.to_sparse()` (conversión de formato de tensor). Para BGE, "convert numpy array to pytorch tensor" y "convert dense tensor to sparse tensor" son vecinos en el espacio de embeddings porque ambos son operaciones de conversión sobre tensores. El modelo abstrae correctamente la intención verbal ("convertir algo en un tensor") pero pierde la especificidad del sustantivo crítico ("numpy").

**Contraste con hybrid de v2:** hybrid acertó Q13 porque BM25 (rank 2 por match literal de "numpy") compensó a MiniLM (rank 14). BGE puro pierde esa red de seguridad.

**Conclusión transferible:** los embeddings capturan intención pero pierden especificidad en los argumentos nominales. Cuando la diferencia entre la respuesta correcta y la incorrecta es un sustantivo específico ("numpy") y no el verbo/patrón ("convert to tensor"), el retrieval puramente denso no alcanza.

---

## Diagnóstico: Q19 — "why is my loss nan"

**Resultado:** BGE fue a faq.rst (score 0.6341). Clasificado como tangencial. MiniLM (v1) iba a autograd.rst (bueno — el ejemplo de división por cero que produce NaN en gradientes). Es la única regresión de un bueno de v1 que BGE perdió.

**El chunk ganador de BGE (faq.rst):**

```python
total_loss = 0
for i in range(10000):
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output)
    loss.backward()
    optimizer.step()
    total_loss += loss
```

"Here, `total_loss` is accumulating history across your training loop, since `loss` is a differentiable variable with autograd history."

Es un FAQ real sobre un training loop roto — pero el problema es acumulación de historial, no NaN.

**El chunk correcto (autograd.rst, rank 4 en BGE, score 0.6157):**

```python
x = torch.tensor([1., 1.], requires_grad=True)
div = torch.tensor([0., 1.])
y = x / div          # Results in [inf, 1]
mask = div != 0      # [False, True]
loss = y[mask].sum()
loss.backward()
print(x.grad)        # [nan, 1], not [0, 1]
```

Muestra literalmente por qué un loss backward produce NaN. Este es el chunk que responde la pregunta.

**Margen:** 0.018 — rank 1 vs rank 4. Mucho más apretado que Q13.

**Dos hipótesis:**

**H1 (densidad léxica):** faq.rst tiene "loss" 5 veces en contexto de training loop (`total_loss` ×3, `loss = criterion(output)`, `loss.backward()`). autograd.rst tiene "loss" pero embebido en vocabulario más técnico ("gradient expressions", "backpropagation", "differentiable"). El average pooling de BGE da más peso al chunk donde "loss" + "training" co-ocurren con mayor densidad.

**H2 (representacional):** BGE agrupa "training loop problems" en un cluster semántico. El faq.rst sigue el patrón "problema → código → explicación → fix" que es semánticamente cercano a preguntas del tipo "why is my X". El autograd.rst está en el cluster "autograd internals / gradient math" que es más lejano al intent del usuario.

**Más probable: H1.** La evidencia es el margen: 0.018 es muy chico. El chunk correcto está en rank 4, no en rank 50. BGE *casi* acierta. La diferencia está en densidad de "loss" + "training" como tokens, no en una confusión representacional profunda. Además, "nan" como token es raro y técnico — probablemente pesa menos en el embedding promediado que "loss" que aparece 5 veces.

**Diferencia con Q13:** Q13 es un fallo representacional (BGE fue a un tema completamente distinto, sparse tensors, con margen alto). Q19 es un fallo de densidad léxica (BGE fue a un tema genuinamente relacionado, training loop problems, con margen mínimo). Q13 necesita hybrid search para corregirse (BM25 aporta "numpy" como señal discriminante). Q19 podría corregirse con hybrid también (BM25 daría señal fuerte a "nan" como keyword), pero también podría corregirse simplemente con un modelo más grande que le dé más peso al token "nan" en el embedding.

---

## Conclusiones principales

### 1. La regla de decisión se cumplió — camino 3 confirmado

11/21 buenos ≥ 10/21 umbral. Según la regla pre-comprometida en prediction.md: "camino 3 confirmado. El modelo de embeddings era el cuello de botella." El resultado es inequívoco — no necesita asteriscos, no necesita interpretación. BGE-base, el experimento más barato del camino 3, fue suficiente. No hace falta BGE-large.

### 2. La calibración agregada fue un espejismo de cancelación

Predicción: 9/21. Resultado: 11/21. Δ = +2. Parece calibración decente.

Pero la composición interna:

| Bucket | Predicción | Resultado | Error |
|--------|-----------|-----------|-------|
| Baseline (retener 4) | 4/4 | 3/4 | −1 |
| Alta confianza (3 flipean) | 3/3 | 1/3 | −2 |
| Confianza media (2/3 flipean) | 2/3 | 2/3 | 0 |
| Apuestas largas (0/3 flipean) | 0/3 | 2/3 | +2 |
| Descartadas gap corpus (0/3) | 0/3 | 3/3 | +3 |

Los buckets "alta confianza" y "baseline" sobreestimaron por 3. Los buckets "apuestas largas" y "descartadas" subestimaron por 5. El neto (+2) es compensación de errores en direcciones opuestas. El modelo mental query-por-query estaba más roto de lo que el agregado sugiere.

Lección metodológica: cuando predigas resultados de un experimento en el futuro, medí tu calibración por bucket, no por el número total. Si el agregado da +2 pero los buckets dan +5/−3, tu modelo mental necesita revisión, no celebración.

### 3. La frontera corpus/modelo no se puede determinar con un solo modelo

Q7, Q10, Q18 fueron descartadas como "gap de corpus, no de modelo" — asumiendo que ningún chunk del corpus podía responderlas. BGE encontró chunks buenos para las tres.

La conclusión es que "el corpus no tiene la respuesta" solo es afirmable después de probar con un modelo suficientemente capaz. Con MiniLM, Q7 (memorizing → overfitting) parecía imposible porque MiniLM no conectaba el vocabulario coloquial con el contenido técnico. BGE sí. No era gap de corpus — era gap de modelo disfrazado.

Implicación operativa: cualquier análisis de cobertura del corpus que use un solo modelo de embeddings está condicionado a la capacidad de ese modelo. Lo que parece "el documento no existe" puede ser "mi modelo no lo encuentra".

### 4. Los embeddings capturan intención pero pierden especificidad nominal — y eso reabre hybrid search

El deep dive de Q13 reveló una tesis concreta y falsable: cuando la diferencia entre la respuesta correcta y la incorrecta es un sustantivo específico (no el verbo/patrón), el retrieval denso pierde. BGE matcheó "convert X to Y tensor" pero no distinguió X=numpy de X=sparse. Es una limitación estructural de embeddings, no de BGE específicamente.

Esto explica retroactivamente por qué hybrid search con BM25 tenía sentido como idea. BM25 preserva la especificidad nominal (matchea "numpy" literalmente) mientras embeddings preserva la intención (matchea "convert to tensor" semánticamente). En queries donde el sustantivo clave es raro y específico ("numpy", "nan", "torch.nn.functional.linear"), BM25 tiene ventaja estructural sobre cualquier modelo denso por más grande que sea.

**Corolario — hybrid search reabre como hipótesis para v3:**

v2 hybrid falló porque MiniLM + BM25-smart eran componentes de calidad dispar (MiniLM 4/21 buenos, BM25-smart ~2/21). RRF con k=60 penalizaba la señal densa buena al promediarla con la señal sparse mala. Es el régimen donde "naive ensembling hurts when component quality differs".

Con BGE-base como componente denso (11/21 buenos), ambos componentes serían de calidad comparable. Ese es exactamente el régimen donde hybrid search funciona según la literatura. Weighted RRF o no, el régimen es distinto y merece ser medido.

Esta hipótesis no existía antes del deep dive de Q13. Antes, "v3 hybrid sobre BGE" era una idea genérica. Después del deep dive, es una hipótesis con razonamiento mecánico: BGE pierde Q13 por falta de especificidad nominal → BM25 aporta exactamente esa especificidad → la combinación debería rescatar Q13 sin perder los 11 que BGE ya acierta, siempre que los pesos de fusión no destruyan las señales buenas como pasó en v2.
