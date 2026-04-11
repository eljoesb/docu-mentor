# v6 — Ideas futuras

## Q15: 8b genera vago con chunks buenos en queries de implementacion

Q15 ("implement custom autograd function") recupero chunks relevantes de extending.func.rst con ejemplos reales de custom autograd functions. Pero 8b produjo una respuesta evasiva ("Yes, you can implement custom autograd functions based on these contexts") en vez de extraer los pasos o el codigo de los chunks.

Esto no es un fallo de retrieval ni de ambiguedad de query. Es un fallo de generation: el modelo tiene los chunks correctos y no extrae el contenido actionable. El patron parece especifico a queries de "implement X" donde los chunks muestran ejemplos pero no un step-by-step explicito — el modelo describe los chunks en vez de sintetizar una respuesta.

Posible v6.2: ajuste de prompt para forzar extraccion de codigo de los chunks cuando la query pide implementacion. No se ataca en v7 (que va por otro eje: clarificacion al usuario).
