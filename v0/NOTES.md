Bug 1: las palabras comunes dominan. "to", "a", "the", "is" aparecen en todos lados y inflan el conteo de chunks irrelevantes. → Esto se llama el problema de las stopwords y la falta de ponderación por frecuencia inversa. Las palabras raras deberían valer más que las comunes.

Bug 2: los chunks largos ganan injustamente. El chunk de RPC que ganó para DataLoader tiene como 40 líneas. Un chunk corto y perfectamente relevante de 3 líneas no tiene chance de competir en conteo bruto contra un monstruo así. → Esto se llama falta de normalización por longitud.
Hay un algoritmo clásico que resuelve los dos al mismo tiempo, y se llama TF-IDF. No te voy a explicar la teoría todavía. Te voy a hacer implementarlo y vas a ver la diferencia, y después hablamos de por qué funciona.

En cualquier sistema de retrieval, cómo partes los documentos importa tanto o más que cómo los buscas. Un buen retriever sobre chunks malos pierde contra un retriever mediocre sobre chunks bien formados.