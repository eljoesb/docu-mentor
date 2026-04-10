# DocuMentor

Asistente de consulta sobre documentación técnica. Proyecto guía del roadmap de transición a AI/ML Engineer.

## Versiones

- **[v0](./v0/README.md)** — Retrieval léxico y semántico. Recorre los primeros 30 años de Information Retrieval en una sola sesión: grep tonto → TF-IDF → chunking → embeddings.
- **[v1](./v1/README.md)** — Pipeline RAG completo (caché persistente + generación con Ollama) + evaluación sistemática de 21 queries. El sistema acierta 48% del tiempo por las razones equivocadas; el eval lo demuestra.
- **[v2](./v2/README.md)** — Hybrid search con BM25 + RRF para atacar el cuello de botella de retrieval de v1. Hipótesis: combinar dense (embeddings) + sparse (BM25) vía Reciprocal Rank Fusion recupera más chunks relevantes. Resultado: 2/21 buenos, peor que embeddings puros. El diagnóstico reveló que el problema no es el algoritmo de retrieval sino la estructura del corpus indexado.
- **[v2.1](./v2.1/README.md)** — Cambio de modelo de embeddings: MiniLM-L6-v2 (384 dim) → BGE-base-en-v1.5 (768 dim). Resultado: 11/21 buenos, casi el triple del baseline. El modelo de embeddings era el cuello de botella. Camino 3 confirmado.
- **[v3](./v3/README.md)** — Pipeline completo BGE + Ollama con evaluación de 3 ejes (retrieval/generation/utilidad). Descubrió dos cuellos de botella simultáneos. Una línea de prompt engineering eliminó 7/9 invenciones del LLM. El sistema dejó de mentir (dañinos: 6→2) pero la utilidad subió poco (10→11). Varianza intra-evaluador de ±9 queries reveló necesidad de LLM-as-judge.
- **[v4](./v4/README.md)** — Meta-evaluación: LLM-as-judge (GPT-4o-mini) contra clasificaciones humanas de v3b. Dos iteraciones de prompt. Hallazgo central: las reglas explícitas en prompts de juez se aplican como leyes, no como matices — una regla pensada para 2-3 edge cases se aplicó a 7+ casos. Generation alcanzó 76% de acuerdo con una sola regla; retrieval y utilidad siguen en ~50%.

## Estado actual

**Pipeline:** BGE-base-en-v1.5 (retrieval) + Ollama llama3.2:3b (generation) + prompt mejorado con instrucción de honestidad ante contexto tangencial.

**Resultados v3 (criterio estricto):**

| Eje | v1 | v3 | Delta |
|-----|----|----|-------|
| Retrieval bueno | 3/21 | 7/21 | +4 |
| Generation fiel | 8/21 | 14/21 | +6 |
| Generation inventado | 11/21 | 2/21 | -9 |
| Utilidad útil | 10/21 | 11/21 | +1 |
| Utilidad dañino | 6/21 | 2/21 | -4 |

**Acuerdo LLM-as-judge v4.1 (GPT-4o-mini vs humano):**

| Eje | Acuerdo | Status |
|-----|---------|--------|
| Retrieval | 10/21 (48%) | Necesita iteración |
| Generation | 16/21 (76%) | Primer filtro viable |
| Utility | 11/21 (52%) | Necesita iteración |
| **Global** | **37/63 (59%)** | **<70% — seguir iterando** |

**Perros en la nieve:** parcialmente cerrados. El sistema dejó de mentir (2 dañinos). Todavía no es suficientemente útil (11/21, necesita 13).

**Hipótesis validadas:**
- El cuello de botella de retrieval era capacidad semántica del modelo (MiniLM → BGE-base).
- Había un segundo cuello de botella en generation: el LLM inventaba ante chunks tangenciales. Una línea de prompt lo resolvió en 7/9 casos.
- LLM-as-judge funciona para generation (76% acuerdo con una regla). No funciona aún para retrieval ni utilidad (~50%).

**Hipótesis abiertas:**
- Few-shot examples calibrados para el juez (en vez de reglas explícitas) para subir acuerdo en retrieval y utilidad.
- Simplificar escalas a binarias (bueno/malo, útil/dañino) para reducir desacuerdos en categorías intermedias.
- Hybrid search (BGE + BM25) para mejorar retrieval (7/21 con criterio estricto).
- Modelo 8B si prompt engineering + hybrid no cierran los perros en la nieve.

## Corpus actual

Documentación oficial de PyTorch (`docs/source`, 248 archivos).

## Stack

Python + scikit-learn + sentence-transformers (BAAI/bge-base-en-v1.5) + Ollama (llama3.2:3b local) + OpenAI API (GPT-4o-mini para LLM-as-judge).
