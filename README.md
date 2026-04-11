# DocuMentor

Asistente de consulta sobre documentación técnica. Proyecto guía del roadmap de transición a AI/ML Engineer.

## Versiones

- **[v0](./v0/README.md)** — Retrieval léxico y semántico. Recorre los primeros 30 años de Information Retrieval en una sola sesión: grep tonto → TF-IDF → chunking → embeddings.
- **[v1](./v1/README.md)** — Pipeline RAG completo (caché persistente + generación con Ollama) + evaluación sistemática de 21 queries. El sistema acierta 48% del tiempo por las razones equivocadas; el eval lo demuestra.
- **[v2](./v2/README.md)** — Hybrid search con BM25 + RRF para atacar el cuello de botella de retrieval de v1. Hipótesis: combinar dense (embeddings) + sparse (BM25) vía Reciprocal Rank Fusion recupera más chunks relevantes. Resultado: 2/21 buenos, peor que embeddings puros. El diagnóstico reveló que el problema no es el algoritmo de retrieval sino la estructura del corpus indexado.
- **[v2.1](./v2.1/README.md)** — Cambio de modelo de embeddings: MiniLM-L6-v2 (384 dim) → BGE-base-en-v1.5 (768 dim). Resultado: 11/21 buenos, casi el triple del baseline. El modelo de embeddings era el cuello de botella. Camino 3 confirmado.
- **[v3](./v3/README.md)** — Pipeline completo BGE + Ollama con evaluación de 3 ejes (retrieval/generation/utilidad). Descubrió dos cuellos de botella simultáneos. Una línea de prompt engineering eliminó 7/9 invenciones del LLM. El sistema dejó de mentir (dañinos: 6→2) pero la utilidad subió poco (10→11). Varianza intra-evaluador de ±9 queries reveló necesidad de LLM-as-judge.
- **[v4](./v4/README.md)** — Meta-evaluación: LLM-as-judge (GPT-4o-mini) contra clasificaciones humanas de v3b. Dos iteraciones de prompt. Hallazgo central: las reglas explícitas en prompts de juez se aplican como leyes, no como matices — una regla pensada para 2-3 edge cases se aplicó a 7+ casos. Generation alcanzó 76% de acuerdo con una sola regla; retrieval y utilidad siguen en ~50%.
- **[v4.2](./v4.2/NOTES.md)** — Hybrid search revisitada: BGE + BM25 con weighted RRF (alpha=0.7/0.3). Resultado: 7/21 bueno, igual que BGE puro. Ganó Q8 y Q13 (rescates reales), perdió Q2 y Q14 (breakage por compromise). Net = 0. BGE puro queda como definitivo. Primera predicción exacta del proyecto (7/21 predicho, 7/21 real). Validación cruzada del juez sobre datos nuevos: 17/21 (81%) de acuerdo, 90% con escala binaria. Cierra Fase 2 del roadmap (retrieval).

## Estado actual — Fin de Fase 2

**Pipeline definitivo:** BGE-base-en-v1.5 (retrieval, sin hybrid) + Ollama llama3.2:3b (generation) + prompt mejorado con instrucción de honestidad ante contexto tangencial.

**Resultados de retrieval (criterio estricto):**

| Version | Retrieval bueno | Que se probo |
|---------|----------------|-------------|
| v1 | 3/21 | MiniLM baseline |
| v2 | 2/21 | Hybrid MiniLM + BM25 (fallo) |
| v2.1 | 7/21 (estricto) | BGE-base (cuello de botella era el modelo) |
| v3 | 7/21 | BGE-base en pipeline completo |
| v4.2 | 7/21 | Hybrid BGE + BM25 (wash, no se sube) |

**Resultados de pipeline (v3, criterio estricto):**

| Eje | v1 | v3 | Delta |
|-----|----|----|-------|
| Retrieval bueno | 3/21 | 7/21 | +4 |
| Generation fiel | 8/21 | 14/21 | +6 |
| Generation inventado | 11/21 | 2/21 | -9 |
| Utilidad útil | 10/21 | 11/21 | +1 |
| Utilidad dañino | 6/21 | 2/21 | -4 |

**LLM-as-judge (GPT-4o-mini vs humano):**

| Eje | v4.1 (v3b data) | v4.2 (hybrid data) |
|-----|-----------------|-------------------|
| Retrieval | 10/21 (48%) | 17/21 (81%) |
| Retrieval (binario) | — | 19/21 (90%) |

**Perros en la nieve:** parcialmente cerrados. El sistema dejó de mentir (2 dañinos). Todavía no es suficientemente útil (11/21, necesita 13).

**Hipótesis validadas:**
- El cuello de botella de retrieval era capacidad semántica del modelo (MiniLM → BGE-base).
- Había un segundo cuello de botella en generation: el LLM inventaba ante chunks tangenciales. Una línea de prompt lo resolvió en 7/9 casos.
- Hybrid search no mejora retrieval sobre BGE puro cuando el componente sparse es débil (v2 y v4.2).
- LLM-as-judge funciona para generation (76%) y para retrieval con escala binaria (90%).
- El sesgo optimista del predictor se corrige adoptando null hypothesis como default.

**Fase 2 cerrada.** Próxima fase: Agents (v5).

## Corpus actual

Documentación oficial de PyTorch (`docs/source`, 248 archivos).

## Stack

Python + scikit-learn + sentence-transformers (BAAI/bge-base-en-v1.5) + Ollama (llama3.2:3b local) + OpenAI API (GPT-4o-mini para LLM-as-judge).
