# METHODOLOGICAL NOTE:
# Las clasificaciones en CLASSIFICATIONS fueron hechas mientras se editaba
# este archivo directamente, lo cual significa que no hubo separación física
# entre el acto de clasificar y el de contar. Esto introduce un riesgo de
# sesgo de confirmación hacia la predicción escrita al inicio. Los tres
# deltas (+1, +1, +1) son consistentes con ese sesgo, aunque también son
# consistentes con una subestimación uniforme honesta. No hay forma de
# distinguir las dos hipótesis con los datos actuales.
#
# Para v2.1 y posteriores: clasificar en archivo separado (CSV sin fórmulas)
# antes de cualquier agregación.

"""
rebaseline.py — Tabla comparativa de los tres sistemas de retrieval bajo criterio estricto.

En base a los tres archivos JSONL de resultados (v1 embeddings, BM25-smart, hybrid RRF)
y unifica en una tabla de 21 queries × 3 sistemas. Cada celda clasificada como
bueno/tangencial/malo bajo criterio estricto:

    "bueno = el chunk responde la pregunta, no solo la menciona"

Las clasificaciones son manuales, hechas leyendo cada chunk con atención.
No es un re-run del pipeline — es una auditoría retroactiva.
"""

import json
import os
from pathlib import Path

BASE = Path(__file__).parent.parent
V1_RESULTS = BASE / "v1" / "eval_results.jsonl"
BM25_RESULTS = BASE / "v2" / "bm25_results_smart.jsonl"
HYBRID_RESULTS = BASE / "v2" / "hybrid_results.jsonl"


def load_jsonl(path):
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            results[rec["id"]] = rec
    return results


def short_source(path):
    """Extrae solo el nombre del archivo desde la ruta completa."""
    if not path:
        return "?"
    p = Path(path)
    # Tomar últimas 2 partes si hay subdirectorio relevante
    parts = p.parts
    docs_idx = None
    for i, part in enumerate(parts):
        if part == "docs-source":
            docs_idx = i
            break
    if docs_idx is not None and docs_idx + 1 < len(parts):
        return "/".join(parts[docs_idx + 1:])
    return p.name


# ─── PREDICCIONES (comprometidas antes de clasificar) ─────────────────────────
#
# Basadas en lo que sé del NOTES.md de v2 y el README, pero SIN haber releído
# cada chunk con criterio estricto todavía:
#
#   v1 embeddings:  3 buenos
#   BM25-smart:     1 bueno
#   Hybrid RRF:     2 buenos
#
# Razonamiento:
#   - v1 embeddings: NOTES dice 2-3/21. Los 3 originales eran Q1 (autograd),
#     Q9 (leaf tensor), Q19 (loss nan). Bajo criterio estricto, alguno podría
#     caer. Apuesto a 3 porque Q1 y Q9 son definiciones directas.
#   - BM25-smart: NOTES dice ~1-2/21. BM25 es peor en queries conceptuales,
#     y smart tokenizer perdió Q12 (save/load). Solo Q9 debería sobrevivir.
#   - Hybrid: NOTES dice 2/21 (Q9, Q13). RRF demostró perder chunks buenos
#     de embeddings (Q1, Q19) por el efecto "decent in both". Apuesto a 2.

PREDICTIONS = {
    "v1_embeddings": 3,
    "bm25_smart": 1,
    "hybrid": 2,
}

# ─── CLASIFICACIONES (63 celdas, criterio estricto) ──────────────────────────
#
# Cada entrada: (label, razón_corta)
# label ∈ {"bueno", "tangencial", "malo"}
#
# Criterio: "bueno = el chunk RESPONDE la pregunta, no solo la menciona"

CLASSIFICATIONS = {
    # ── Q1: "what is autograd" ──
    (1, "emb"): ("bueno",
        "Chunk define autograd como 'reverse automatic differentiation system', "
        "explica el DAG de operaciones, leaves/roots, chain rule. Respuesta directa."),
    (1, "bm25"): ("malo",
        "distributed_autograd.rst sobre contextos de autograd distribuido. "
        "No define qué es autograd — asume que ya lo sabés."),
    (1, "hybrid"): ("tangencial",
        "Código de autograd graph con a+b, d.sum().backward(). Explica dependency "
        "computation en backward pass, pero no define qué es autograd."),

    # ── Q2: "how do I move a tensor to GPU" ──
    (2, "emb"): ("tangencial",
        "tensor_attributes.rst muestra device='cuda:1' para CREAR tensores en GPU. "
        "No muestra .to() para MOVER un tensor existente."),
    (2, "bm25"): ("malo",
        "rpc.md sobre framework RPC. Menciona 'move input tensors to CPU' en "
        "contexto de RPC — completamente distinto al uso normal."),
    (2, "hybrid"): ("tangencial",
        "notes/mps.rst muestra model.to(mps_device) y device='mps'. El patrón "
        ".to() es correcto pero para MPS, no CUDA. Transferible pero no directo."),

    # ── Q3: "how to use DataLoader with multiple workers" ──
    (3, "emb"): ("tangencial",
        "data.md sobre compatibilidad Windows con multi-process loading. Menciona "
        "__name__=='__main__' y worker_init_fn pero no muestra num_workers=N."),
    (3, "bm25"): ("tangencial",
        "faq.rst sobre workers devolviendo random numbers idénticos. Problema "
        "específico de workers, no guía de setup."),
    (3, "hybrid"): ("tangencial",
        "data.md explica mecanismo interno: 'num_workers worker processes are "
        "created... dataset, collate_fn passed to each worker'. Explica el qué "
        "pero no muestra el cómo (la llamada a DataLoader con num_workers)."),

    # ── Q4: "what is the capital of France" (OOD) ──
    (4, "emb"): ("malo", "Cauchy-Riemann equations en autograd.rst. OOD."),
    (4, "bm25"): ("malo", "FX Graph primer. OOD."),
    (4, "hybrid"): ("malo", "FX placeholder/output nodes. OOD."),

    # ── Q5: "how do I make chocolate chip cookies" (OOD) ──
    (5, "emb"): ("malo", "Sparse BSR tensor code. OOD."),
    (5, "bm25"): ("malo", "Governance: 'How do I contribute code'. OOD."),
    (5, "hybrid"): ("malo", "torch.package: 'How do I see inside a package'. OOD."),

    # ── Q6: "what is the best optimizer for training GANs" (opinion) ──
    (6, "emb"): ("malo",
        "modules.rst sobre quantization y pruning. No menciona optimizadores "
        "ni GANs."),
    (6, "bm25"): ("malo",
        "autograd.rst sobre graph recreation at every iteration. Sin relación "
        "con optimizadores ni GANs."),
    (6, "hybrid"): ("tangencial",
        "modules.rst muestra SGD en un training loop. Al menos es sobre un "
        "optimizador en uso, pero no compara optimizadores ni menciona GANs."),

    # ── Q7: "how do I stop my model from memorizing the training data" ──
    (7, "emb"): ("malo",
        "faq.rst: 'Don't accumulate history across training loop'. Sobre memory "
        "de computation graph, NO sobre overfitting/regularización."),
    (7, "bm25"): ("malo",
        "dynamic_shapes troubleshooting sobre torch._check branching. "
        "Cero relación con memorización."),
    (7, "hybrid"): ("malo",
        "get_start_xpu.rst: training examples en Intel GPU con torch.compile. "
        "Cero relación con overfitting."),

    # ── Q8: "How to use multiple GPUs in pytorch" ──
    (8, "emb"): ("tangencial",
        "notes/cuda.rst: CUDA_VISIBLE_DEVICES y torch.cuda.device context manager. "
        "Muestra selección de device, no entrenamiento distribuido (DP/DDP)."),
    (8, "bm25"): ("tangencial",
        "Mismo chunk que embeddings. Device management, no multi-GPU training."),
    (8, "hybrid"): ("tangencial",
        "Mismo chunk. Consensus de ambos sistemas en el mismo chunk de device "
        "management."),

    # ── Q9: "what is a leaf tensor" ──
    (9, "emb"): ("bueno",
        "autograd.rst: 'leaf tensors (tensors that do not have a grad_fn, e.g., "
        "nn.Module parameters)'. Definición directa con contraste non-leaf."),
    (9, "bm25"): ("bueno",
        "Mismo chunk exacto. Definición directa."),
    (9, "hybrid"): ("bueno",
        "Mismo chunk exacto. Consensus."),

    # ── Q10: "difference between .size() and .shape" ──
    (10, "emb"): ("malo",
        "compiler_dynamic_shapes.md sobre dynamic shapes en torch.compile. "
        "Menciona 'size' en otro contexto completamente."),
    (10, "bm25"): ("malo",
        "compiler_faq.md: 'What's the difference between torch._dynamo.disable "
        "and torch._dynamo.disallow_in_graph'. Matcheó 'difference between', "
        "tema equivocado."),
    (10, "hybrid"): ("malo",
        "export/programming_model.md sobre propagación de shapes dinámicos. "
        "'backed vs unbacked dynamic shapes' — no es .size() vs .shape."),

    # ── Q11: "what does torch.nn.functional.linear do" ──
    (11, "emb"): ("tangencial",
        "extending.rst sobre implementar un módulo Linear custom. Menciona "
        "'Linear module' pero no describe qué hace F.linear (y = xA^T + b)."),
    (11, "bm25"): ("tangencial",
        "fx.md con self.linear = nn.Linear(512, 512) en forward. Ejemplo de "
        "nn.Linear (módulo), no nn.functional.linear (función)."),
    (11, "hybrid"): ("tangencial",
        "Mismo chunk que BM25. sparse_wins. Sobre nn.Linear módulo."),

    # ── Q12: "how to save and load a model for inference" ──
    (12, "emb"): ("tangencial",
        "hub.md: download_url_to_file, load_state_dict_from_url. Sobre torch.hub "
        "loading, no el flujo estándar torch.save/torch.load."),
    (12, "bm25"): ("malo",
        "autograd.rst sobre inference mode y eval mode. Matcheó 'inference' pero "
        "no habla de save/load en absoluto."),
    (12, "hybrid"): ("tangencial",
        "Mismo chunk de hub.md que embeddings. dense_wins."),

    # ── Q13: "convert numpy array to pytorch tensor" ──
    (13, "emb"): ("malo",
        "tensor_view.rst: 'PyTorch follows Numpy behaviors that basic indexing "
        "returns views'. Sobre indexing, no sobre conversión de arrays."),
    (13, "bm25"): ("malo",
        "extending.rst sobre __array_function__ protocol. Meta-nivel de "
        "interop PyTorch-NumPy, no la conversión en sí."),
    (13, "hybrid"): ("bueno",
        "tensors.rst muestra literalmente torch.tensor(np.array([[1,2,3],[4,5,6]])) "
        "y recomienda torch.as_tensor para evitar copias. Respuesta directa con "
        "código y consejo de performance."),

    # ── Q14: "how to flatten a tensor before linear layer" ──
    (14, "emb"): ("malo",
        "sparse.rst: operaciones sparse semi-structured (addmm, mm, linear). "
        "Menciona 'linear' en sparse context, no flatten."),
    (14, "bm25"): ("tangencial",
        "nn.rst lista nn.modules.flatten.Flatten y Unflatten en autosummary. "
        "Muestra que existe la API pero sin ejemplo ni explicación de uso."),
    (14, "hybrid"): ("tangencial",
        "named_tensor.md: 'Use flatten and unflatten to flatten and unflatten "
        "dimensions'. Menciona flatten pero en contexto de named tensors, "
        "no el patrón CNN flatten→linear."),

    # ── Q15: "implement custom autograd function" ──
    (15, "emb"): ("malo",
        "autograd.rst: intro general 'Autograd is a reverse automatic "
        "differentiation system'. No muestra cómo implementar una Function."),
    (15, "bm25"): ("tangencial",
        "extending.func.rst: 'you'd like to use torch.autograd.Function with "
        "torch.func transforms'. Sobre el tema correcto (autograd.Function) "
        "pero caso de uso avanzado (torch.func), no el tutorial básico."),
    (15, "hybrid"): ("tangencial",
        "Mismo chunk que BM25 (sparse_wins). Lista dos use cases pero no "
        "muestra implementación concreta."),

    # ── Q16: "how do I batch insert records into a database" (OOD) ──
    (16, "emb"): ("malo", "sparse.rst sobre batching de sparse tensors. OOD."),
    (16, "bm25"): ("malo", "data.md: automatic batching en DataLoader. OOD."),
    (16, "hybrid"): ("malo", "Mismo chunk data.md. OOD (sparse_wins)."),

    # ── Q17: "pytorch vs tensorflow for deep learning" (opinion) ──
    (17, "emb"): ("tangencial",
        "pytorch_main_components.md describe PyTorch como 'flexible and powerful "
        "library for deep learning'. Describe un lado de la comparación."),
    (17, "bm25"): ("tangencial",
        "Mismo chunk. Describe PyTorch pero no compara con TensorFlow."),
    (17, "hybrid"): ("tangencial",
        "Mismo chunk. Consensus de los tres sistemas."),

    # ── Q18: "my model weights are not changing during training" ──
    (18, "emb"): ("tangencial",
        "optim.md: ejemplo de OneLayerModel con SGD y torch.save(optimizer."
        "state_dict()). Sobre guardar estado de optimizer — menciona 'weights "
        "remain unchanged' pero en contexto de cambio de arquitectura, no debugging."),
    (18, "bm25"): ("malo",
        "optim.md sobre EMA/Polyak averaging y AveragedModel. Técnica de "
        "optimización, no diagnóstico de pesos estáticos."),
    (18, "hybrid"): ("tangencial",
        "optim.md sobre MoE expert duplication: 'loading model weights and "
        "optimizer states'. Menciona 'model weights' en contexto de clonación, "
        "no debugging de training."),

    # ── Q19: "why is my loss nan" ──
    (19, "emb"): ("bueno",
        "autograd.rst: ejemplo concreto de x/div con div=[0.,1.] → [inf,1], "
        "mask, backward → x.grad=[nan,1]. Muestra mecanismo exacto de NaN "
        "por división por cero en backward pass."),
    (19, "bm25"): ("malo",
        "compiler_faq.md: 'Why is my code crashing?' sobre troubleshooting "
        "torch.compile. Matcheó 'Why is my...' — tema equivocado."),
    (19, "hybrid"): ("tangencial",
        "faq.rst: total_loss += loss acumulando history. Es sobre pitfall de "
        "training loop (memory) pero no sobre NaN. Mal diagnóstico — el "
        "problema es memory, no numerical instability."),

    # ── Q20: "view vs reshape" ──
    (20, "emb"): ("tangencial",
        "tensor_view.rst: 'views share underlying data... view ops avoid "
        "unnecessary data copy'. Explica views pero no compara con reshape."),
    (20, "bm25"): ("bueno",
        "tensor_view.rst: 'reshape, reshape_as and flatten can return either "
        "a view or new tensor, user code shouldn't rely on whether it's view "
        "or not'. Esta ES la diferencia clave: reshape puede devolver view O "
        "copia, a diferencia de view que siempre devuelve view."),
    (20, "hybrid"): ("bueno",
        "Mismo chunk que BM25. sparse_wins. Contiene el dato clave de reshape."),

    # ── Q21: "how to update learning rate in middle of training" ──
    (21, "emb"): ("bueno",
        "optim.md: ejemplo completo con ExponentialLR, training loop con "
        "scheduler.step() después de cada epoch. Código directamente usable. "
        "'Most learning rate schedulers can be called back-to-back'."),
    (21, "bm25"): ("tangencial",
        "optim.md sobre SWA learning rate schedules: SWALR anneals to fixed "
        "value. Scheduling especializado (SWA), no el uso general de schedulers."),
    (21, "hybrid"): ("tangencial",
        "optim.md sobre SWA/EMA averaged models y SWALR. Mismo problema que "
        "BM25 — scheduling específico de SWA, no general."),
}


def main():
    # Cargar datos
    v1 = load_jsonl(V1_RESULTS)
    bm25 = load_jsonl(BM25_RESULTS)
    hybrid = load_jsonl(HYBRID_RESULTS)

    queries = sorted(v1.keys())

    # ── PREDICCIONES ──────────────────────────────────────────────────────────
    print("=" * 90)
    print("PREDICCIONES (comprometidas antes de clasificar)")
    print("=" * 90)
    print()
    print(f"  v1 embeddings:  {PREDICTIONS['v1_embeddings']} buenos")
    print(f"  BM25-smart:     {PREDICTIONS['bm25_smart']} bueno(s)")
    print(f"  Hybrid RRF:     {PREDICTIONS['hybrid']} buenos")
    print()
    print("Razonamiento:")
    print("  v1: NOTES reportó 2-3/21. Los 3 originales eran Q1, Q9, Q19.")
    print("       Bajo criterio estricto alguno podría caer. Apuesto a 3.")
    print("  BM25: NOTES reportó ~1-2/21. Smart perdió Q12 (save/load).")
    print("        Solo Q9 (consensus con embeddings) debería sobrevivir.")
    print("  Hybrid: NOTES reportó 2/21 (Q9, Q13). RRF demostró perder buenos")
    print("          de embeddings. Mantengo 2.")
    print()

    # ── TABLA COMPLETA ────────────────────────────────────────────────────────
    print("=" * 90)
    print("TABLA COMPARATIVA — 21 queries × 3 sistemas (criterio estricto)")
    print("=" * 90)
    print()

    # Header
    print(f"{'Q':>2}  {'Query':<45}  {'v1 emb':<11} {'BM25-s':<11} {'Hybrid':<11}")
    print(f"{'':>2}  {'':45}  {'source':<11} {'source':<11} {'source':<11}")
    print("-" * 90)

    counts = {"emb": {"bueno": 0, "tangencial": 0, "malo": 0},
              "bm25": {"bueno": 0, "tangencial": 0, "malo": 0},
              "hybrid": {"bueno": 0, "tangencial": 0, "malo": 0}}

    for qid in queries:
        query = v1[qid]["query"]
        query_short = query[:43] + ".." if len(query) > 45 else query

        emb_src = short_source(v1[qid]["chunk_source"])
        bm25_src = short_source(bm25[qid]["bm25_source"])
        hyb_src = short_source(hybrid[qid]["hybrid_source"])

        emb_label = CLASSIFICATIONS[(qid, "emb")][0]
        bm25_label = CLASSIFICATIONS[(qid, "bm25")][0]
        hyb_label = CLASSIFICATIONS[(qid, "hybrid")][0]

        counts["emb"][emb_label] += 1
        counts["bm25"][bm25_label] += 1
        counts["hybrid"][hyb_label] += 1

        # Formato con label entre corchetes
        def cell(label):
            symbols = {"bueno": "●", "tangencial": "◐", "malo": "○"}
            return symbols.get(label, "?")

        print(f"{qid:>2}  {query_short:<45}  "
              f"{cell(emb_label)} {emb_label:<9}  "
              f"{cell(bm25_label)} {bm25_label:<9}  "
              f"{cell(hyb_label)} {hyb_label:<9}")

    print("-" * 90)
    print()

    # ── FUENTES (tabla separada para legibilidad) ─────────────────────────────
    print("=" * 90)
    print("FUENTES — chunk ganador de cada sistema")
    print("=" * 90)
    print()
    print(f"{'Q':>2}  {'v1 embeddings':<35} {'BM25-smart':<35} {'Hybrid RRF':<35}")
    print("-" * 107)

    for qid in queries:
        emb_src = short_source(v1[qid]["chunk_source"])
        bm25_src = short_source(bm25[qid]["bm25_source"])
        hyb_src = short_source(hybrid[qid]["hybrid_source"])

        # Truncar a 33 chars
        def trunc(s, n=33):
            return s[:n-2] + ".." if len(s) > n else s

        print(f"{qid:>2}  {trunc(emb_src):<35} {trunc(bm25_src):<35} {trunc(hyb_src):<35}")

    print("-" * 107)
    print()

    # ── RESUMEN ───────────────────────────────────────────────────────────────
    print("=" * 90)
    print("CONTEOS BAJO CRITERIO ESTRICTO")
    print("=" * 90)
    print()
    print(f"  {'Sistema':<20} {'Bueno':>8} {'Tangencial':>12} {'Malo':>8}")
    print(f"  {'-'*48}")

    actuals = {}
    for sys_key, sys_name in [("emb", "v1 embeddings"), ("bm25", "BM25-smart"), ("hybrid", "Hybrid RRF")]:
        b = counts[sys_key]["bueno"]
        t = counts[sys_key]["tangencial"]
        m = counts[sys_key]["malo"]
        actuals[sys_key] = b
        print(f"  {sys_name:<20} {b:>5}/21  {t:>9}/21  {m:>5}/21")

    print()

    # ── COMPARACIÓN CON PREDICCIONES ──────────────────────────────────────────
    print("=" * 90)
    print("PREDICCIÓN vs RESULTADO")
    print("=" * 90)
    print()

    pred_map = [("emb", "v1_embeddings", "v1 embeddings"),
                ("bm25", "bm25_smart", "BM25-smart"),
                ("hybrid", "hybrid", "Hybrid RRF")]

    any_off_by_more_than_2 = False

    print(f"  {'Sistema':<20} {'Predicción':>12} {'Resultado':>12} {'Δ':>5}")
    print(f"  {'-'*52}")

    for sys_key, pred_key, sys_name in pred_map:
        pred = PREDICTIONS[pred_key]
        actual = actuals[sys_key]
        delta = actual - pred
        marker = " ✓" if abs(delta) <= 2 else " ← ERROR"
        if abs(delta) > 2:
            any_off_by_more_than_2 = True
        print(f"  {sys_name:<20} {pred:>9}/21  {actual:>9}/21  {delta:>+3}{marker}")

    print()

    # ── ANÁLISIS DE ERRORES (si alguno erra por más de 2) ─────────────────────
    if any_off_by_more_than_2:
        print("=" * 90)
        print("ANÁLISIS DE ERROR DE PREDICCIÓN (Δ > 2)")
        print("=" * 90)
        print()
        for sys_key, pred_key, sys_name in pred_map:
            pred = PREDICTIONS[pred_key]
            actual = actuals[sys_key]
            if abs(actual - pred) > 2:
                print(f"  {sys_name}: predije {pred}, obtuve {actual} (Δ={actual-pred:+d})")
                print()
                print("  PARANDO PARA EXPLICAR:")
                print()
                _explain_error(sys_key, pred, actual, queries, v1, bm25, hybrid)
                print()
    else:
        print("  Todas las predicciones dentro del margen de ±2. Sin errores graves.")
        print()

    # ── DETALLE DE CLASIFICACIONES ────────────────────────────────────────────
    print("=" * 90)
    print("DETALLE — razón de cada clasificación")
    print("=" * 90)
    print()

    for qid in queries:
        query = v1[qid]["query"]
        cat = v1[qid]["category"]
        expected = v1[qid]["expected"]
        print(f"Q{qid} — \"{query}\" [{cat}, expected={expected}]")
        print()
        for sys_key, sys_name in [("emb", "v1 embeddings"), ("bm25", "BM25-smart"), ("hybrid", "Hybrid RRF")]:
            label, reason = CLASSIFICATIONS[(qid, sys_key)]
            symbols = {"bueno": "●", "tangencial": "◐", "malo": "○"}
            print(f"  {symbols[label]} {sys_name:<15} → {label}")
            # Wrap reason at ~72 chars
            words = reason.split()
            line = "    "
            for w in words:
                if len(line) + len(w) + 1 > 76:
                    print(line)
                    line = "    " + w
                else:
                    line += " " + w if line.strip() else "    " + w
            if line.strip():
                print(line)
        print()

    # ── PATRONES OBSERVADOS ───────────────────────────────────────────────────
    print("=" * 90)
    print("PATRONES OBSERVADOS")
    print("=" * 90)
    print()
    print("1. Q9 (leaf tensor) es el ÚNICO bueno en los tres sistemas.")
    print("   Es una definición directa, sin ambigüedad, con chunk perfecto.")
    print("   Ambos retrievers lo rankean #1 → consensus en hybrid.")
    print()
    print("2. v1 embeddings tiene 2 buenos exclusivos que los otros pierden:")
    print("   - Q1 (autograd): embeddings encuentra la definición en autograd.rst.")
    print("     BM25 va a distributed_autograd (más repeticiones de 'autograd').")
    print("     Hybrid pierde el chunk por RRF compromise.")
    print("   - Q19 (loss NaN): embeddings encuentra el ejemplo de div/0→NaN.")
    print("     BM25 matchea 'Why is my...' en compiler_faq (tema equivocado).")
    print("     Hybrid va a faq.rst (memory accumulation, no NaN).")
    print()
    print("3. v1 embeddings gana en Q21 (LR scheduling) con ejemplo completo.")
    print("   BM25 y hybrid van a SWA-specific scheduling, no general.")
    print("   Embeddings captura la semántica 'update learning rate' → schedulers.")
    print()
    print("4. BM25-smart tiene un bueno exclusivo: Q20 (view vs reshape).")
    print("   El smart tokenizer libera 'reshape' de referencias compuestas.")
    print("   Hybrid hereda este bueno (sparse_wins).")
    print()
    print("5. Hybrid tiene un bueno exclusivo: Q13 (numpy→tensor).")
    print("   tensors.rst con torch.tensor(np.array(...)) estaba en posición 14")
    print("   de embeddings y 2 de BM25 — ninguno lo elegía como top-1.")
    print("   RRF lo surfaceó por ser decente en ambos rankings.")
    print()
    print("6. El patrón más revelador: hybrid MOVIÓ queries de malo a tangencial")
    print("   (Q1, Q6, Q19) pero no de tangencial a bueno. RRF suaviza los")
    print("   extremos — menos malos, más tangenciales, pocos buenos nuevos.")
    print()
    print("7. Las 5 queries OOD/opinion (Q4,Q5,Q6,Q16,Q17) son consistentemente")
    print("   malo en los tres sistemas, excepto Q6 hybrid (tangencial por mostrar")
    print("   un training loop con SGD) y Q17 en los tres (tangencial por describir")
    print("   PyTorch). El criterio estricto en queries que deberían rechazarse")
    print("   muestra que el retriever no aporta nada útil — como se espera.")
    print()


def _explain_error(sys_key, pred, actual, queries, v1, bm25, hybrid):
    """Genera párrafo explicativo cuando la predicción erra por más de 2."""
    buenos = []
    for qid in queries:
        label = CLASSIFICATIONS[(qid, sys_key)][0]
        if label == "bueno":
            buenos.append(qid)

    sys_names = {"emb": "v1 embeddings", "bm25": "BM25-smart", "hybrid": "Hybrid RRF"}
    name = sys_names[sys_key]

    print(f"  Buenos encontrados en {name}: {', '.join(f'Q{q}' for q in buenos)}")
    print()

    if actual > pred:
        print(f"  Subestimé {name} por {actual - pred}. Los buenos que no predije:")
        for qid in buenos:
            was_expected = False
            if sys_key == "emb" and qid in [1, 9, 19]:
                was_expected = True
            elif sys_key == "bm25" and qid in [9]:
                was_expected = True
            elif sys_key == "hybrid" and qid in [9, 13]:
                was_expected = True
            if not was_expected:
                reason = CLASSIFICATIONS[(qid, sys_key)][1]
                print(f"    Q{qid}: {reason[:72]}...")
    else:
        print(f"  Sobreestimé {name} por {pred - actual}.")
        print(f"  Buenos esperados que no se materializaron:")
        # Identify what was expected but not bueno
        for qid in queries:
            label = CLASSIFICATIONS[(qid, sys_key)][0]
            if label != "bueno":
                # Check if this was likely in the prediction
                if sys_key == "emb" and qid in [1, 9, 19]:
                    if label != "bueno":
                        reason = CLASSIFICATIONS[(qid, sys_key)][1]
                        print(f"    Q{qid} ({label}): {reason[:72]}...")


if __name__ == "__main__":
    main()
