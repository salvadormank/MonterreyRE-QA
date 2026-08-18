"""
baseline_fewshot_rag_xgb.py — Few-shot+XGBoost y RAG+XGBoost
Combina contexto de ejemplos similares con el anchor numérico de XGBoost.
"""
import sys, json, re
import numpy as np
from pathlib import Path

sys.path.insert(0, "/opt/homebrew/Cellar/mlx/0.31.2/lib/python3.14/site-packages")
from mlx_lm import load, generate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parent
RES  = BASE / "results"

MODEL_ID    = "mlx-community/Qwen2.5-7B-Instruct-4bit"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

SYSTEM = (
    "Eres un tasador inmobiliario experto en Monterrey, México. "
    "Responde ÚNICAMENTE con el precio de renta estimado en MXN/mes. "
    "Formato obligatorio: solo el número, sin explicaciones ni texto adicional. "
    "Ejemplo: 22500"
)

def extract_price(text):
    patterns = [
        r'^\s*\$?\s*([\d,]+(?:\.\d+)?)\s*$',
        r'\$\s*([\d,]+(?:\.\d+)?)\s*(?:MXN|pesos)?',
        r'([\d,]+(?:\.\d+)?)\s*(?:MXN|pesos)',
        r'\b([\d]{4,6})\b',
    ]
    for p in patterns:
        m = re.search(p, text.strip(), re.IGNORECASE | re.MULTILINE)
        if m:
            val = float(m.group(1).replace(',', ''))
            if 3000 <= val <= 150000:
                return val
    return None

def prop_to_text(q):
    attrs = []
    if q.get('m2') and q['m2'] > 1:
        attrs.append(f"{q['m2']:.0f}m²")
    if q.get('recamaras'):
        attrs.append(f"{q['recamaras']:.0f} recámaras")
    if q.get('amueblado'):
        attrs.append("amueblado")
    if q.get('lujo'):
        attrs.append("lujo")
    return f"{q.get('colonia','')} {q.get('municipio','')} {' '.join(attrs)}"

def examples_text(items):
    lines = ""
    for i, t in enumerate(items, 1):
        attrs = []
        if t.get('m2') and t['m2'] > 1: attrs.append(f"{t['m2']:.0f}m²")
        if t.get('recamaras'): attrs.append(f"{t['recamaras']:.0f} rec")
        if t.get('amueblado'): attrs.append("amueblado")
        if t.get('lujo'): attrs.append("lujo")
        lines += f"  {i}. {t.get('colonia','?')} — {', '.join(attrs)} → ${t['precio_real']:,.0f} MXN/mes\n"
    return lines

def find_similar_structured(q, train_qs, k=3):
    scored = []
    for t in train_qs:
        score = 0
        if q.get('municipio') and q.get('municipio') == t.get('municipio'): score += 1
        if q.get('colonia') and q.get('colonia') == t.get('colonia'): score += 10
        if q.get('recamaras') and t.get('recamaras'):
            score -= abs(q['recamaras'] - t['recamaras'])
        if q.get('m2') and t.get('m2') and q['m2'] > 0 and t['m2'] > 0:
            score -= abs(q['m2'] - t['m2']) / max(q['m2'], t['m2'])
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:k]]

# ── Cargar datos ──────────────────────────────────────────────────────────────
test_qs   = [json.loads(l) for l in open(BASE / "data/benchmark_test.jsonl")]
train_qs  = [json.loads(l) for l in open(BASE / "data/benchmark_train.jsonl")]
xgb_prices = {r['id']: r['precio_xgb']
               for r in [json.loads(l) for l in open(RES / "benchmark_base_hybrid_test.jsonl")]}

# ── Embeddings para RAG ───────────────────────────────────────────────────────
print("Cargando embeddings...")
embedder    = SentenceTransformer(EMBED_MODEL)
train_texts = [prop_to_text(t) for t in train_qs]
train_embs  = embedder.encode(train_texts, batch_size=64, show_progress_bar=False)
test_texts  = [prop_to_text(q) for q in test_qs]
test_embs   = embedder.encode(test_texts, batch_size=64, show_progress_bar=False)
print("✓ Embeddings listos")

# ── Cargar LLM ────────────────────────────────────────────────────────────────
print("Cargando Qwen BASE...")
model, tokenizer = load(MODEL_ID)
print("✓ Listo\n")

# ── Inferencia ────────────────────────────────────────────────────────────────
results_fs_xgb  = []
results_rag_xgb = []
n = len(test_qs)

for i, q in enumerate(test_qs):
    if i % 10 == 0:
        print(f"  {i}/{n} ({i/n*100:.0f}%)")

    precio_xgb  = xgb_prices.get(q['id'], 0)
    precio_real = q.get('precio_real')

    # ── Few-shot + XGBoost ────────────────────────────────────────────────────
    ejemplos = find_similar_structured(q, train_qs, k=3)
    ej_text  = examples_text(ejemplos)
    prompt_fs = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Propiedades similares en el mercado:\n{ej_text}\n"
        f"Precio calculado por modelo estadístico (XGBoost): ${precio_xgb:,.0f} MXN/mes\n\n"
        f"Basándote en los ejemplos y el precio estadístico, responde:\n{q['pregunta']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    resp_fs = generate(model, tokenizer, prompt=prompt_fs, max_tokens=20, verbose=False)
    pred_fs = extract_price(resp_fs)
    err_fs  = abs(pred_fs - precio_real) / precio_real * 100 if pred_fs and precio_real else 100.0
    results_fs_xgb.append({
        "id": q['id'], "precio_real": precio_real, "precio_xgb": precio_xgb,
        "precio_predicho": pred_fs, "respuesta_raw": resp_fs.strip(),
        "error_pct": err_fs, "modelo": "fewshot_xgb"
    })

    # ── RAG + XGBoost ─────────────────────────────────────────────────────────
    sims    = cosine_similarity([test_embs[i]], train_embs)[0]
    top_idx = np.argsort(sims)[::-1][:5]
    retrieved = [train_qs[j] for j in top_idx]
    rag_text  = examples_text(retrieved)
    prompt_rag = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Propiedades similares recuperadas del mercado:\n{rag_text}\n"
        f"Precio calculado por modelo estadístico (XGBoost): ${precio_xgb:,.0f} MXN/mes\n\n"
        f"Basándote en las propiedades similares y el precio estadístico, responde:\n{q['pregunta']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    resp_rag = generate(model, tokenizer, prompt=prompt_rag, max_tokens=20, verbose=False)
    pred_rag = extract_price(resp_rag)
    err_rag  = abs(pred_rag - precio_real) / precio_real * 100 if pred_rag and precio_real else 100.0
    results_rag_xgb.append({
        "id": q['id'], "precio_real": precio_real, "precio_xgb": precio_xgb,
        "precio_predicho": pred_rag, "respuesta_raw": resp_rag.strip(),
        "error_pct": err_rag, "modelo": "rag_xgb"
    })

# ── Métricas ──────────────────────────────────────────────────────────────────
def metricas(results, nombre):
    errs   = [r["error_pct"] for r in results]
    p_resp = [r for r in results if r["precio_predicho"] is not None]
    print(f"\n{'='*55}")
    print(f"  {nombre}")
    print(f"{'='*55}")
    print(f"  MAE% (penalizado)  : {np.mean(errs):.1f}%")
    print(f"  Dentro de ±15%     : {sum(1 for e in errs if e<=15)/len(errs)*100:.1f}%")
    print(f"  Dentro de ±20%     : {sum(1 for e in errs if e<=20)/len(errs)*100:.1f}%")
    print(f"  Non-response rate  : {sum(1 for r in results if r['precio_predicho'] is None)/len(results)*100:.1f}%")
    print(f"  Respondidas        : {len(p_resp)}/{len(results)}")

metricas(results_fs_xgb,  "FEW-SHOT + XGBOOST ANCHOR")
metricas(results_rag_xgb, "RAG + XGBOOST ANCHOR")

for results, fname in [(results_fs_xgb, "benchmark_fewshot_xgb_test.jsonl"),
                        (results_rag_xgb, "benchmark_rag_xgb_test.jsonl")]:
    out = RES / fname
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"✓ Guardado: {out}")
