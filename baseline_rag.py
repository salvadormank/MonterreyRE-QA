"""
baseline_rag.py — RAG baseline
Embeds las 872 propiedades de train con sentence-transformers,
recupera las 5 más similares a cada pregunta de test y las inyecta como contexto.
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

MODEL_ID   = "mlx-community/Qwen2.5-7B-Instruct-4bit"
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
    colonia = q.get('colonia', '')
    municipio = q.get('municipio', '')
    return f"{colonia} {municipio} {' '.join(attrs)}"

def build_rag_prompt(q, retrieved):
    ctx = ""
    for i, (t, score) in enumerate(retrieved, 1):
        attrs = []
        if t.get('m2') and t['m2'] > 1:
            attrs.append(f"{t['m2']:.0f}m²")
        if t.get('recamaras'):
            attrs.append(f"{t['recamaras']:.0f} rec")
        if t.get('amueblado'):
            attrs.append("amueblado")
        if t.get('lujo'):
            attrs.append("lujo")
        attr_str = ", ".join(attrs) if attrs else "sin datos"
        ctx += f"  {i}. {t.get('colonia','?')}, {t.get('municipio','?')} — {attr_str} → ${t['precio_real']:,.0f} MXN/mes\n"

    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Propiedades reales del mercado de Monterrey más similares a la consulta:\n{ctx}\n"
        f"Usando estas propiedades como referencia, responde:\n{q['pregunta']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

# ── Cargar datos ──────────────────────────────────────────────────────────────
test_qs  = [json.loads(l) for l in open(BASE / "data/benchmark_test.jsonl")]
train_qs = [json.loads(l) for l in open(BASE / "data/benchmark_train.jsonl")]

# ── Embeddings ────────────────────────────────────────────────────────────────
print("Cargando modelo de embeddings...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Generando embeddings del corpus de train...")
train_texts = [prop_to_text(t) for t in train_qs]
train_embs  = embedder.encode(train_texts, batch_size=64, show_progress_bar=True)

print("Generando embeddings de test...")
test_texts = [prop_to_text(q) for q in test_qs]
test_embs  = embedder.encode(test_texts, batch_size=64, show_progress_bar=True)

# ── Cargar LLM ────────────────────────────────────────────────────────────────
print("\nCargando Qwen BASE para RAG...")
model, tokenizer = load(MODEL_ID)
print("✓ Listo\n")

# ── Inferencia ────────────────────────────────────────────────────────────────
results = []
n = len(test_qs)
K = 5  # propiedades a recuperar

for i, q in enumerate(test_qs):
    if i % 10 == 0:
        print(f"  {i}/{n} ({i/n*100:.0f}%)")

    sims = cosine_similarity([test_embs[i]], train_embs)[0]
    top_idx = np.argsort(sims)[::-1][:K]
    retrieved = [(train_qs[j], float(sims[j])) for j in top_idx]

    prompt = build_rag_prompt(q, retrieved)
    resp   = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
    precio_pred = extract_price(resp)
    precio_real = q.get('precio_real')

    err = abs(precio_pred - precio_real) / precio_real * 100 \
          if precio_pred and precio_real else 100.0

    results.append({
        "id": q.get("id"),
        "pregunta": q["pregunta"],
        "precio_real": precio_real,
        "precio_predicho": precio_pred,
        "respuesta_raw": resp.strip(),
        "error_pct": err,
        "retrieved_colonias": [t.get('colonia') for t, _ in retrieved],
        "modelo": "base_rag"
    })

# ── Métricas ──────────────────────────────────────────────────────────────────
errs   = [r["error_pct"] for r in results]
p_resp = [r for r in results if r["precio_predicho"] is not None]
mae_pct  = np.mean(errs)
p15 = sum(1 for e in errs if e <= 15) / len(errs) * 100
p20 = sum(1 for e in errs if e <= 20) / len(errs) * 100
non_resp = sum(1 for r in results if r["precio_predicho"] is None) / len(results) * 100

print(f"\n{'='*55}")
print("  BASE LLM — RAG (top-5 propiedades similares)")
print(f"{'='*55}")
print(f"  MAE% (penalizado)  : {mae_pct:.1f}%")
print(f"  Dentro de ±15%     : {p15:.1f}%")
print(f"  Dentro de ±20%     : {p20:.1f}%")
print(f"  Non-response rate  : {non_resp:.1f}%")
print(f"  Respondidas        : {len(p_resp)}/{len(results)}")

out = RES / "benchmark_base_rag_test.jsonl"
with open(out, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
print(f"\n✓ Guardado: {out}")
