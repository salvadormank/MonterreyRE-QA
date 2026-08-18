"""
benchmark_bm25.py — BM25 retrieval baseline vs RAG y Few-shot
Misma arquitectura: recupera k=3 ejemplos del train set, los inyecta
como few-shot en el mismo prompt que usan RAG y Few-shot.
"""
import json, re, time, numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

BASE = Path(__file__).resolve().parent
RES  = BASE / "results"

# ── Cargar datos ────────────────────────────────────────────────────────────

train_qs = [json.loads(l) for l in open(BASE / "data/benchmark_train.jsonl")]
test_qs  = [json.loads(l) for l in open(BASE / "data/benchmark_test.jsonl")]

# ── Construir corpus BM25 ────────────────────────────────────────────────────

def tokenize(text):
    text = (text or "").lower()
    text = re.sub(r"[^\w\sáéíóúüñ]", " ", text)
    return text.split()

# Cada documento = pregunta + colonia + municipio (enriquecido con metadata)
def doc_text(q):
    parts = [q.get("pregunta", "")]
    if q.get("colonia"):   parts.append(q["colonia"])
    if q.get("municipio"): parts.append(q["municipio"])
    if q.get("recamaras"): parts.append(f"{q['recamaras']} recamaras")
    if q.get("m2"):        parts.append(f"{q['m2']} metros")
    return " ".join(str(p) for p in parts)

corpus_tokens = [tokenize(doc_text(q)) for q in train_qs]
bm25 = BM25Okapi(corpus_tokens)
print(f"BM25 corpus: {len(corpus_tokens)} documentos")

# ── Retrieval y formato ──────────────────────────────────────────────────────

def retrieve_bm25(pregunta, k=3):
    tokens = tokenize(pregunta)
    scores = bm25.get_scores(tokens)
    top_k  = np.argsort(scores)[::-1][:k]
    return [train_qs[i] for i in top_k]

def format_examples(ejemplos):
    lines = ["Propiedades similares en el mercado de Monterrey:"]
    for i, e in enumerate(ejemplos, 1):
        col  = e.get("colonia", "?")
        mun  = e.get("municipio", "")
        m2   = f"{e['m2']}m²" if e.get("m2") else ""
        rec  = f"{int(e['recamaras'])} rec" if e.get("recamaras") else ""
        luj  = "lujo" if e.get("lujo") else ""
        amue = "amueblado" if e.get("amueblado") else ""
        tags = ", ".join(x for x in [m2, rec, luj, amue] if x)
        precio = f"${e['precio_real']:,.0f} MXN/mes"
        lines.append(f"  {i}. {col}, {mun} — {tags} → {precio}")
    return "\n".join(lines)

# ── Inferencia con Qwen ──────────────────────────────────────────────────────

import sys
sys.path.insert(0, str(BASE))

import mlx.core as mx
from mlx_lm import load, generate

MODEL_PATH = "mlx-community/Qwen2.5-7B-Instruct-4bit"
print("Cargando modelo...", flush=True)
model, tokenizer = load(MODEL_PATH)
print("Modelo cargado.", flush=True)

SYSTEM = (
    "Eres un experto en el mercado inmobiliario de renta en Monterrey, "
    "Nuevo León. Respondes en español con argumentos concretos basados "
    "en el mercado local."
)

def extract_price(text):
    patterns = [
        r'\$\s*([\d,]+(?:\.\d+)?)\s*(?:MXN|pesos)?',
        r'([\d,]+(?:\.\d+)?)\s*(?:MXN|pesos)',
        r'\b(\d{4,6})\b',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            val = float(m.group(1).replace(',', ''))
            if 3000 <= val <= 150000:
                return val
    return None

def ask_llm(pregunta, ejemplos_txt):
    prompt = (
        f"{ejemplos_txt}\n\n"
        f"Basándote en estos ejemplos, responde: {pregunta}\n"
        "Indica el precio estimado en MXN/mes."
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = generate(model, tokenizer, prompt=text,
                        max_tokens=120, verbose=False)
    return response

# ── Evaluación ───────────────────────────────────────────────────────────────

results = []
n = len(test_qs)

print(f"\nEvaluando {n} preguntas con BM25+LLM...\n")
for i, q in enumerate(test_qs):
    if i % 10 == 0:
        print(f"  {i}/{n}", flush=True)

    ejemplos    = retrieve_bm25(q["pregunta"], k=3)
    ejemplos_txt = format_examples(ejemplos)
    respuesta   = ask_llm(q["pregunta"], ejemplos_txt)
    pred        = extract_price(respuesta)

    precio_real = q.get("precio_real")
    err = abs(pred - precio_real) / precio_real * 100 if pred and precio_real else 100.0

    results.append({
        "id": q["id"],
        "precio_real": precio_real,
        "precio_predicho": pred,
        "respuesta_raw": respuesta,
        "error_pct": err,
        "modelo": "bm25_qwen",
        "ejemplos_recuperados": [e["id"] for e in ejemplos],
    })

# ── Métricas ─────────────────────────────────────────────────────────────────

errs    = [r["error_pct"] for r in results]
resp_ok = [r for r in results if r["precio_predicho"] is not None]

print(f"\n{'='*55}")
print(f"  BM25 + LLM (Qwen 7B, k=3)")
print(f"{'='*55}")
print(f"  MAE% (penalizado)  : {np.mean(errs):.1f}%")
print(f"  Dentro de ±15%     : {sum(1 for e in errs if e<=15)/len(errs)*100:.1f}%")
print(f"  Dentro de ±20%     : {sum(1 for e in errs if e<=20)/len(errs)*100:.1f}%")
print(f"  Non-response rate  : {sum(1 for r in results if r['precio_predicho'] is None)/len(results)*100:.1f}%")
print(f"  Respondidas        : {len(resp_ok)}/{len(results)}")

# Comparativa rápida
print(f"\n{'─'*55}")
print(f"  Comparativa retrieval")
print(f"{'─'*55}")

for fname, label in [
    ("benchmark_base_fewshot_test.jsonl", "Few-shot (reglas)  "),
    ("benchmark_base_rag_test.jsonl",     "RAG (semántico)    "),
]:
    path = RES / fname
    if path.exists():
        r2 = [json.loads(l) for l in open(path)]
        e2 = [x["error_pct"] for x in r2]
        print(f"  {label}: MAE%={np.mean(e2):.1f}%  ±15%={sum(1 for x in e2 if x<=15)/len(e2)*100:.1f}%")

print(f"  BM25 + LLM         : MAE%={np.mean(errs):.1f}%  ±15%={sum(1 for e in errs if e<=15)/len(errs)*100:.1f}%")

# Guardar
out = RES / "benchmark_bm25_test.jsonl"
with open(out, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\n✓ Guardado: {out}")
