"""
benchmark_salamandra_fix.py — Re-run Salamandra with corrected chat template.
Salamandra (Mistral-based) requires <s> BOS token before the ChatML prompt.
"""
import sys, json, re, gc
import numpy as np
from pathlib import Path

sys.path.insert(0, "/opt/homebrew/Cellar/mlx/0.31.2/lib/python3.14/site-packages")
import mlx.core as mx
from mlx_lm import load, generate

BASE    = Path(__file__).resolve().parent
RES     = BASE / "results"
SAL_PATH = BASE / "models/salamandra-7b-4bit"

SYSTEM = (
    "Eres un tasador inmobiliario experto en Monterrey, México. "
    "Responde ÚNICAMENTE con el precio de renta estimado en MXN/mes. "
    "Formato obligatorio: solo el número, sin explicaciones ni texto adicional. "
    "Ejemplo: 22500"
)

def prompt_salamandra(system: str, user: str) -> str:
    # Salamandra is Mistral-based: needs <s> BOS + ChatML format
    return (
        f"<s><|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def extract_price(text: str):
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

def find_similar(q, train_qs, k=3):
    scored = []
    for t in train_qs:
        score = 0
        if q.get('municipio') and q.get('municipio') == t.get('municipio'): score += 1
        if q.get('colonia')   and q.get('colonia')   == t.get('colonia'):   score += 10
        if q.get('recamaras') and t.get('recamaras'):
            score -= abs(q['recamaras'] - t['recamaras'])
        if q.get('m2') and t.get('m2') and q['m2'] > 0 and t['m2'] > 0:
            score -= abs(q['m2'] - t['m2']) / max(q['m2'], t['m2'])
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:k]]

def examples_text(items):
    lines = ""
    for i, t in enumerate(items, 1):
        attrs = []
        if t.get('m2') and t['m2'] > 1:   attrs.append(f"{t['m2']:.0f}m²")
        if t.get('recamaras'):             attrs.append(f"{t['recamaras']:.0f} rec")
        if t.get('amueblado'):             attrs.append("amueblado")
        if t.get('lujo'):                  attrs.append("lujo")
        lines += f"  {i}. {t.get('colonia','?')} — {', '.join(attrs) or 'sin datos'} → ${t['precio_real']:,.0f} MXN/mes\n"
    return lines

def metricas(results, nombre):
    errs   = [r["error_pct"] for r in results]
    p_resp = [r for r in results if r["precio_predicho"] is not None]
    print(f"\n{'='*60}")
    print(f"  {nombre}")
    print(f"{'='*60}")
    print(f"  MAE% (penalizado)  : {np.mean(errs):.1f}%")
    print(f"  Dentro de ±15%     : {sum(1 for e in errs if e<=15)/len(errs)*100:.1f}%")
    print(f"  Dentro de ±20%     : {sum(1 for e in errs if e<=20)/len(errs)*100:.1f}%")
    print(f"  Non-response rate  : {sum(1 for r in results if r['precio_predicho'] is None)/len(results)*100:.1f}%")
    print(f"  Respondidas        : {len(p_resp)}/{len(results)}")

# ── Load data ─────────────────────────────────────────────────────────────────
test_qs   = [json.loads(l) for l in open(BASE / "data/benchmark_test.jsonl")]
train_qs  = [json.loads(l) for l in open(BASE / "data/benchmark_train.jsonl")]
xgb_prices = {
    r['id']: r['precio_xgb']
    for r in [json.loads(l) for l in open(RES / "benchmark_base_hybrid_test.jsonl")]
}

print(f"Cargando Salamandra desde {SAL_PATH}...")
model, tokenizer = load(str(SAL_PATH))
print("✓ Listo\n")

n = len(test_qs)

# ── Config 1: Zero-shot ────────────────────────────────────────────────────────
print("Config 1/3: Zero-shot...")
results_zs = []
for i, q in enumerate(test_qs):
    if i % 20 == 0: print(f"  {i}/{n}")
    prompt = prompt_salamandra(SYSTEM, q['pregunta'])
    resp   = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
    pred   = extract_price(resp)
    pr     = q.get('precio_real')
    err    = abs(pred - pr) / pr * 100 if pred and pr else 100.0
    results_zs.append({"id": q.get("id"), "precio_real": pr,
                        "precio_predicho": pred, "respuesta_raw": resp.strip(),
                        "error_pct": err, "modelo": "salamandra_7b", "config": "zeroshot"})
metricas(results_zs, "Salamandra 7B (fixed) — ZERO-SHOT")

# ── Config 2: Few-shot ─────────────────────────────────────────────────────────
print("\nConfig 2/3: Few-shot (k=3)...")
results_fs = []
for i, q in enumerate(test_qs):
    if i % 20 == 0: print(f"  {i}/{n}")
    ejemplos = find_similar(q, train_qs, k=3)
    user_msg = (f"Propiedades similares en el mercado de Monterrey:\n{examples_text(ejemplos)}\n"
                f"Basándote en estos ejemplos, responde:\n{q['pregunta']}")
    prompt = prompt_salamandra(SYSTEM, user_msg)
    resp   = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
    pred   = extract_price(resp)
    pr     = q.get('precio_real')
    err    = abs(pred - pr) / pr * 100 if pred and pr else 100.0
    results_fs.append({"id": q.get("id"), "precio_real": pr,
                       "precio_predicho": pred, "respuesta_raw": resp.strip(),
                       "error_pct": err, "modelo": "salamandra_7b", "config": "fewshot"})
metricas(results_fs, "Salamandra 7B (fixed) — FEW-SHOT (k=3)")

# ── Config 3: Few-shot + XGBoost ───────────────────────────────────────────────
print("\nConfig 3/3: Few-shot + XGBoost anchor...")
results_xgb = []
for i, q in enumerate(test_qs):
    if i % 20 == 0: print(f"  {i}/{n}")
    ejemplos   = find_similar(q, train_qs, k=3)
    precio_xgb = xgb_prices.get(q['id'], 0)
    user_msg = (f"Propiedades similares en el mercado:\n{examples_text(ejemplos)}\n"
                f"Precio calculado por modelo estadístico (XGBoost): ${precio_xgb:,.0f} MXN/mes\n\n"
                f"Basándote en los ejemplos y el precio estadístico, responde:\n{q['pregunta']}")
    prompt = prompt_salamandra(SYSTEM, user_msg)
    resp   = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
    pred   = extract_price(resp)
    pr     = q.get('precio_real')
    err    = abs(pred - pr) / pr * 100 if pred and pr else 100.0
    results_xgb.append({"id": q.get("id"), "precio_real": pr, "precio_xgb": precio_xgb,
                        "precio_predicho": pred, "respuesta_raw": resp.strip(),
                        "error_pct": err, "modelo": "salamandra_7b", "config": "fewshot_xgb"})
metricas(results_xgb, "Salamandra 7B (fixed) — FEW-SHOT + XGBOOST ANCHOR")

# ── Save ───────────────────────────────────────────────────────────────────────
for results, config_name in [(results_zs, "zeroshot"), (results_fs, "fewshot"), (results_xgb, "fewshot_xgb")]:
    out = RES / f"benchmark_salamandra_7b_{config_name}_test.jsonl"
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"✓ Guardado: {out}")

del model, tokenizer
gc.collect()
mx.metal.clear_cache()
print("\n✓ Salamandra (fixed) — done")
