"""
benchmark_cai.py — Evalúa el modelo fine-tuneado con datos SL-CAI en 109 preguntas de test.
Compara directamente contra benchmark_finetuned_test.jsonl (LoRA original).
"""
import sys, json, re
import numpy as np
from pathlib import Path

sys.path.insert(0, "/opt/homebrew/Cellar/mlx/0.31.2/lib/python3.14/site-packages")
from mlx_lm import load, generate

BASE = Path(__file__).resolve().parent
RES  = BASE / "results"

SYSTEM = (
    "Eres un experto en el mercado inmobiliario de renta en Monterrey, "
    "Nuevo León. Respondes en español con argumentos concretos basados "
    "en el mercado local."
)

def extract_price(text):
    patterns = [
        r'\$\s*([\d,]+(?:\.\d+)?)\s*(?:MXN|pesos|al mes|/mes)?',
        r'([\d,]+(?:\.\d+)?)\s*(?:MXN|pesos)',
        r'\b([\d]{4,6})\b',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            val = float(m.group(1).replace(',', ''))
            if 3000 <= val <= 200000:
                return val
    return None

print("Cargando Qwen + adapter CAI...", flush=True)
model, tokenizer = load(
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    adapter_path=str(BASE / "adapters_cai")
)
print("✓ Modelo CAI listo\n", flush=True)

test_qs = [json.loads(l) for l in open(BASE / "data/benchmark_test.jsonl")]
n = len(test_qs)
results = []

for i, q in enumerate(test_qs):
    if i % 10 == 0:
        print(f"  {i}/{n}", flush=True)

    prompt = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{q['pregunta']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    resp = generate(model, tokenizer, prompt=prompt, max_tokens=300, verbose=False)
    pred = extract_price(resp)
    real = q.get("precio_real")
    err  = abs(pred - real) / real * 100 if pred and real else 100.0

    results.append({
        "id":              q["id"],
        "tipo":            q.get("tipo"),
        "pregunta":        q["pregunta"],
        "precio_real":     real,
        "colonia":         q.get("colonia"),
        "municipio":       q.get("municipio"),
        "recamaras":       q.get("recamaras"),
        "m2":              q.get("m2"),
        "amueblado":       q.get("amueblado", 0),
        "lujo":            q.get("lujo", 0),
        "split":           "test",
        "respuesta":       resp.strip(),
        "precio_predicho": pred,
        "error_pct":       err,
        "dentro_15pct":    err <= 15,
        "dentro_20pct":    err <= 20,
        "modelo":          "lora_cai",
    })

# ── Métricas ──────────────────────────────────────────────────────────────────
errs   = [r["error_pct"] for r in results]
nr     = sum(1 for r in results if r["precio_predicho"] is None)
p15    = sum(1 for e in errs if e <= 15)
p20    = sum(1 for e in errs if e <= 20)

print(f"\n{'='*55}")
print(f"  LoRA fine-tuned (SL-CAI data) — 109 preguntas")
print(f"{'='*55}")
print(f"  MAE% (penalizado)  : {np.mean(errs):.1f}%")
print(f"  Dentro de ±15%     : {p15/n*100:.1f}%")
print(f"  Dentro de ±20%     : {p20/n*100:.1f}%")
print(f"  Non-response rate  : {nr/n*100:.1f}%")
print(f"  Respondidas        : {n-nr}/{n}")

# ── Comparación con LoRA original ────────────────────────────────────────────
orig = [json.loads(l) for l in open(RES / "benchmark_finetuned_test.jsonl")]
o_errs = [r["error_pct"] for r in orig]
o_nr   = sum(1 for r in orig if r["precio_predicho"] is None)
o_p15  = sum(1 for e in o_errs if e <= 15)
o_p20  = sum(1 for e in o_errs if e <= 20)

print(f"\n{'='*55}")
print(f"  COMPARACIÓN DIRECTA")
print(f"{'='*55}")
print(f"  {'Métrica':<22} {'LoRA original':>14} {'LoRA CAI':>10}")
print(f"  {'-'*48}")
print(f"  {'MAE%':<22} {np.mean(o_errs):>13.1f}% {np.mean(errs):>9.1f}%")
print(f"  {'Dentro ±15%':<22} {o_p15/n*100:>13.1f}% {p15/n*100:>9.1f}%")
print(f"  {'Dentro ±20%':<22} {o_p20/n*100:>13.1f}% {p20/n*100:>9.1f}%")
print(f"  {'Non-response':<22} {o_nr/n*100:>13.1f}% {nr/n*100:>9.1f}%")

# ── Guardar ──────────────────────────────────────────────────────────────────
out = RES / "benchmark_cai_test.jsonl"
with open(out, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
print(f"\n✓ Guardado: {out}")
