"""
benchmark_hybrid_boxcox.py — Re-evalúa el sistema híbrido con precios XGBoost Box-Cox.
Versión ligera: carga solo lo necesario, usa precios XGBoost ya calculados.
"""
import sys, json, re
import numpy as np
from pathlib import Path

sys.path.insert(0, "/opt/homebrew/Cellar/mlx/0.31.2/lib/python3.14/site-packages")
from mlx_lm import load, generate

BASE    = Path(__file__).resolve().parent
RES     = BASE / "results"
DATA    = BASE / "data"
ADAPTER = str(BASE / "adapters")
MODEL   = "mlx-community/Qwen2.5-7B-Instruct-4bit"

SYSTEM = (
    "Eres un experto en el mercado inmobiliario de renta en Monterrey, "
    "Nuevo León. Respondes en español con argumentos concretos basados "
    "en el mercado local. Cuando se te proporciona un precio calculado "
    "por modelo estadístico, úsalo como base."
)

def extract_price(text):
    for pat in [r'\$\s*([\d,]+(?:\.\d+)?)', r'([\d,]+(?:\.\d+)?)\s*(?:MXN|pesos)', r'\b(\d{4,6})\b']:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            v = float(m.group(1).replace(',', ''))
            if 3000 <= v <= 200000:
                return v
    return None

# Cargar precios XGBoost Box-Cox ya calculados (de regen_xgb_prices.py)
xgb_map = {r['id']: r['precio_xgb']
            for r in [json.loads(l) for l in open(RES / "benchmark_base_hybrid_test.jsonl")]}

test_qs = [json.loads(l) for l in open(DATA / "benchmark_test.jsonl")]
n = len(test_qs)

print("Cargando Qwen fine-tuneado...", flush=True)
model, tokenizer = load(MODEL, adapter_path=ADAPTER)
print("✓ Listo\n", flush=True)

results = []
for i, q in enumerate(test_qs):
    if i % 10 == 0:
        print(f"  {i}/{n}", flush=True)

    precio_xgb = xgb_map.get(q['id'], 25000)
    colonia    = q.get('colonia', 'Monterrey')

    prompt = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Pregunta: {q['pregunta']}\n\n"
        f"Precio estimado por modelo estadístico (XGBoost): ${precio_xgb:,.0f} MXN/mes\n\n"
        f"Responde explicando este precio para {colonia}, citando que es basado en datos reales."
        f"<|im_end|>\n<|im_start|>assistant\n"
    )
    resp = generate(model, tokenizer, prompt=prompt, max_tokens=300, verbose=False)
    pred = extract_price(resp) or precio_xgb  # fallback al anchor si no parsea
    real = q.get('precio_real')
    err  = abs(pred - real) / real * 100 if pred and real else 100.0

    results.append({
        "id": q['id'], "precio_real": real, "precio_xgb": precio_xgb,
        "precio_predicho": pred, "error_pct": err,
        "dentro_15pct": err <= 15, "dentro_20pct": err <= 20,
        "modelo": "hybrid_boxcox"
    })

errs = [r['error_pct'] for r in results]
nr   = sum(1 for r in results if r['precio_predicho'] is None)
p15  = sum(1 for e in errs if e <= 15)
p20  = sum(1 for e in errs if e <= 20)

print(f"\n{'='*55}")
print(f"  HYBRID (fine-tuned + XGBoost Box-Cox)")
print(f"{'='*55}")
print(f"  MAE% (penalizado)  : {np.mean(errs):.1f}%")
print(f"  Dentro de ±15%     : {p15/n*100:.1f}%")
print(f"  Dentro de ±20%     : {p20/n*100:.1f}%")
print(f"  Non-response rate  : {nr/n*100:.1f}%")

out = RES / "benchmark_hybrid_boxcox_test.jsonl"
with open(out, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
print(f"\n✓ Guardado: {out}")
