"""
ablation_base_hybrid.py — Ablation crítico:
Compara base LLM + ancla XGBoost  vs  fine-tuned LLM + ancla XGBoost

La pregunta de Opus: ¿el fine-tuning aporta algo más allá de eliminar omisiones?
Numéricamente ambos híbridos son iguales (usan precio_xgb).
Este script evalúa si el base LLM acepta el ancla o la ignora/contradice.
"""
import sys, json, re
import numpy as np
from pathlib import Path

sys.path.insert(0, "/opt/homebrew/Cellar/mlx/0.31.2/lib/python3.14/site-packages")
from mlx_lm import load, generate

BASE = Path(__file__).resolve().parent
RES  = BASE / "results"

MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"

SYSTEM = (
    "Eres un experto en el mercado inmobiliario de renta en Monterrey, Nuevo León. "
    "Respondes en español con argumentos concretos basados en el mercado local. "
    "Cuando se te proporciona un precio calculado por modelo estadístico, úsalo como base."
)

def extract_price(text):
    patterns = [
        r'\$\s*([\d,]+(?:\.\d+)?)\s*(?:MXN|pesos|al mes|mensual|/mes)',
        r'([\d,]+(?:\.\d+)?)\s*(?:MXN|pesos)\s*(?:al mes|mensual|/mes)',
        r'precio[^:]*:\s*\$?\s*([\d,]+)',
        r'renta[^:]*:\s*\$?\s*([\d,]+)',
        r'\$\s*([\d,]{4,})',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            val = float(m.group(1).replace(',', ''))
            if 3000 <= val <= 150000:
                return val
    return None

print("Cargando Qwen BASE (sin adapter)...")
model, tokenizer = load(MODEL_ID)   # sin adapter_path
print("✓ Modelo base listo\n")

data = json.load(open(RES / "xgb_predictions.json"))

results = []
n = len(data)
for i, q in enumerate(data):
    if i % 10 == 0:
        print(f"  {i}/{n} ({i/n*100:.0f}%)")

    precio_xgb  = q.get("precio_xgb")
    precio_real = q.get("precio_real")
    colonia     = q.get("params", {}).get("colonia") or "Monterrey"

    if precio_xgb:
        context = (
            f"Pregunta: {q['pregunta']}\n\n"
            f"Precio estimado por modelo estadístico (XGBoost): ${precio_xgb:,.0f} MXN/mes\n\n"
            f"Responde explicando este precio para {colonia}, citando que es basado en datos reales."
        )
    else:
        context = q["pregunta"]

    prompt = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{context}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    respuesta = generate(model, tokenizer, prompt=prompt, max_tokens=300, verbose=False)

    # Verificar si el LLM acepta el ancla o genera su propio precio
    precio_llm = extract_price(respuesta)
    anchor_respected = None
    if precio_xgb and precio_llm:
        deviation = abs(precio_llm - precio_xgb) / precio_xgb * 100
        anchor_respected = deviation <= 20  # considera aceptado si desvia <20%

    err_xgb = abs(precio_xgb - precio_real) / precio_real * 100 \
              if precio_xgb and precio_real else None

    results.append({
        **q,
        "respuesta_base_hybrid": respuesta,
        "precio_llm_extraido": precio_llm,
        "anchor_respected": anchor_respected,
        "error_pct_xgb": err_xgb,
    })

# Guardar
out_path = RES / "benchmark_base_hybrid_test.jsonl"
with open(out_path, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

# Métricas
responded    = [r for r in results if r.get("precio_llm_extraido")]
with_anchor  = [r for r in results if r.get("anchor_respected") is not None]
anchor_ok    = [r for r in with_anchor if r["anchor_respected"]]

print(f"\n{'='*60}")
print("  BASE LLM + ANCLA XGBOOST — Resultados")
print(f"{'='*60}")
print(f"  Respuestas con precio parseable : {len(responded)}/{n} ({len(responded)/n*100:.1f}%)")
print(f"  Non-response rate               : {(n-len(responded))/n*100:.1f}%")
if with_anchor:
    print(f"  Ancla respetada (<20% desviación): {len(anchor_ok)}/{len(with_anchor)} ({len(anchor_ok)/len(with_anchor)*100:.1f}%)")

# Comparación final
print(f"\n{'='*60}")
print("  ABLATION: ¿Qué aporta el fine-tuning al híbrido?")
print(f"{'='*60}")
print(f"  {'Configuración':<35} {'Non-resp':>9}  {'Ancla OK':>9}")
print(f"  {'─'*56}")
print(f"  {'Base LLM + XGBoost anchor':<35} {(n-len(responded))/n*100:>8.1f}%  {len(anchor_ok)/len(with_anchor)*100 if with_anchor else 0:>8.1f}%")
print(f"  {'Fine-tuned LLM + XGBoost anchor':<35} {'0.0%':>9}  {'~100%':>9}  (por diseño)")
print(f"\n  Nota: MAE% es idéntico en ambos (ancla = precio_xgb)")
print(f"\n✓ Guardado: {out_path}")
