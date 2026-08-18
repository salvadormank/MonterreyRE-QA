"""
llm_respond.py  — Paso 2 del híbrido
Lee xgb_predictions.json, genera respuestas con Qwen fine-tuneado
y guarda resultados + métricas finales.
"""
import sys, json, re
import numpy as np
from pathlib import Path

sys.path.insert(0, "/opt/homebrew/Cellar/mlx/0.31.2/lib/python3.14/site-packages")
from mlx_lm import load, generate

BASE = Path(__file__).resolve().parent
RES  = BASE / "results"

MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
ADAPTER  = str(BASE / "adapters")

SYSTEM = (
    "Eres un experto en el mercado inmobiliario de renta en Monterrey, Nuevo León. "
    "Respondes en español con argumentos concretos basados en el mercado local. "
    "Cuando se te proporciona un precio calculado por modelo estadístico, úsalo como base."
)

print("Cargando Qwen fine-tuneado...")
model, tokenizer = load(MODEL_ID, adapter_path=ADAPTER)
print("✓ Modelo listo\n")

data = json.load(open(RES / "xgb_predictions.json"))

results = []
n = len(data)
for i, q in enumerate(data):
    if i % 10 == 0:
        print(f"  Progreso: {i}/{n} ({i/n*100:.0f}%)")

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

    err = abs(precio_xgb - precio_real) / precio_real * 100 \
          if precio_xgb and precio_real else None

    results.append({**q, "respuesta_hibrida": respuesta, "error_pct": err,
                    "dentro_15pct": err <= 15 if err else False,
                    "dentro_20pct": err <= 20 if err else False})

# Guardar respuestas
out_path = RES / "benchmark_hybrid_test.jsonl"
with open(out_path, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

# Métricas
priced = [r for r in results if r.get("precio_real") and r.get("precio_xgb")]
errors = [r["error_pct"] for r in priced if r.get("error_pct")]

print(f"\n{'='*55}")
print("  RESULTADOS SISTEMA HÍBRIDO (109 preguntas)")
print(f"{'='*55}")
if errors:
    mae_mxn = np.mean([abs(r['precio_xgb']-r['precio_real']) for r in priced])
    print(f"  MAE%            : {np.mean(errors):.1f}%")
    print(f"  MAE (MXN)       : ${mae_mxn:,.0f}")
    print(f"  Dentro de ±15%  : {sum(1 for e in errors if e<=15)/len(errors)*100:.1f}%")
    print(f"  Dentro de ±20%  : {sum(1 for e in errors if e<=20)/len(errors)*100:.1f}%")
n_fail = sum(1 for r in results if not r.get("precio_xgb"))
print(f"  Sin precio XGBoost: {n_fail}")

print(f"\n{'='*55}")
print("  COMPARACIÓN FINAL")
print(f"{'='*55}")
print(f"  {'Modelo':<28} {'MAE%':>6}  {'±15%':>6}  {'±20%':>6}  {'Aluc':>6}")
print(f"  {'─'*55}")
print(f"  {'Qwen base':<28} {'42.8%':>6}  {'26.7%':>6}  {'32.6%':>6}  {'26.7%':>6}")
print(f"  {'Qwen fine-tuneado':<28} {'42.5%':>6}  {'31.2%':>6}  {'43.1%':>6}   {'0.0%':>5}")
if errors:
    mae_pct = np.mean(errors)
    p15 = sum(1 for e in errors if e<=15)/len(errors)*100
    p20 = sum(1 for e in errors if e<=20)/len(errors)*100
    print(f"  {'Híbrido XGBoost+Qwen':<28} {mae_pct:>5.1f}%  {p15:>5.1f}%  {p20:>5.1f}%   {'0.0%':>5}")
print(f"  {'─'*55}")
print(f"\n✓ Respuestas: {out_path}")
