"""
fix_metrics.py — Recalcula métricas correctamente:
1. MAE% sobre intersección (preguntas respondidas por TODOS los modelos)
2. Penalización MAE%=100 para no-respuestas (comparación completa)
3. Reporta ambos para el paper
"""
import json, numpy as np
from pathlib import Path

RES = Path("./results")

base = [json.loads(l) for l in open(RES / "benchmark_base_test.jsonl")]
ft   = [json.loads(l) for l in open(RES / "benchmark_finetuned_test.jsonl")]
hyb  = [json.loads(l) for l in open(RES / "benchmark_hybrid_test.jsonl")]

def get_pred(r, model):
    if model == "hybrid":
        return r.get("precio_xgb")
    return r.get("precio_predicho")

def has_response(r, model):
    p = get_pred(r, model)
    return p is not None and r.get("precio_real") is not None

# Índices con respuesta por modelo
base_idx = {i for i,r in enumerate(base) if has_response(r, "base")}
ft_idx   = {i for i,r in enumerate(ft)   if has_response(r, "ft")}
hyb_idx  = {i for i,r in enumerate(hyb)  if has_response(r, "hybrid")}

intersection = base_idx & ft_idx & hyb_idx
n_total = len(base)

print("="*65)
print("  RESPONSE RATES")
print("="*65)
print(f"  Base:       {len(base_idx)}/{n_total} ({len(base_idx)/n_total*100:.1f}%)  non-response: {(n_total-len(base_idx))/n_total*100:.1f}%")
print(f"  Fine-tuned: {len(ft_idx)}/{n_total} ({len(ft_idx)/n_total*100:.1f}%)  non-response: {(n_total-len(ft_idx))/n_total*100:.1f}%")
print(f"  Hybrid:     {len(hyb_idx)}/{n_total} ({len(hyb_idx)/n_total*100:.1f}%)  non-response: {(n_total-len(hyb_idx))/n_total*100:.1f}%")
print(f"  Intersection (all responded): {len(intersection)}/{n_total}")

def calc_metrics(records, indices, model, label):
    errs, p15, p20 = [], [], []
    for i in indices:
        r = records[i]
        pred = get_pred(r, model)
        real = r["precio_real"]
        err = abs(pred - real) / real * 100
        errs.append(err)
        p15.append(1 if err <= 15 else 0)
        p20.append(1 if err <= 20 else 0)
    return {
        "mae_pct": round(np.mean(errs), 1),
        "within_15": round(np.mean(p15)*100, 1),
        "within_20": round(np.mean(p20)*100, 1),
        "n": len(errs)
    }

def calc_full_penalized(records, model, penalty=100.0):
    errs, p15, p20 = [], [], []
    for i, r in enumerate(records):
        real = r.get("precio_real")
        if real is None:
            continue
        pred = get_pred(r, model)
        if pred is None:
            err = penalty
        else:
            err = abs(pred - real) / real * 100
        errs.append(err)
        p15.append(1 if err <= 15 else 0)
        p20.append(1 if err <= 20 else 0)
    return {
        "mae_pct": round(np.mean(errs), 1),
        "within_15": round(np.mean(p15)*100, 1),
        "within_20": round(np.mean(p20)*100, 1),
        "n": len(errs)
    }

print("\n" + "="*65)
print("  MÉTRICAS SOBRE INTERSECCIÓN (preguntas respondidas por todos)")
print("="*65)
print(f"  {'Model':<20} {'MAE%':>6}  {'±15%':>6}  {'±20%':>6}  {'n':>4}")
print(f"  {'─'*50}")
for label, recs, model in [("Base",       base, "base"),
                             ("Fine-tuned", ft,   "ft"),
                             ("Hybrid",     hyb,  "hybrid")]:
    m = calc_metrics(recs, list(intersection), model, label)
    print(f"  {label:<20} {m['mae_pct']:>6.1f}%  {m['within_15']:>5.1f}%  {m['within_20']:>5.1f}%  {m['n']:>4}")

print("\n" + "="*65)
print("  MÉTRICAS CON PENALIZACIÓN (MAE%=100 para no-respuestas)")
print("  (comparación justa sobre los 109 completos)")
print("="*65)
print(f"  {'Model':<20} {'MAE%':>6}  {'±15%':>6}  {'±20%':>6}  {'n':>4}")
print(f"  {'─'*50}")
for label, recs, model in [("Base",       base, "base"),
                             ("Fine-tuned", ft,   "ft"),
                             ("Hybrid",     hyb,  "hybrid")]:
    m = calc_full_penalized(recs, model)
    print(f"  {label:<20} {m['mae_pct']:>6.1f}%  {m['within_15']:>5.1f}%  {m['within_20']:>5.1f}%  {m['n']:>4}")

print("\n" + "="*65)
print("  MÉTRICAS ORIGINALES (solo sobre respondidas — sesgado)")
print("="*65)
print(f"  {'Model':<20} {'MAE%':>6}  {'±15%':>6}  {'±20%':>6}  {'n':>4}")
print(f"  {'─'*50}")
for label, recs, model, idx in [("Base",       base, "base", base_idx),
                                  ("Fine-tuned", ft,   "ft",   ft_idx),
                                  ("Hybrid",     hyb,  "hybrid",hyb_idx)]:
    m = calc_metrics(recs, list(idx), model, label)
    print(f"  {label:<20} {m['mae_pct']:>6.1f}%  {m['within_15']:>5.1f}%  {m['within_20']:>5.1f}%  {m['n']:>4}")

# Guardar
results = {}
for label, recs, model, idx in [("base", base, "base", base_idx),
                                  ("finetuned", ft, "ft", ft_idx),
                                  ("hybrid", hyb, "hybrid", hyb_idx)]:
    results[label] = {
        "intersection": calc_metrics(recs, list(intersection), model, label),
        "penalized":    calc_full_penalized(recs, model),
        "original":     calc_metrics(recs, list(idx), model, label),
        "response_rate": round(len(idx)/n_total*100, 1),
        "non_response_rate": round((n_total-len(idx))/n_total*100, 1),
    }
json.dump(results, open(RES / "metrics_corrected.json", "w"), indent=2)
print(f"\n✓ Guardado: results/metrics_corrected.json")
