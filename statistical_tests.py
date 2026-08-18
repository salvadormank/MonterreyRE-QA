"""
statistical_tests.py — Tests estadísticos para el paper
- McNemar test para métricas binarias (±15%, ±20%)
- Bootstrap CI para MAE%
Comparaciones pareadas sobre los 109 test questions
"""
import json, numpy as np
from pathlib import Path
from scipy.stats import chi2
from itertools import combinations

RES = Path("./results")

base = [json.loads(l) for l in open(RES / "benchmark_base_test.jsonl")]
ft   = [json.loads(l) for l in open(RES / "benchmark_finetuned_test.jsonl")]
hyb  = [json.loads(l) for l in open(RES / "benchmark_hybrid_test.jsonl")]

n = 109

def get_data(records, model):
    out = []
    for r in records:
        real = r.get("precio_real")
        if model == "hybrid":
            pred = r.get("precio_xgb")
        else:
            pred = r.get("precio_predicho")
        if real and pred:
            err = abs(pred - real) / real * 100
        else:
            err = 100.0  # penalización
        out.append(err)
    return np.array(out)

base_errs = get_data(base, "base")
ft_errs   = get_data(ft,   "ft")
hyb_errs  = get_data(hyb,  "hybrid")

# ── McNemar test ──────────────────────────────────────────────────────────────
def mcnemar(a_correct, b_correct):
    """
    a_correct, b_correct: arrays booleanos pareados
    H0: los dos modelos tienen el mismo error rate
    Retorna p-value (con corrección de continuidad)
    """
    n01 = np.sum(~a_correct & b_correct)   # A falla, B acierta
    n10 = np.sum(a_correct & ~b_correct)   # A acierta, B falla
    if n01 + n10 == 0:
        return 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p = 1 - chi2.cdf(stat, df=1)
    return p, n01, n10

# ── Bootstrap CI para MAE% ───────────────────────────────────────────────────
def bootstrap_ci(errors, n_boot=10000, alpha=0.05):
    means = [np.mean(np.random.choice(errors, len(errors))) for _ in range(n_boot)]
    lo = np.percentile(means, alpha/2*100)
    hi = np.percentile(means, (1-alpha/2)*100)
    return np.mean(errors), lo, hi

np.random.seed(42)

print("="*65)
print("  BOOTSTRAP 95% CI — MAE% (10,000 iterations)")
print("="*65)
models = [("Base",       base_errs),
          ("Fine-tuned", ft_errs),
          ("Hybrid",     hyb_errs)]

ci_results = {}
for name, errs in models:
    mean, lo, hi = bootstrap_ci(errs)
    ci_results[name] = (mean, lo, hi)
    print(f"  {name:<20} MAE% = {mean:.1f}%  [95% CI: {lo:.1f}% – {hi:.1f}%]")

print("\n" + "="*65)
print("  MCNEMAR TEST — Within ±15% (pareado, n=109)")
print("="*65)
for threshold, label in [(15, "±15%"), (20, "±20%")]:
    print(f"\n  Threshold: {label}")
    print(f"  {'Comparison':<35} {'p-value':>10}  {'n01':>5}  {'n10':>5}  {'sig':>5}")
    print(f"  {'─'*60}")
    pairs = [("Base vs Fine-tuned", base_errs, ft_errs),
             ("Base vs Hybrid",     base_errs, hyb_errs),
             ("Fine-tuned vs Hybrid", ft_errs, hyb_errs)]
    for cname, ea, eb in pairs:
        a_ok = ea <= threshold
        b_ok = eb <= threshold
        result = mcnemar(a_ok, b_ok)
        if isinstance(result, tuple):
            p, n01, n10 = result
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {cname:<35} {p:>10.4f}  {n01:>5}  {n10:>5}  {sig:>5}")
        else:
            print(f"  {cname:<35} {'1.0000':>10}  {'—':>5}  {'—':>5}  {'n.s.':>5}")

print("\n" + "="*65)
print("  OVERLAP DE INTERVALOS DE CONFIANZA (MAE%)")
print("="*65)
for (n1, (m1,l1,h1)), (n2, (m2,l2,h2)) in combinations(ci_results.items(), 2):
    overlap = not (h1 < l2 or h2 < l1)
    print(f"  {n1} vs {n2}: {'CIs OVERLAP (no significativo)' if overlap else 'CIs NO overlap (significativo)'}")

# Guardar para el paper
output = {
    "bootstrap_ci": {name: {"mean": float(m), "ci_lo": float(l), "ci_hi": float(h)}
                     for name, (m,l,h) in ci_results.items()},
    "mcnemar_15pct": {},
    "mcnemar_20pct": {},
}
for threshold, key in [(15, "mcnemar_15pct"), (20, "mcnemar_20pct")]:
    pairs = [("base_vs_ft",      base_errs, ft_errs),
             ("base_vs_hybrid",  base_errs, hyb_errs),
             ("ft_vs_hybrid",    ft_errs,   hyb_errs)]
    for cname, ea, eb in pairs:
        result = mcnemar(ea <= threshold, eb <= threshold)
        if isinstance(result, tuple):
            p, n01, n10 = result
            output[key][cname] = {"p": float(p), "n01": int(n01), "n10": int(n10)}

json.dump(output, open(RES / "statistical_tests.json", "w"), indent=2)
print(f"\n✓ Guardado: results/statistical_tests.json")
