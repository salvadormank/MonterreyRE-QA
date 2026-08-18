"""
statistical_tests_multimodel.py — Wilcoxon signed-rank tests for multi-model benchmark
Compares zero-shot vs few-shot+XGBoost for each model (local + commercial).
Also tests our best system vs each commercial model.
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats

BASE = Path(__file__).resolve().parent
RES  = BASE / "results"

def load_errors(path):
    if not Path(path).exists():
        return None
    rows = [json.loads(l) for l in open(path)]
    return [r["error_pct"] for r in rows]

results_out = {"zeroshot_vs_fewshot_xgb": {}, "best_local_vs_commercial": {}}

def wilcoxon_test(errs_a, errs_b, label_a, label_b):
    if errs_a is None or errs_b is None:
        print(f"  ⚠ Falta archivo para {label_a} o {label_b}")
        return None
    stat, p = stats.wilcoxon(errs_a, errs_b, alternative='two-sided')
    direction = "mejor" if np.mean(errs_a) < np.mean(errs_b) else "peor"
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {label_a} vs {label_b}")
    print(f"    MAE%: {np.mean(errs_a):.1f}% vs {np.mean(errs_b):.1f}% ({direction})")
    print(f"    W={stat:.1f}, p={p:.4f} {sig}")
    return {
        "mae_a": float(np.mean(errs_a)), "mae_b": float(np.mean(errs_b)),
        "W": float(stat), "p": float(p), "sig": sig,
    }

print("=" * 65)
print("  WILCOXON SIGNED-RANK TESTS — MULTI-MODEL BENCHMARK")
print("=" * 65)

# ── 1. Zero-shot vs Few-shot+XGBoost por modelo ───────────────────────────────
print("\n[1] Zero-shot → Few-shot+XGBoost (per model)")
print("-" * 65)

configs = [
    # (tag, label)
    ("llama31_8b",   "Llama 3.1 8B"),
    ("mistral_7b",   "Mistral 7B"),
    ("salamandra_7b","Salamandra 7B"),
    ("gpt4o_mini",   "GPT-4o-mini"),
    ("claude_haiku", "Claude Haiku 4.5"),
    ("claude_sonnet","Claude Sonnet 4.6"),
]

for tag, label in configs:
    zs  = load_errors(RES / f"benchmark_{tag}_zeroshot_test.jsonl")
    xgb = load_errors(RES / f"benchmark_{tag}_fewshot_xgb_test.jsonl")
    if zs and xgb:
        print(f"\n  {label}:")
        r = wilcoxon_test(xgb, zs, "few-shot+XGB", "zero-shot")
        if r:
            results_out["zeroshot_vs_fewshot_xgb"][tag] = r

# ── 2. Nuestro mejor sistema vs cada modelo comercial ─────────────────────────
print("\n\n[2] Best local system vs commercial APIs (few-shot+XGBoost)")
print("-" * 65)

# Best local = Qwen few-shot+XGBoost (from existing results)
best_local = load_errors(RES / "benchmark_fewshot_xgb_test.jsonl")

commercial = [
    ("gpt4o_mini",    "GPT-4o-mini"),
    ("claude_haiku",  "Claude Haiku 4.5"),
    ("claude_sonnet", "Claude Sonnet 4.6"),
]

if best_local:
    print(f"\n  Best local (Qwen few-shot+XGBoost): MAE%={np.mean(best_local):.1f}%")
    for tag, label in commercial:
        errs = load_errors(RES / f"benchmark_{tag}_fewshot_xgb_test.jsonl")
        if errs:
            print(f"\n  Best local vs {label}:")
            r = wilcoxon_test(best_local, errs, "Local (Qwen)", label)
            if r:
                results_out["best_local_vs_commercial"][tag] = r

# ── 3. Resumen tabla p-values ─────────────────────────────────────────────────
print("\n\n[3] Tabla resumen p-values (zero-shot vs few-shot+XGB)")
print("-" * 65)
print(f"  {'Model':<25} {'MAE% ZS':>8} {'MAE% FS+XGB':>12} {'p-value':>9} {'sig':>5}")
print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*9} {'-'*5}")

for tag, label in configs:
    zs  = load_errors(RES / f"benchmark_{tag}_zeroshot_test.jsonl")
    xgb = load_errors(RES / f"benchmark_{tag}_fewshot_xgb_test.jsonl")
    if zs and xgb:
        _, p = stats.wilcoxon(zs, xgb, alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {label:<25} {np.mean(zs):>8.1f}% {np.mean(xgb):>12.1f}% {p:>9.4f} {sig:>5}")
    else:
        print(f"  {label:<25} {'--':>8} {'--':>12} {'pending':>9} {'':>5}")

print("\n  Significance: *** p<0.001  ** p<0.01  * p<0.05  n.s. p≥0.05")
print("=" * 65)

json.dump(results_out, open(RES / "statistical_tests_multimodel.json", "w"), indent=2)
print(f"\n✓ Guardado: results/statistical_tests_multimodel.json")
