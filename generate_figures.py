"""
generate_figures.py — Genera las 4 figuras del paper
Figura 1: Curva de entrenamiento LoRA
Figura 2: Comparación de modelos (barras)
Figura 3: Error por segmento de precio (sesgo de distribución)
Figura 4: Predicho vs Real (scatter XGBoost)
"""

import re, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE = Path(__file__).resolve().parent
RES  = BASE / "results"
FIG  = BASE / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "legend.fontsize": 10,
    "figure.dpi":      150,
})

COLORS = {
    "base":      "#7bafd4",
    "finetuned": "#f4a261",
    "hybrid":    "#2a9d8f",
    "train":     "#e76f51",
    "val":       "#264653",
}


# ── Figura 1: Curva de entrenamiento ─────────────────────────────────────────

print("Figura 1: Curva de entrenamiento...")

log_path = RES / "finetune_log.txt"
train_iters, train_losses = [], []
val_iters,   val_losses   = [], []

for line in open(log_path):
    m = re.search(r'Iter (\d+): Train loss ([\d.]+)', line)
    if m:
        train_iters.append(int(m.group(1)))
        train_losses.append(float(m.group(2)))
    m = re.search(r'Iter (\d+): Val loss ([\d.]+)', line)
    if m:
        val_iters.append(int(m.group(1)))
        val_losses.append(float(m.group(2)))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(train_iters, train_losses, color=COLORS["train"], linewidth=1.5,
        alpha=0.7, label="Train loss")
ax.plot(val_iters, val_losses, color=COLORS["val"], linewidth=2.5,
        marker="o", markersize=5, label="Validation loss")

ax.set_xlabel("Iteration")
ax.set_ylabel("Cross-entropy loss")
ax.set_title("Figure 1: LoRA Fine-tuning Convergence (Qwen2.5-7B, 872 examples)")
ax.legend()
ax.set_ylim(bottom=0)
ax.annotate(f"Val loss: 2.249 → 0.028\n(−98.7%)",
            xy=(600, 0.028), xytext=(400, 0.4),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=9, color="gray")
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(FIG / "fig1_training_curve.pdf", bbox_inches="tight")
plt.savefig(FIG / "fig1_training_curve.png", bbox_inches="tight")
plt.close()
print("  ✓ fig1_training_curve.pdf")


# ── Figura 2: Comparación de 8 configuraciones ───────────────────────────────

print("Figura 2: Comparación de 8 configuraciones...")

configs = [
    ("Base LLM\n(zero-shot)",        54.9, 21.1, 25.7, 21.1, "base"),
    ("Base + prompt",                 40.1, 30.3, 40.4,  0.9, "base"),
    ("LoRA fine-tuned",               42.5, 31.2, 43.1,  0.0, "finetune"),
    ("Hybrid: fine-tuned\n+XGBoost",  33.2, 44.0, 54.1,  0.0, "hybrid"),
    ("RAG solo\n(top-5)",             29.5, 47.7, 58.7,  0.0, "context"),
    ("Few-shot solo\n(3 similar)",    29.6, 50.5, 59.6,  0.0, "context"),
    ("RAG + XGBoost",                 29.8, 49.5, 57.8,  0.0, "context"),
    ("Few-shot\n+XGBoost ★",          28.7, 55.0, 58.7,  0.0, "context"),
]

CAT_COLORS = {
    "base":     "#adb5bd",
    "finetune": "#f4a261",
    "hybrid":   "#2a9d8f",
    "context":  "#457b9d",
}

labels   = [c[0] for c in configs]
mae_vals = [c[1] for c in configs]
p15_vals = [c[2] for c in configs]
p20_vals = [c[3] for c in configs]
nonr     = [c[4] for c in configs]
bar_cols = [CAT_COLORS[c[5]] for c in configs]

y = np.arange(len(configs))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: MAE% (barras horizontales) + non-response como overlay
ax = axes[0]
bars = ax.barh(y, mae_vals, 0.6, color=bar_cols, alpha=0.85)
for i, (bar, nr) in enumerate(zip(bars, nonr)):
    ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height()/2,
            f"{bar.get_width():.1f}%", va="center", fontsize=9)
    if nr > 0:
        ax.text(bar.get_width() - 1, bar.get_y() + bar.get_height()/2,
                f"NR:{nr:.0f}%", va="center", ha="right", fontsize=7.5,
                color="white", fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel("MAE% (penalized, lower is better)")
ax.set_title("(a) Mean Absolute Error %")
ax.set_xlim(0, 68)
ax.axvline(28.7, color="#457b9d", linestyle="--", linewidth=1, alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.3)

# Legend
from matplotlib.patches import Patch
legend_els = [Patch(color=CAT_COLORS["base"],    label="Zero-shot / Prompt"),
              Patch(color=CAT_COLORS["finetune"], label="LoRA fine-tuning"),
              Patch(color=CAT_COLORS["hybrid"],   label="Hybrid (fine-tuned+XGB)"),
              Patch(color=CAT_COLORS["context"],  label="Context-based (no training)")]
ax.legend(handles=legend_els, fontsize=8, loc="lower right")

# Right: within ±15% y ±20%
ax = axes[1]
w = 0.28
bars3 = ax.barh(y - w/2, p15_vals, w, label="Within ±15%",
                color=[c + "bb" for c in bar_cols], alpha=0.9)
bars4 = ax.barh(y + w/2, p20_vals, w, label="Within ±20%",
                color=bar_cols, alpha=0.65)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel("Percentage of predictions (%)")
ax.set_title("(b) Predictions Within Error Margin")
ax.set_xlim(0, 75)
for bar in bars3:
    ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height()/2,
            f"{bar.get_width():.1f}%", va="center", fontsize=8)
for bar in bars4:
    ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height()/2,
            f"{bar.get_width():.1f}%", va="center", fontsize=8)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.3)

fig.suptitle("Figure 2: LLM Adaptation Strategy Comparison (109 held-out test questions)",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(FIG / "fig2_model_comparison.pdf", bbox_inches="tight")
plt.savefig(FIG / "fig2_model_comparison.png", bbox_inches="tight")
plt.close()
print("  ✓ fig2_model_comparison.pdf")


# ── Figura 3: Error por segmento de precio (sesgo de distribución) ───────────

print("Figura 3: Sesgo de distribución del fine-tuneado...")

ft_results = [json.loads(l) for l in
              open(RES / "benchmark_finetuned_test.jsonl")]
priced = [r for r in ft_results
          if r.get("precio_real") and r.get("precio_predicho") and r.get("error_pct")]

# Segmentar por precio real
bins   = [0, 15000, 20000, 25000, 30000, 40000, 999999]
labels = ["<$15k", "$15k–20k", "$20k–25k", "$25k–30k", "$30k–40k", ">$40k"]
df_ft  = pd.DataFrame(priced)
df_ft["segment"] = pd.cut(df_ft["precio_real"], bins=bins, labels=labels)

seg_stats = df_ft.groupby("segment", observed=True).agg(
    mae_pct=("error_pct", "mean"),
    n=("error_pct", "count"),
    med_pred=("precio_predicho", "median"),
    med_real=("precio_real", "median"),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: MAE% por segmento
ax = axes[0]
bar_colors = [COLORS["finetuned"] if v <= 30 else "#d62728"
              for v in seg_stats["mae_pct"]]
bars = ax.bar(seg_stats["segment"].astype(str), seg_stats["mae_pct"],
              color=bar_colors, alpha=0.85, edgecolor="white")
ax.axhline(30, color="gray", linestyle="--", linewidth=1, label="30% threshold")
ax.set_xlabel("Actual price segment (MXN/month)")
ax.set_ylabel("Mean Absolute Error %")
ax.set_title("(a) Prediction Error by Price Segment")
ax.legend()
for bar, n in zip(bars, seg_stats["n"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"n={n}", ha="center", va="bottom", fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)

# Right: precio predicho vs real por segmento
ax = axes[1]
ax.bar(seg_stats["segment"].astype(str),
       seg_stats["med_real"] / 1000, alpha=0.6,
       label="Actual median price", color=COLORS["val"])
ax.bar(seg_stats["segment"].astype(str),
       seg_stats["med_pred"] / 1000, alpha=0.75, width=0.4,
       label="Predicted median price", color=COLORS["finetuned"])
ax.axhline(23.5, color="gray", linestyle=":", linewidth=1.5,
           label="Training median (~$23.5k)")
ax.set_xlabel("Actual price segment (MXN/month)")
ax.set_ylabel("Price (thousands MXN/month)")
ax.set_title("(b) Median Bias Toward Training Distribution")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Figure 3: Distributional Bias of Fine-tuned Model (109 test questions)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(FIG / "fig3_bias_analysis.pdf", bbox_inches="tight")
plt.savefig(FIG / "fig3_bias_analysis.png", bbox_inches="tight")
plt.close()
print("  ✓ fig3_bias_analysis.pdf")


# ── Figura 4: Predicho vs Real — XGBoost ─────────────────────────────────────

print("Figura 4: Scatter predicho vs real (XGBoost)...")

scored = pd.read_csv(RES / "propiedades_scored.csv")
# Filtrar outliers para el plot
scored = scored[(scored["price"] > 4500) & (scored["price"] < 65000)]

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(scored["price"] / 1000, scored["precio_estimado"] / 1000,
           alpha=0.35, s=15, color=COLORS["hybrid"], edgecolors="none")

lim = max(scored["price"].max(), scored["precio_estimado"].max()) / 1000 + 2
ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5, label="Perfect prediction")
ax.fill_between([0, lim], [0*0.85, lim*0.85], [0*1.15, lim*1.15],
                alpha=0.08, color="gray", label="±15% band")

ax.set_xlabel("Actual price (thousands MXN/month)")
ax.set_ylabel("Predicted price (thousands MXN/month)")
ax.set_title("Figure 4: XGBoost Predicted vs. Actual Prices\n(R²=0.70, MAE=±$3,610 MXN/month)")
ax.legend(fontsize=9)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_aspect("equal")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(FIG / "fig4_xgb_scatter.pdf", bbox_inches="tight")
plt.savefig(FIG / "fig4_xgb_scatter.png", bbox_inches="tight")
plt.close()
print("  ✓ fig4_xgb_scatter.pdf")


# ── Figura 5: Comparación de 7 modelos de regresión ─────────────────────────

print("Figura 5: Comparación de modelos de regresión...")

comp = json.load(open(RES / "model_comparison.json"))

model_names = list(comp.keys())
r2_vals  = [comp[m]["r2"]     for m in model_names]
cv_vals  = [comp[m]["cv_r2"]  for m in model_names]
cv_stds  = [comp[m]["cv_std"] for m in model_names]
mae_vals = [comp[m]["mae"]    for m in model_names]

# Ordenar por R² test
order = sorted(range(len(model_names)), key=lambda i: r2_vals[i])
model_names = [model_names[i] for i in order]
r2_vals  = [r2_vals[i]  for i in order]
cv_vals  = [cv_vals[i]  for i in order]
cv_stds  = [cv_stds[i]  for i in order]
mae_vals = [mae_vals[i] for i in order]

bar_colors = [COLORS["hybrid"] if "XGBoost" in n else
              COLORS["finetuned"] if n in ("LightGBM","CatBoost","Random Forest") else
              "#aaaaaa" for n in model_names]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: R² test + CV R²
ax = axes[0]
y_pos = np.arange(len(model_names))
bars = ax.barh(y_pos, r2_vals, 0.5, color=bar_colors, alpha=0.85, label="R² test")
ax.errorbar(cv_vals, y_pos, xerr=cv_stds, fmt="D", color="#264653",
            markersize=5, linewidth=1.5, label="CV R² ± std", zorder=5)
ax.set_yticks(y_pos)
ax.set_yticklabels(model_names)
ax.set_xlabel("R²")
ax.set_title("(a) R² Test vs Cross-Validation R²")
ax.set_xlim(0, 1.0)
ax.axvline(0.7, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax.legend(fontsize=9)
for bar, val in zip(bars, r2_vals):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.3)

# Right: MAE
ax = axes[1]
bars2 = ax.barh(y_pos, [m/1000 for m in mae_vals], 0.5, color=bar_colors, alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(model_names)
ax.set_xlabel("MAE (thousands MXN/month)")
ax.set_title("(b) Mean Absolute Error")
for bar, val in zip(bars2, mae_vals):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f"${val:,.0f}", va="center", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.3)

fig.suptitle("Figure 5: Regression Model Comparison (same features and split, n=1,090)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(FIG / "fig5_model_comparison_regression.pdf", bbox_inches="tight")
plt.savefig(FIG / "fig5_model_comparison_regression.png", bbox_inches="tight")
plt.close()
print("  ✓ fig5_model_comparison_regression.pdf")

print(f"\n✓ 5 figuras guardadas en {FIG}")
print("  Archivos: fig1_training_curve, fig2_model_comparison,")
print("            fig3_bias_analysis, fig4_xgb_scatter,")
print("            fig5_model_comparison_regression")
print("  Formatos: .pdf (para LaTeX) + .png (para preview)")
