"""
train_xgboost.py
────────────────────────────────────────────────────────────────────────────
Modelo híbrido XGBoost para predicción de precio de renta en Monterrey.
Fuente: propiedades_enriquecido.xlsx (Websights, 1,102 registros)

Features usadas:
  Numéricas  : m2, recamaras, banos, estacionamientos, lat, lon,
               userViews, publishedSince, amenidades_count
  Categóricas: colonia (target-encoded), municipio
  Binarias   : Lujo, Amueblado, Nuevo + 8 amenidades de featuresPills

Salidas:
  results/xgb_model.json          → modelo guardado
  results/xgb_metrics.json        → métricas (MAE, RMSE, R², CV)
  figures/xgb_importance.png      → feature importance
  figures/xgb_real_vs_pred.png    → real vs predicho
  figures/xgb_residuals.png       → distribución de residuales
────────────────────────────────────────────────────────────────────────────
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import boxcox
from scipy.special import inv_boxcox

warnings.filterwarnings("ignore")

# ── Rutas ─────────────────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parent
INPUT  = Path(Path(__file__).resolve().parent.parent / "propiedades_enriquecido.xlsx")
RES    = BASE / "results"
FIGS   = BASE / "figures"
RES.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

# ── 1. Carga ───────────────────────────────────────────────────────────────
print("Cargando datos...")
df = pd.read_excel(INPUT)
print(f"  {len(df):,} registros, {df.shape[1]} columnas")

# ── 2. Filtro de precio ────────────────────────────────────────────────────
# Quitar outliers extremos (precio > p99)
p99 = df["price"].quantile(0.99)
df  = df[df["price"].between(4_500, p99)].copy()
print(f"  Tras filtro precio (<= ${p99:,.0f}): {len(df):,} registros")

# ── 3. Parsear lat / lon ───────────────────────────────────────────────────
def parse_latlon(s):
    try:
        lat, lon = str(s).split(",")
        return float(lat.strip()), float(lon.strip())
    except Exception:
        return np.nan, np.nan

df[["lat", "lon"]] = pd.DataFrame(
    df["latlon"].apply(parse_latlon).tolist(), index=df.index
)

# ── 4. Parsear colonia / municipio desde location ─────────────────────────
def parse_location(s):
    if not isinstance(s, str):
        return None, None
    parts = [p.strip() for p in s.split(",")]
    colonia   = parts[0] if len(parts) >= 1 else None
    municipio = parts[1] if len(parts) >= 2 else None
    return colonia, municipio

df[["colonia", "municipio"]] = pd.DataFrame(
    df["location"].apply(parse_location).tolist(), index=df.index
)

# ── 5. Amenidades desde featuresPills ─────────────────────────────────────
AMENIDAD_MAP = {
    "gimnasio":          "pill_gym",
    "alberca":           "pill_pool",
    "jardín":            "pill_garden",
    "jardín ":           "pill_garden",
    "circuito cerrado":  "pill_security",
    "elevador":          "pill_elevator",
    "terraza":           "pill_terrace",
    "rooftop":           "pill_rooftop",
    "área de juegos":    "pill_playground",
}
pill_cols = [c for c in df.columns if c.startswith("featuresPills")]
for feat_col in AMENIDAD_MAP.values():
    df[feat_col] = 0

for _, row in df.iterrows():
    pills = [str(row[c]).lower().strip() for c in pill_cols if pd.notna(row[c])]
    for keyword, feat_col in AMENIDAD_MAP.items():
        if any(keyword in p for p in pills):
            df.at[row.name, feat_col] = 1

df["amenidades_count"] = df[[c for c in AMENIDAD_MAP.values()]].sum(axis=1)

# ── 6. publishedSince → días numérico ─────────────────────────────────────
def parse_days(s):
    if pd.isna(s):
        return np.nan
    s = str(s).lower()
    if "hoy" in s or "today" in s:
        return 0
    import re
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else np.nan

df["days_listed"] = df["publishedSince"].apply(parse_days)
df["userViews"]   = pd.to_numeric(df["userViews"], errors="coerce")

# ── 7. Feature matrix ──────────────────────────────────────────────────────
NUM_FEATURES = ["m2", "recamaras", "banos", "estacionamientos",
                "lat", "lon", "userViews", "days_listed", "amenidades_count"]

BIN_FEATURES = ["Lujo", "Amueblado", "Nuevo"] + list(AMENIDAD_MAP.values())

CAT_FEATURES = ["colonia", "municipio"]

# Target transform: Box-Cox (mejora R² vs log1p; λ≈0.36 para precios de Monterrey)
bc_values, BC_LAMBDA = boxcox(df["price"].values)
df["bc_price"] = bc_values
global_mean = df["bc_price"].mean()
print(f"  Box-Cox lambda: {BC_LAMBDA:.4f}")

for cat in CAT_FEATURES:
    counts = df.groupby(cat)["bc_price"].agg(["mean", "count"])
    smooth_k = 10  # factor de suavizado bayesiano
    smoothed = (counts["mean"] * counts["count"] + global_mean * smooth_k) / \
               (counts["count"] + smooth_k)
    df[f"{cat}_enc"] = df[cat].map(smoothed).fillna(global_mean)

ENC_FEATURES = [f"{c}_enc" for c in CAT_FEATURES]

ALL_FEATURES = NUM_FEATURES + BIN_FEATURES + ENC_FEATURES

# Imputar nulos con mediana (solo columnas numéricas)
df_feat = df[ALL_FEATURES].copy().astype(float)
medians  = df_feat.median()
df_feat  = df_feat.fillna(medians)

X = df_feat.values
y = df["bc_price"].values

print(f"\nFeatures: {len(ALL_FEATURES)}")
print(f"Registros: {len(X)}")

# ── 8. XGBoost ────────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    print(f"  XGBoost {xgb.__version__}")
except ImportError:
    print("\n✗ XGBoost no instalado. Corre: pip install xgboost")
    raise

params = dict(
    n_estimators     = 500,
    max_depth        = 5,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    random_state     = 42,
    n_jobs           = -1,
)

model = xgb.XGBRegressor(**params)

# Cross-validation 5-fold
print("\nCross-validation 5-fold...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2  = cross_val_score(model, X, y, cv=kf, scoring="r2")
cv_mae = cross_val_score(model, X, y, cv=kf,
                         scoring="neg_mean_absolute_error")
print(f"  CV R²  : {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
print(f"  CV MAE : {inv_boxcox(-cv_mae.mean(), BC_LAMBDA):,.0f} ± {inv_boxcox(cv_mae.std(), BC_LAMBDA):,.0f} MXN")

# Train/test split 80-20
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_te, y_te)],
    verbose=False,
)

# ── 9. Métricas ───────────────────────────────────────────────────────────
y_pred_bc = model.predict(X_te)
y_pred    = inv_boxcox(y_pred_bc, BC_LAMBDA)
y_true    = inv_boxcox(y_te, BC_LAMBDA)

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)

print(f"\n{'='*50}")
print(f"  MAE  : ${mae:,.0f} MXN/mes")
print(f"  RMSE : ${rmse:,.0f} MXN/mes")
print(f"  R²   : {r2:.3f}")
print(f"  CV R²: {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
print(f"{'='*50}")

metrics = {
    "mae": round(mae, 2),
    "rmse": round(rmse, 2),
    "r2_test": round(r2, 4),
    "cv_r2_mean": round(float(cv_r2.mean()), 4),
    "cv_r2_std":  round(float(cv_r2.std()),  4),
    "cv_mae_mean": round(float(-cv_mae.mean()), 2),
    "n_train": len(X_tr),
    "n_test":  len(X_te),
    "features": ALL_FEATURES,
    "target_transform": "boxcox",
    "boxcox_lambda": round(float(BC_LAMBDA), 6),
}
(RES / "xgb_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

# ── 10. Score de sobrevaluación ───────────────────────────────────────────
# Predecimos sobre TODO el dataset para identificar outliers
X_all   = df_feat.values
y_all_p = inv_boxcox(model.predict(X_all), BC_LAMBDA)
df["precio_estimado"]    = y_all_p
df["diferencia_pct"]     = (df["price"] - df["precio_estimado"]) / df["precio_estimado"] * 100
df["sobrevaluado"]       = df["diferencia_pct"] > 20
df["subvaluado"]         = df["diferencia_pct"] < -20

sobre = df["sobrevaluado"].sum()
sub   = df["subvaluado"].sum()
print(f"\n  Sobrevaluadas (>20% sobre estimado): {sobre} ({sobre/len(df)*100:.1f}%)")
print(f"  Subvaluadas  (>20% bajo estimado):   {sub}  ({sub/len(df)*100:.1f}%)")

# Guardar CSV con scores
out_csv = RES / "propiedades_scored.csv"
df[["location", "price", "precio_estimado", "diferencia_pct",
    "sobrevaluado", "subvaluado", "m2", "recamaras", "banos", "Lujo",
    "Amueblado", "colonia"]].to_csv(out_csv, index=False)
print(f"  CSV con scores: {out_csv}")

# ── 11. Guardar modelo ────────────────────────────────────────────────────
model_path = RES / "xgb_model.json"
model.save_model(str(model_path))
print(f"  Modelo guardado: {model_path}")

# ── 12. Figuras ───────────────────────────────────────────────────────────
RED  = "#c0392b"
BLUE = "#2980b9"
GRAY = "#7f8c8d"

# --- Feature importance ---
importances = model.feature_importances_
idx = np.argsort(importances)[::-1]
feat_names = [ALL_FEATURES[i].replace("_enc", "").replace("pill_", "").replace("_", " ")
              for i in idx[:15]]
feat_imp   = importances[idx[:15]]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.barh(range(len(feat_names))[::-1], feat_imp, color=BLUE, alpha=0.8)
ax.set_yticks(range(len(feat_names))[::-1])
ax.set_yticklabels(feat_names, fontsize=10)
ax.set_xlabel("Importancia (gain)", fontsize=11)
ax.set_title("XGBoost — Importancia de variables\nPredicción de precio de renta (Monterrey)", fontsize=11)
ax.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "xgb_importance.png", dpi=150, bbox_inches="tight")
print(f"\n  Figura: {FIGS}/xgb_importance.png")

# --- Real vs Predicho ---
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(y_true / 1000, y_pred / 1000, alpha=0.4, s=18, color=BLUE)
lim = max(y_true.max(), y_pred.max()) / 1000
ax.plot([0, lim], [0, lim], "k--", linewidth=1.2, label="Predicción perfecta")
ax.set_xlabel("Precio real (miles MXN/mes)", fontsize=11)
ax.set_ylabel("Precio predicho (miles MXN/mes)", fontsize=11)
ax.set_title(f"Real vs Predicho — XGBoost\nR²={r2:.3f}, MAE=${mae:,.0f}", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "xgb_real_vs_pred.png", dpi=150, bbox_inches="tight")
print(f"  Figura: {FIGS}/xgb_real_vs_pred.png")

# --- Residuales ---
residuals = y_true - y_pred
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.hist(residuals / 1000, bins=40, color=BLUE, alpha=0.7, edgecolor="white")
ax.axvline(0, color=RED, linewidth=1.5, linestyle="--")
ax.set_xlabel("Residual (miles MXN)", fontsize=11)
ax.set_ylabel("Frecuencia", fontsize=11)
ax.set_title("Distribución de residuales — XGBoost", fontsize=11)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "xgb_residuals.png", dpi=150, bbox_inches="tight")
print(f"  Figura: {FIGS}/xgb_residuals.png")

plt.close("all")
print(f"\n✓ Listo.")
