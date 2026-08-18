"""
boxcox_experiment.py — Compara log1p vs Box-Cox como transformación del target
Reporta R², MAE y el lambda óptimo encontrado por Box-Cox.
"""
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE  = Path(__file__).resolve().parent
INPUT = Path(Path(__file__).resolve().parent.parent / "propiedades_enriquecido.xlsx")

# ── Carga y features (igual que train_xgboost.py) ────────────────────────────

df = pd.read_excel(INPUT)
p99 = df["price"].quantile(0.99)
df  = df[df["price"].between(4500, p99)].copy()

def parse_loc(s):
    if not isinstance(s, str): return None, None
    parts = [x.strip() for x in s.split(",")]
    return (parts[0] if parts else None), (parts[1] if len(parts) > 1 else None)

def parse_latlon(s):
    try:
        lat, lon = str(s).split(",")
        return float(lat.strip()), float(lon.strip())
    except Exception:
        return np.nan, np.nan

df[["lat","lon"]] = pd.DataFrame(df["latlon"].apply(parse_latlon).tolist(), index=df.index)
df[["colonia","municipio"]] = pd.DataFrame(df["location"].apply(parse_loc).tolist(), index=df.index)

AMENIDAD_MAP = {
    "gimnasio":"pill_gym","alberca":"pill_pool","jardín":"pill_garden",
    "jardín ":"pill_garden","circuito cerrado":"pill_security",
    "elevador":"pill_elevator","terraza":"pill_terrace",
    "rooftop":"pill_rooftop","área de juegos":"pill_playground",
}
pill_cols = [c for c in df.columns if c.startswith("featuresPills")]
for col in AMENIDAD_MAP.values():
    df[col] = 0
for _, row in df.iterrows():
    pills = [str(row[c]).lower().strip() for c in pill_cols if pd.notna(row[c])]
    for kw, col in AMENIDAD_MAP.items():
        if any(kw in p for p in pills):
            df.at[row.name, col] = 1

df["amenidades_count"] = df[[c for c in AMENIDAD_MAP.values()]].sum(axis=1)

import re as _re
def parse_days(s):
    if pd.isna(s): return np.nan
    s = str(s).lower()
    if "hoy" in s or "today" in s: return 0
    m = _re.search(r"(\d+)", s)
    return int(m.group(1)) if m else np.nan

df["days_listed"] = df["publishedSince"].apply(parse_days)
df["userViews"]   = pd.to_numeric(df["userViews"], errors="coerce")

NUM_FEATURES = ["m2","recamaras","banos","estacionamientos",
                "lat","lon","userViews","days_listed","amenidades_count"]
BIN_FEATURES = ["Lujo","Amueblado","Nuevo"] + list(set(AMENIDAD_MAP.values()))
ALL_FEATURES = NUM_FEATURES + BIN_FEATURES + ["colonia_enc","municipio_enc"]

PARAMS = dict(n_estimators=500, max_depth=5, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8,
              reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def run_experiment(transform_name, y_transform_fn, y_inverse_fn, lambda_=None):
    """Entrena XGBoost con una transformación dada y reporta métricas."""
    df_exp = df.copy()
    df_exp["y_transformed"] = y_transform_fn(df_exp["price"].values)
    global_mean = df_exp["y_transformed"].mean()

    for cat in ["colonia","municipio"]:
        counts = df_exp.groupby(cat)["y_transformed"].agg(["mean","count"])
        smoothed = (counts["mean"]*counts["count"] + global_mean*10) / (counts["count"]+10)
        df_exp[f"{cat}_enc"] = df_exp[cat].map(smoothed).fillna(global_mean)

    df_feat = df_exp[ALL_FEATURES].copy().astype(float)
    df_feat = df_feat.fillna(df_feat.median())
    X = df_feat.values
    y = df_exp["y_transformed"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(**PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    y_pred_t = model.predict(X_te)
    y_pred   = y_inverse_fn(y_pred_t, lambda_) if lambda_ is not None else y_inverse_fn(y_pred_t)
    y_true   = y_inverse_fn(y_te, lambda_) if lambda_ is not None else y_inverse_fn(y_te)

    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    cv_r2 = cross_val_score(model, X, y, cv=kf, scoring="r2")

    lam_str = f"  λ = {lambda_:.4f}" if lambda_ is not None else ""
    print(f"\n{'─'*50}")
    print(f"  Transformación: {transform_name}{lam_str}")
    print(f"{'─'*50}")
    print(f"  R² (test)      : {r2:.4f}")
    print(f"  CV R² (5-fold) : {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
    print(f"  MAE            : {mae:,.0f} MXN/month")
    return r2, mae, cv_r2.mean()

# ── Experimento 1: log1p (baseline actual) ───────────────────────────────────
r2_log, mae_log, cv_log = run_experiment(
    "log1p  (actual)",
    lambda y: np.log1p(y),
    lambda y, _=None: np.expm1(y),
)

# ── Experimento 2: Box-Cox ───────────────────────────────────────────────────
prices = df["price"].values
_, lambda_opt = boxcox(prices)
print(f"\nBox-Cox lambda óptimo encontrado: {lambda_opt:.4f}")

r2_bc, mae_bc, cv_bc = run_experiment(
    "Box-Cox",
    lambda y: boxcox(y, lmbda=lambda_opt),
    lambda y, lam: inv_boxcox(y, lam),
    lambda_=lambda_opt,
)

# ── Resumen ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  RESUMEN COMPARATIVO")
print(f"{'='*50}")
print(f"  {'Transformación':<20} {'R² test':>8} {'CV R²':>8} {'MAE':>10}")
print(f"  {'─'*48}")
print(f"  {'log1p (actual)':<20} {r2_log:>8.4f} {cv_log:>8.3f} {mae_log:>9,.0f}")
print(f"  {'Box-Cox (λ='+f'{lambda_opt:.3f})':<20} {r2_bc:>8.4f} {cv_bc:>8.3f} {mae_bc:>9,.0f}")

delta_r2 = r2_bc - r2_log
print(f"\n  ΔR² (Box-Cox - log1p) = {delta_r2:+.4f}")
if abs(delta_r2) < 0.005:
    print("  → Diferencia no significativa (<0.005). log1p es suficiente.")
elif delta_r2 > 0:
    print(f"  → Box-Cox mejora R² en {delta_r2:.4f}. Vale la pena.")
else:
    print(f"  → log1p supera a Box-Cox. Mantener transformación actual.")
