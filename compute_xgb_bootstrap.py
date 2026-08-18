"""
compute_xgb_bootstrap.py — Bootstrap CIs para XGBoost (MAE y R²) sobre el test split real.
Usa el modelo guardado + mismo split que train_xgboost.py (random_state=42, test_size=0.2).
"""
import json, numpy as np, pandas as pd
import xgboost as xgb
from pathlib import Path
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

BASE      = Path(__file__).resolve().parent
WEBSIGHTS = Path(Path(__file__).resolve().parent.parent / "propiedades_enriquecido.xlsx")
RES       = BASE / "results"

np.random.seed(42)

# ── 1. Cargar datos (igual que train_xgboost.py) ──────────────────────────
df = pd.read_excel(WEBSIGHTS)
p99 = df["price"].quantile(0.99)
df  = df[df["price"].between(4500, p99)].copy()
df["userViews"] = pd.to_numeric(df["userViews"], errors="coerce")

def parse_loc(s):
    if not isinstance(s, str): return None, None
    parts = [x.strip() for x in s.split(",")]
    return (parts[0] if parts else None), (parts[1] if len(parts) > 1 else None)

df[["colonia","municipio"]] = pd.DataFrame(df["location"].apply(parse_loc).tolist(), index=df.index)

bc_values, BC_LAMBDA = boxcox(df["price"].values)
df["bc_price"] = bc_values
gm = df["bc_price"].mean(); k = 10
for cat in ["colonia","municipio"]:
    c = df.groupby(cat)["bc_price"].agg(["mean","count"])
    df[cat+"_enc"] = df[cat].map(((c["mean"]*c["count"]+gm*k)/(c["count"]+k))).fillna(gm)

AMENIDAD_MAP = {
    "gimnasio": "pill_gym", "alberca": "pill_pool",
    "jardín": "pill_garden", "jardín ": "pill_garden",
    "circuito cerrado": "pill_security", "elevador": "pill_elevator",
    "terraza": "pill_terrace", "rooftop": "pill_rooftop",
    "área de juegos": "pill_playground",
}
pill_cols = [c for c in df.columns if c.startswith("featuresPills")]
for feat_col in AMENIDAD_MAP.values():
    df[feat_col] = 0
for _, row in df.iterrows():
    pills = [str(row[c]).lower().strip() for c in pill_cols if pd.notna(row[c])]
    for keyword, feat_col in AMENIDAD_MAP.items():
        if any(keyword in p for p in pills):
            df.at[row.name, feat_col] = 1

def parse_days(s):
    import re
    if pd.isna(s): return np.nan
    s = str(s).lower()
    if "hoy" in s or "today" in s: return 0
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else np.nan

df["days_listed"] = df["publishedSince"].apply(parse_days) if "publishedSince" in df.columns else 22
df["userViews"] = pd.to_numeric(df["userViews"], errors="coerce")

ALL_FEATURES = [
    "m2","recamaras","banos","estacionamientos",
    "lat","lon","userViews","days_listed","amenidades_count",
    "Lujo","Amueblado","Nuevo",
    "pill_gym","pill_pool","pill_garden","pill_garden","pill_security",
    "pill_elevator","pill_terrace","pill_rooftop","pill_playground",
    "colonia_enc","municipio_enc"
]

# Parsear lat/lon
latlon = df["latlon"].str.split(",", expand=True) if "latlon" in df.columns else None
if latlon is not None:
    df["lat"] = pd.to_numeric(latlon[0], errors="coerce")
    df["lon"] = pd.to_numeric(latlon[1], errors="coerce")
df["amenidades_count"] = df[list(dict.fromkeys(AMENIDAD_MAP.values()))].sum(axis=1)

df_feat = df[ALL_FEATURES].copy().astype(float)
medians = df_feat.median()
df_feat = df_feat.fillna(medians)
X = df_feat.values
y = df["bc_price"].values

# ── 2. Mismo split que train_xgboost.py ──────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  n_train={len(X_tr)}, n_test={len(X_te)}")

# ── 3. Cargar modelo guardado ─────────────────────────────────────────────
model = xgb.XGBRegressor()
model.load_model(str(RES / "xgb_model.json"))
print(f"  Modelo cargado ({model.n_features_in_} features)")

# ── 4. Predicciones en test set ───────────────────────────────────────────
y_pred_bc = model.predict(X_te)
y_true    = inv_boxcox(y_te, BC_LAMBDA)
y_pred    = inv_boxcox(y_pred_bc, BC_LAMBDA)

mae_point = mean_absolute_error(y_true, y_pred)
r2_point  = r2_score(y_true, y_pred)
print(f"\n  MAE  = ${mae_point:,.0f} MXN/mes")
print(f"  R²   = {r2_point:.4f}")

# ── 5. Bootstrap (10,000 resamples) ──────────────────────────────────────
N_BOOT = 10_000
n      = len(y_true)
mae_boots, r2_boots = [], []

for _ in range(N_BOOT):
    idx = np.random.randint(0, n, n)
    mae_boots.append(mean_absolute_error(y_true[idx], y_pred[idx]))
    r2_boots.append(r2_score(y_true[idx], y_pred[idx]))

mae_lo, mae_hi = np.percentile(mae_boots, [2.5, 97.5])
r2_lo,  r2_hi  = np.percentile(r2_boots,  [2.5, 97.5])

print(f"\n  Bootstrap 95% CI (n={n}, {N_BOOT:,} resamples):")
print(f"  MAE = ${mae_point:,.0f}  [${mae_lo:,.0f} – ${mae_hi:,.0f}]")
print(f"  R²  = {r2_point:.3f}   [{r2_lo:.3f} – {r2_hi:.3f}]")

# ── 6. Guardar ────────────────────────────────────────────────────────────
result = {
    "n_test": n,
    "n_boot": N_BOOT,
    "mae":    round(mae_point, 2),
    "mae_ci_lo": round(mae_lo, 2),
    "mae_ci_hi": round(mae_hi, 2),
    "r2":     round(r2_point, 4),
    "r2_ci_lo": round(r2_lo, 4),
    "r2_ci_hi": round(r2_hi, 4),
}
(RES / "xgb_bootstrap_ci.json").write_text(json.dumps(result, indent=2))
print(f"\n✓ Guardado: results/xgb_bootstrap_ci.json")
