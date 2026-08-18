"""
baselines_and_tests.py
1. Baseline: mediana por barrio (colonia)
2. Diebold-Mariano test: XGBoost vs CatBoost vs LightGBM
"""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats
from pathlib import Path

RES = Path("./results")

df = pd.read_excel(Path(__file__).resolve().parent.parent / "propiedades_enriquecido.xlsx")
p99 = df['price'].quantile(0.99)
df  = df[df['price'].between(4500, p99)].copy()

df[['colonia','municipio']] = pd.DataFrame(
    df['location'].apply(lambda s: [x.strip() for x in str(s).split(',')][:2]
    if isinstance(s,str) else [None,None]).tolist(), index=df.index)
df['log_price'] = np.log1p(df['price'])

latlon = df['latlon'].str.split(',', expand=True)
df['lat'] = pd.to_numeric(latlon[0], errors='coerce')
df['lon']  = pd.to_numeric(latlon[1], errors='coerce')
pill_cols = [c for c in df.columns if 'featuresPills' in c]
df['amenidades_count'] = df[pill_cols].notna().sum(axis=1)

y = df['log_price'].values
idx = np.arange(len(df))
tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42)

# OOF target encoding
k = 10
train_df = df.iloc[tr_idx]
gm = train_df['log_price'].mean()
for cat in ['colonia','municipio']:
    c = train_df.groupby(cat)['log_price'].agg(['mean','count'])
    enc = (c['mean']*c['count'] + gm*k) / (c['count']+k)
    df[cat+'_enc'] = df[cat].map(enc).fillna(gm)

FEATS = ['m2','recamaras','banos','estacionamientos','lat','lon',
         'amenidades_count','Lujo','Amueblado','Nuevo','colonia_enc','municipio_enc']
X = df[FEATS].fillna(0).values
Xtr, Xte = X[tr_idx], X[te_idx]
ytr, yte  = y[tr_idx], y[te_idx]

# ── Baseline 1: mediana global ────────────────────────────────────────────────
median_global = np.median(np.expm1(ytr))
yp_median_global = np.full(len(yte), np.log1p(median_global))
r2_mg  = r2_score(yte, yp_median_global)
mae_mg = mean_absolute_error(np.expm1(yte), np.expm1(yp_median_global))

# ── Baseline 2: mediana por colonia ──────────────────────────────────────────
col_median = train_df.groupby('colonia')['log_price'].median()
global_med = np.log1p(median_global)
te_preds = df.iloc[te_idx]['colonia'].map(col_median).fillna(global_med).values
r2_mc  = r2_score(yte, te_preds)
mae_mc = mean_absolute_error(np.expm1(yte), np.expm1(te_preds))

print("="*60)
print("  BASELINES ADICIONALES")
print("="*60)
print(f"  {'Modelo':<30} {'R² test':>8}  {'MAE (MXN)':>12}")
print(f"  {'─'*55}")
print(f"  {'Mediana global':<30} {r2_mg:>8.3f}  ${mae_mg:>10,.0f}")
print(f"  {'Mediana por colonia':<30} {r2_mc:>8.3f}  ${mae_mc:>10,.0f}")

# ── Entrenar XGBoost, CatBoost, LightGBM para DM test ────────────────────────
params_xgb = dict(n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1)

m_xgb = xgb.XGBRegressor(**params_xgb)
m_xgb.fit(Xtr, ytr, eval_set=[(Xte,yte)], verbose=False)
yp_xgb = m_xgb.predict(Xte)

m_cat = CatBoostRegressor(iterations=500, depth=5, learning_rate=0.05,
    random_seed=42, verbose=0)
m_cat.fit(Xtr, ytr)
yp_cat = m_cat.predict(Xte)

m_lgb = lgb.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
m_lgb.fit(Xtr, ytr)
yp_lgb = m_lgb.predict(Xte)

# ── Diebold-Mariano test ──────────────────────────────────────────────────────
def diebold_mariano(yp_a, yp_b, yte):
    """DM test: H0: modelos tienen igual MAE. Retorna t-stat y p-value."""
    e_a = np.abs(np.expm1(yte) - np.expm1(yp_a))
    e_b = np.abs(np.expm1(yte) - np.expm1(yp_b))
    d   = e_a - e_b  # positivo = A peor que B
    t_stat, p_val = stats.ttest_1samp(d, 0)
    return t_stat, p_val

print(f"\n{'='*60}")
print("  DIEBOLD-MARIANO TEST (MAE, dos colas)")
print("  H0: igual MAE entre modelos")
print(f"{'='*60}")
print(f"  {'Comparación':<35} {'t-stat':>8}  {'p-value':>10}  {'sig':>5}")
print(f"  {'─'*60}")

pairs = [
    ("XGBoost vs CatBoost",  yp_xgb, yp_cat),
    ("XGBoost vs LightGBM",  yp_xgb, yp_lgb),
    ("CatBoost vs LightGBM", yp_cat, yp_lgb),
]
dm_results = {}
for name, ypa, ypb in pairs:
    t, p = diebold_mariano(ypa, ypb, yte)
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
    print(f"  {name:<35} {t:>8.3f}  {p:>10.4f}  {sig:>5}")
    dm_results[name] = {"t": round(float(t),3), "p": round(float(p),4)}

r2_xgb  = round(float(r2_score(yte, yp_xgb)), 3)
r2_cat  = round(float(r2_score(yte, yp_cat)), 3)
mae_xgb = round(float(mean_absolute_error(np.expm1(yte), np.expm1(yp_xgb))), 0)
mae_cat = round(float(mean_absolute_error(np.expm1(yte), np.expm1(yp_cat))), 0)

print(f"\n  XGBoost: R²={r2_xgb}, MAE=${mae_xgb:,.0f}")
print(f"  CatBoost: R²={r2_cat}, MAE=${mae_cat:,.0f}")

output = {
    "baselines": {
        "mediana_global":  {"r2": round(float(r2_mg),3), "mae": round(float(mae_mg),0)},
        "mediana_colonia": {"r2": round(float(r2_mc),3), "mae": round(float(mae_mc),0)},
    },
    "diebold_mariano": dm_results
}
json.dump(output, open(RES/"baselines_dm.json","w"), indent=2)
print(f"\n✓ Guardado: results/baselines_dm.json")
