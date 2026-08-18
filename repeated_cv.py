"""
repeated_cv.py — 5x2 repeated cross-validation para el sistema híbrido
Reporta MAE% con intervalos de confianza robustos
"""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score, mean_absolute_error
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

FEATS = ['m2','recamaras','banos','estacionamientos','lat','lon',
         'amenidades_count','Lujo','Amueblado','Nuevo','colonia_enc','municipio_enc']

y = df['log_price'].values
k_smooth = 10

params = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8,
              reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1)

rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=42)

r2_scores, mae_scores = [], []

print("5×2 Repeated CV con OOF encoding...")
for fold_i, (tr_idx, te_idx) in enumerate(rkf.split(df)):
    fold_df = df.copy()
    fold_train = fold_df.iloc[tr_idx]
    gm_fold = fold_train['log_price'].mean()
    for cat in ['colonia','municipio']:
        c = fold_train.groupby(cat)['log_price'].agg(['mean','count'])
        enc = (c['mean']*c['count'] + gm_fold*k_smooth) / (c['count']+k_smooth)
        fold_df[cat+'_enc'] = fold_df[cat].map(enc).fillna(gm_fold)

    X = fold_df[FEATS].fillna(0).values
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte  = y[tr_idx], y[te_idx]

    m = xgb.XGBRegressor(**params)
    m.fit(Xtr, ytr, verbose=False)
    yp = m.predict(Xte)

    r2_scores.append(r2_score(yte, yp))
    mae_scores.append(mean_absolute_error(np.expm1(yte), np.expm1(yp)))
    print(f"  Fold {fold_i+1}/10: R²={r2_scores[-1]:.3f}, MAE=${mae_scores[-1]:,.0f}")

r2_arr  = np.array(r2_scores)
mae_arr = np.array(mae_scores)

print(f"\n{'='*55}")
print("  5×2 REPEATED CV — XGBoost (OOF encoding)")
print(f"{'='*55}")
print(f"  R²  : {r2_arr.mean():.3f} ± {r2_arr.std():.3f}  "
      f"[95% CI: {np.percentile(r2_arr,2.5):.3f}–{np.percentile(r2_arr,97.5):.3f}]")
print(f"  MAE : ${mae_arr.mean():,.0f} ± ${mae_arr.std():,.0f}  "
      f"[95% CI: ${np.percentile(mae_arr,2.5):,.0f}–${np.percentile(mae_arr,97.5):,.0f}]")

output = {
    "r2_mean": round(float(r2_arr.mean()),3),
    "r2_std":  round(float(r2_arr.std()),3),
    "r2_ci95": [round(float(np.percentile(r2_arr,2.5)),3),
                round(float(np.percentile(r2_arr,97.5)),3)],
    "mae_mean": round(float(mae_arr.mean()),0),
    "mae_std":  round(float(mae_arr.std()),0),
    "mae_ci95": [round(float(np.percentile(mae_arr,2.5)),0),
                 round(float(np.percentile(mae_arr,97.5)),0)],
}
json.dump(output, open(RES/"repeated_cv.json","w"), indent=2)
print(f"\n✓ Guardado: results/repeated_cv.json")
