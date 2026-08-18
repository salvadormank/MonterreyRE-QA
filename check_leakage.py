"""
check_leakage.py — Compara XGBoost con y sin features de leakage
(userViews, days_listed no están disponibles al tasar un listado nuevo)
"""
import pandas as pd, numpy as np, warnings, json
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path

RES = Path("./results")

df = pd.read_excel(Path(__file__).resolve().parent.parent / "propiedades_enriquecido.xlsx")
p99 = df['price'].quantile(0.99)
df  = df[df['price'].between(4500, p99)].copy()
df['userViews'] = pd.to_numeric(df['userViews'], errors='coerce')

df[['colonia','municipio']] = pd.DataFrame(
    df['location'].apply(lambda s: [x.strip() for x in str(s).split(',')][:2]
    if isinstance(s,str) else [None,None]).tolist(), index=df.index)
df['log_price'] = np.log1p(df['price'])
gm = df['log_price'].mean(); k=10
for cat in ['colonia','municipio']:
    c = df.groupby(cat)['log_price'].agg(['mean','count'])
    df[cat+'_enc'] = df[cat].map(((c['mean']*c['count']+gm*k)/(c['count']+k))).fillna(gm)

latlon = df['latlon'].str.split(',', expand=True)
df['lat'] = pd.to_numeric(latlon[0], errors='coerce')
df['lon']  = pd.to_numeric(latlon[1], errors='coerce')

if 'publishedSince' in df.columns:
    df['days_listed'] = (pd.Timestamp.today() - pd.to_datetime(df['publishedSince'], errors='coerce')).dt.days.fillna(22)
else:
    df['days_listed'] = 22

pill_cols = [c for c in df.columns if 'featuresPills' in c]
df['amenidades_count'] = df[pill_cols].notna().sum(axis=1)

FEATS_FULL = ['m2','recamaras','banos','estacionamientos','lat','lon','userViews',
              'days_listed','amenidades_count','Lujo','Amueblado','Nuevo',
              'colonia_enc','municipio_enc']

FEATS_NO_LEAK = ['m2','recamaras','banos','estacionamientos','lat','lon',
                 'amenidades_count','Lujo','Amueblado','Nuevo',
                 'colonia_enc','municipio_enc']

y = df['log_price'].values

params = dict(n_estimators=500, max_depth=5, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8,
              reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1)

print("="*60)
print("  LEAKAGE CHECK: XGBoost con y sin userViews/days_listed")
print("="*60)

results = {}
for label, feats in [("With userViews+days_listed", FEATS_FULL),
                     ("Without (no leakage)",        FEATS_NO_LEAK)]:
    X = df[feats].fillna(0).values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = xgb.XGBRegressor(**params)
    m.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    yp = m.predict(Xte)
    cv = cross_val_score(m, X, y, cv=5, scoring='r2')
    r2  = round(float(r2_score(yte, yp)), 3)
    mae = round(float(mean_absolute_error(np.expm1(yte), np.expm1(yp))), 0)
    print(f"\n  {label}")
    print(f"    R² test : {r2:.3f}")
    print(f"    CV R²   : {cv.mean():.3f} ± {cv.std():.3f}")
    print(f"    MAE     : ${mae:,.0f} MXN/month")
    results[label] = dict(r2=r2, cv_r2=round(float(cv.mean()),3),
                          cv_std=round(float(cv.std()),3), mae=mae,
                          n_features=len(feats))

delta_r2  = results["With userViews+days_listed"]["r2"] - results["Without (no leakage)"]["r2"]
delta_mae = results["Without (no leakage)"]["mae"] - results["With userViews+days_listed"]["mae"]
print(f"\n  ΔR²  (leakage impact): {delta_r2:+.3f}")
print(f"  ΔMAE (leakage impact): ${delta_mae:+,.0f} MXN/month")

json.dump(results, open(RES / "leakage_check.json", "w"), indent=2)
print(f"\n✓ Guardado: results/leakage_check.json")
