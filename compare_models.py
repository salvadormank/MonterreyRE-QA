import pandas as pd, numpy as np, warnings, json
warnings.filterwarnings('ignore')
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from pathlib import Path

RES = Path("./results")

df = pd.read_excel(Path(__file__).resolve().parent.parent / "propiedades_enriquecido.xlsx")
p99 = df['price'].quantile(0.99)
df  = df[df['price'].between(4500, p99)].copy()
df['userViews'] = pd.to_numeric(df['userViews'], errors='coerce')

df[['colonia','municipio']] = pd.DataFrame(
    df['location'].apply(lambda s: [x.strip() for x in str(s).split(',')][:2]
    if isinstance(s,str) else [None,None]).tolist(), index=df.index)
from scipy.stats import boxcox

latlon = df['latlon'].str.split(',', expand=True)
df['lat'] = pd.to_numeric(latlon[0], errors='coerce')
df['lon']  = pd.to_numeric(latlon[1], errors='coerce')

df[['colonia','municipio']] = pd.DataFrame(
    df['location'].apply(lambda s: [x.strip() for x in str(s).split(',')][:2]
    if isinstance(s,str) else [None,None]).tolist(), index=df.index)

# Box-Cox target (igual que train_xgboost.py)
bc_values, BC_LAMBDA = boxcox(df['price'].values)
df['bc_price'] = bc_values
print(f"  Box-Cox lambda: {BC_LAMBDA:.4f}")
gm = df['bc_price'].mean(); k=10
for cat in ['colonia','municipio']:
    c = df.groupby(cat)['bc_price'].agg(['mean','count'])
    df[cat+'_enc'] = df[cat].map(((c['mean']*c['count']+gm*k)/(c['count']+k))).fillna(gm)

# days_listed
if 'publishedSince' in df.columns:
    df['days_listed'] = (pd.Timestamp.today() - pd.to_datetime(df['publishedSince'], errors='coerce')).dt.days.fillna(22)
else:
    df['days_listed'] = 22

# Amenidades individuales (igual que train_xgboost.py)
AMENIDAD_MAP = {
    "alberca": "pill_pool", "gimnasio": "pill_gym", "estacionamiento de visitas": "pill_visitor_parking",
    "jardín": "pill_garden", "vigilancia": "pill_security", "elevador": "pill_elevator",
    "terraza": "pill_terrace", "rooftop": "pill_rooftop", "área de juegos": "pill_playground",
    "circuito cerrado": "pill_cctv",
}
pill_cols = [c for c in df.columns if c.startswith('featuresPills')]
for feat_col in AMENIDAD_MAP.values():
    df[feat_col] = 0
for _, row in df.iterrows():
    pills = [str(row[c]).lower().strip() for c in pill_cols if pd.notna(row[c])]
    for keyword, feat_col in AMENIDAD_MAP.items():
        if any(keyword in p for p in pills):
            df.at[row.name, feat_col] = 1
df['amenidades_count'] = df[list(AMENIDAD_MAP.values())].sum(axis=1)

FEATS = ['m2','recamaras','banos','estacionamientos','lat','lon','userViews',
         'days_listed','amenidades_count','Lujo','Amueblado','Nuevo',
         'colonia_enc','municipio_enc'] + list(AMENIDAD_MAP.values())

X = df[FEATS].fillna(0).values
y = df['bc_price'].values
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)

# Para invertir Box-Cox en MAE
from scipy.special import inv_boxcox
def inv_bc(vals): return inv_boxcox(vals, BC_LAMBDA)

results = {}

# Ridge (hedonic baseline)
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)
m1 = Ridge(alpha=1.0)
m1.fit(Xtr_s, ytr)
yp1 = m1.predict(Xte_s)
cv1 = cross_val_score(m1, scaler.fit_transform(X), y, cv=5, scoring='r2')
results['Ridge (hedonic)'] = dict(
    r2=round(float(r2_score(yte,yp1)),3),
    cv_r2=round(float(cv1.mean()),3),
    cv_std=round(float(cv1.std()),3),
    mae=round(float(mean_absolute_error(inv_bc(yte),inv_bc(yp1))),0)
)

# Random Forest
m2 = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
m2.fit(Xtr, ytr)
yp2 = m2.predict(Xte)
cv2 = cross_val_score(m2, X, y, cv=5, scoring='r2')
results['Random Forest'] = dict(
    r2=round(float(r2_score(yte,yp2)),3),
    cv_r2=round(float(cv2.mean()),3),
    cv_std=round(float(cv2.std()),3),
    mae=round(float(mean_absolute_error(inv_bc(yte),inv_bc(yp2))),0)
)

# XGBoost
m3 = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1)
m3.fit(Xtr, ytr, eval_set=[(Xte,yte)], verbose=False)
yp3 = m3.predict(Xte)
cv3 = cross_val_score(m3, X, y, cv=5, scoring='r2')
results['XGBoost'] = dict(
    r2=round(float(r2_score(yte,yp3)),3),
    cv_r2=round(float(cv3.mean()),3),
    cv_std=round(float(cv3.std()),3),
    mae=round(float(mean_absolute_error(inv_bc(yte),inv_bc(yp3))),0)
)

print("="*60)
print("  MODEL COMPARISON (same features, same split)")
print("="*60)
print(f"  {'Model':<22} {'R² test':>8}  {'CV R²':>12}  {'MAE (MXN)':>12}")
print(f"  {'─'*56}")
for name, m in results.items():
    print(f"  {name:<22} {m['r2']:>8.3f}  {m['cv_r2']:>5.3f}±{m['cv_std']:.3f}  ${m['mae']:>10,.0f}")

# LightGBM
m4 = lgb.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
                        verbose=-1)
m4.fit(Xtr, ytr)
yp4 = m4.predict(Xte)
cv4 = cross_val_score(m4, X, y, cv=5, scoring='r2')
results['LightGBM'] = dict(
    r2=round(float(r2_score(yte,yp4)),3),
    cv_r2=round(float(cv4.mean()),3),
    cv_std=round(float(cv4.std()),3),
    mae=round(float(mean_absolute_error(inv_bc(yte),inv_bc(yp4))),0)
)

# CatBoost
m5 = CatBoostRegressor(iterations=500, depth=5, learning_rate=0.05,
                        random_seed=42, verbose=0)
m5.fit(Xtr, ytr)
yp5 = m5.predict(Xte)
cv5 = cross_val_score(m5, X, y, cv=5, scoring='r2')
results['CatBoost'] = dict(
    r2=round(float(r2_score(yte,yp5)),3),
    cv_r2=round(float(cv5.mean()),3),
    cv_std=round(float(cv5.std()),3),
    mae=round(float(mean_absolute_error(inv_bc(yte),inv_bc(yp5))),0)
)

# SVR
m6 = SVR(kernel='rbf', C=10, epsilon=0.05)
m6.fit(Xtr_s, ytr)
yp6 = m6.predict(Xte_s)
cv6 = cross_val_score(m6, scaler.fit_transform(X), y, cv=5, scoring='r2')
results['SVR (RBF)'] = dict(
    r2=round(float(r2_score(yte,yp6)),3),
    cv_r2=round(float(cv6.mean()),3),
    cv_std=round(float(cv6.std()),3),
    mae=round(float(mean_absolute_error(inv_bc(yte),inv_bc(yp6))),0)
)

# MLP
m7 = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                   max_iter=500, random_state=42, early_stopping=True,
                   learning_rate_init=1e-3)
m7.fit(Xtr_s, ytr)
yp7 = m7.predict(Xte_s)
cv7 = cross_val_score(m7, scaler.fit_transform(X), y, cv=5, scoring='r2')
results['MLP (Neural Net)'] = dict(
    r2=round(float(r2_score(yte,yp7)),3),
    cv_r2=round(float(cv7.mean()),3),
    cv_std=round(float(cv7.std()),3),
    mae=round(float(mean_absolute_error(inv_bc(yte),inv_bc(yp7))),0)
)

print("="*60)
print("  MODEL COMPARISON (same features, same split)")
print("="*60)
print(f"  {'Model':<22} {'R² test':>8}  {'CV R²':>12}  {'MAE (MXN)':>12}")
print(f"  {'─'*56}")
for name, m in results.items():
    print(f"  {name:<22} {m['r2']:>8.3f}  {m['cv_r2']:>5.3f}±{m['cv_std']:.3f}  ${m['mae']:>10,.0f}")

json.dump(results, open(RES/"model_comparison.json","w"), indent=2)
print(f"\n✓ Saved: results/model_comparison.json")
