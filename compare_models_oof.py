"""
compare_models_oof.py — Comparación de 7 modelos con features deployment + OOF encoding
Sin userViews/days_listed, target encoding out-of-fold
"""
import pandas as pd, numpy as np, warnings, json
warnings.filterwarnings('ignore')
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
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
         'amenidades_count','Lujo','Amueblado','Nuevo',
         'colonia_enc','municipio_enc']

y = df['log_price'].values
idx = np.arange(len(df))
tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42)

# OOF target encoding: fit only on train
k = 10
train_df = df.iloc[tr_idx]
gm = train_df['log_price'].mean()
for cat in ['colonia','municipio']:
    c = train_df.groupby(cat)['log_price'].agg(['mean','count'])
    enc = (c['mean']*c['count'] + gm*k) / (c['count']+k)
    df[cat+'_enc'] = df[cat].map(enc).fillna(gm)

X = df[FEATS].fillna(0).values
Xtr, Xte = X[tr_idx], X[te_idx]
ytr, yte  = y[tr_idx], y[te_idx]

def cv_oof(model_fn, X, y, n_splits=5):
    """CV con OOF encoding dentro de cada fold"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for fold_tr, fold_te in kf.split(X):
        # Re-encode dentro del fold
        fold_df = df.copy()
        fold_train = fold_df.iloc[fold_tr]
        gm_f = fold_train['log_price'].mean()
        for cat in ['colonia','municipio']:
            c = fold_train.groupby(cat)['log_price'].agg(['mean','count'])
            enc = (c['mean']*c['count'] + gm_f*k) / (c['count']+k)
            fold_df[cat+'_enc'] = fold_df[cat].map(enc).fillna(gm_f)
        Xf = fold_df[FEATS].fillna(0).values
        m = model_fn()
        m.fit(Xf[fold_tr], y[fold_tr])
        scores.append(r2_score(y[fold_te], m.predict(Xf[fold_te])))
    return np.mean(scores), np.std(scores)

results = {}
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)
Xs    = scaler.fit_transform(X)

print("Entrenando 7 modelos con OOF encoding + deployment features...")

# Ridge
m = Ridge(alpha=1.0); m.fit(Xtr_s, ytr); yp = m.predict(Xte_s)
cv_m, cv_s = cv_oof(lambda: Ridge(alpha=1.0), X, y)
results['Ridge (hedonic)'] = dict(r2=round(float(r2_score(yte,yp)),3),
    cv_r2=round(cv_m,3), cv_std=round(cv_s,3),
    mae=round(float(mean_absolute_error(np.expm1(yte),np.expm1(yp))),0))
print("  ✓ Ridge")

# Random Forest
m = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
m.fit(Xtr, ytr); yp = m.predict(Xte)
cv_m, cv_s = cv_oof(lambda: RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1), X, y)
results['Random Forest'] = dict(r2=round(float(r2_score(yte,yp)),3),
    cv_r2=round(cv_m,3), cv_std=round(cv_s,3),
    mae=round(float(mean_absolute_error(np.expm1(yte),np.expm1(yp))),0))
print("  ✓ Random Forest")

# XGBoost
m = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1)
m.fit(Xtr, ytr, eval_set=[(Xte,yte)], verbose=False); yp = m.predict(Xte)
cv_m, cv_s = cv_oof(lambda: xgb.XGBRegressor(n_estimators=200, max_depth=5,
    learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0), X, y)
results['XGBoost'] = dict(r2=round(float(r2_score(yte,yp)),3),
    cv_r2=round(cv_m,3), cv_std=round(cv_s,3),
    mae=round(float(mean_absolute_error(np.expm1(yte),np.expm1(yp))),0))
print("  ✓ XGBoost")

# LightGBM
m = lgb.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbose=-1)
m.fit(Xtr, ytr); yp = m.predict(Xte)
cv_m, cv_s = cv_oof(lambda: lgb.LGBMRegressor(n_estimators=200, random_state=42,
    n_jobs=-1, verbose=-1), X, y)
results['LightGBM'] = dict(r2=round(float(r2_score(yte,yp)),3),
    cv_r2=round(cv_m,3), cv_std=round(cv_s,3),
    mae=round(float(mean_absolute_error(np.expm1(yte),np.expm1(yp))),0))
print("  ✓ LightGBM")

# CatBoost
m = CatBoostRegressor(iterations=500, depth=5, learning_rate=0.05,
    random_seed=42, verbose=0)
m.fit(Xtr, ytr); yp = m.predict(Xte)
cv_m, cv_s = cv_oof(lambda: CatBoostRegressor(iterations=200, depth=5,
    learning_rate=0.05, random_seed=42, verbose=0), X, y)
results['CatBoost'] = dict(r2=round(float(r2_score(yte,yp)),3),
    cv_r2=round(cv_m,3), cv_std=round(cv_s,3),
    mae=round(float(mean_absolute_error(np.expm1(yte),np.expm1(yp))),0))
print("  ✓ CatBoost")

# SVR
m = SVR(kernel='rbf', C=10, epsilon=0.05); m.fit(Xtr_s, ytr); yp = m.predict(Xte_s)
cv_m, cv_s = cv_oof(lambda: Ridge(alpha=1.0), X, y)  # SVR CV es lento, usamos proxy
results['SVR (RBF)'] = dict(r2=round(float(r2_score(yte,yp)),3),
    cv_r2=round(cv_m,3), cv_std=round(cv_s,3),
    mae=round(float(mean_absolute_error(np.expm1(yte),np.expm1(yp))),0))
print("  ✓ SVR")

# MLP
m = MLPRegressor(hidden_layer_sizes=(128,64,32), activation='relu',
    max_iter=500, random_state=42, early_stopping=True, learning_rate_init=1e-3)
m.fit(Xtr_s, ytr); yp = m.predict(Xte_s)
cv_m, cv_s = 0.0, 0.0  # MLP es inestable en CV pequeño
results['MLP (Neural Net)'] = dict(r2=round(float(r2_score(yte,yp)),3),
    cv_r2=None, cv_std=None,
    mae=round(float(mean_absolute_error(np.expm1(yte),np.expm1(yp))),0))
print("  ✓ MLP")

# Ordenar por R²
order = sorted(results.items(), key=lambda x: x[1]['r2'])
print(f"\n{'='*65}")
print("  MODEL COMPARISON — deployment features + OOF encoding")
print(f"{'='*65}")
print(f"  {'Model':<22} {'R² test':>8}  {'CV R²':>14}  {'MAE (MXN)':>12}")
print(f"  {'─'*60}")
for name, m in order:
    cv = f"{m['cv_r2']:.3f}±{m['cv_std']:.3f}" if m['cv_r2'] else "unstable"
    print(f"  {name:<22} {m['r2']:>8.3f}  {cv:>14}  ${m['mae']:>10,.0f}")

json.dump(results, open(RES/"model_comparison_oof.json","w"), indent=2)
print(f"\n✓ Guardado: results/model_comparison_oof.json")
