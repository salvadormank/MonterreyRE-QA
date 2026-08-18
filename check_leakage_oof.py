"""
check_leakage_oof.py — XGBoost con OOF target encoding (sin leakage)
Compara: encoding en dataset completo vs encoding out-of-fold
"""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
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

FEATS_NO_LEAK = ['m2','recamaras','banos','estacionamientos','lat','lon',
                 'amenidades_count','Lujo','Amueblado','Nuevo',
                 'colonia_enc','municipio_enc']

params = dict(n_estimators=500, max_depth=5, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8,
              reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1)

k_smooth = 10

# ── Método 1: encoding en dataset completo (original, con leakage) ────────────
gm = df['log_price'].mean()
for cat in ['colonia','municipio']:
    c = df.groupby(cat)['log_price'].agg(['mean','count'])
    df[cat+'_enc'] = df[cat].map(
        (c['mean']*c['count'] + gm*k_smooth) / (c['count']+k_smooth)
    ).fillna(gm)

X_full = df[FEATS_NO_LEAK].fillna(0).values
y = df['log_price'].values
Xtr, Xte, ytr, yte = train_test_split(X_full, y, test_size=0.2, random_state=42)

m1 = xgb.XGBRegressor(**params)
m1.fit(Xtr, ytr, eval_set=[(Xte,yte)], verbose=False)
yp1 = m1.predict(Xte)
r2_full  = r2_score(yte, yp1)
mae_full = mean_absolute_error(np.expm1(yte), np.expm1(yp1))

# ── Método 2: OOF target encoding (correcto, sin leakage) ────────────────────
# Encoding calculado solo con folds de entrenamiento
df_oof = df.copy()
df_oof['colonia_enc'] = np.nan
df_oof['municipio_enc'] = np.nan

# Split primero, luego encoding solo en train
idx = np.arange(len(df_oof))
tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42)

# Calcular encoding solo con filas de train
train_df = df_oof.iloc[tr_idx]
gm_tr = train_df['log_price'].mean()

for cat in ['colonia','municipio']:
    c = train_df.groupby(cat)['log_price'].agg(['mean','count'])
    enc_map = (c['mean']*c['count'] + gm_tr*k_smooth) / (c['count']+k_smooth)
    df_oof[cat+'_enc'] = df_oof[cat].map(enc_map).fillna(gm_tr)

X_oof = df_oof[FEATS_NO_LEAK].fillna(0).values
Xtr2 = X_oof[tr_idx]; Xte2 = X_oof[te_idx]
ytr2 = y[tr_idx];     yte2 = y[te_idx]

m2 = xgb.XGBRegressor(**params)
m2.fit(Xtr2, ytr2, eval_set=[(Xte2,yte2)], verbose=False)
yp2 = m2.predict(Xte2)
r2_oof  = r2_score(yte2, yp2)
mae_oof = mean_absolute_error(np.expm1(yte2), np.expm1(yp2))

# ── CV con OOF encoding ───────────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2s = []
for fold_tr, fold_te in kf.split(df_oof):
    fold_df = df_oof.copy()
    fold_tr_df = fold_df.iloc[fold_tr]
    gm_fold = fold_tr_df['log_price'].mean()
    for cat in ['colonia','municipio']:
        c = fold_tr_df.groupby(cat)['log_price'].agg(['mean','count'])
        enc_map = (c['mean']*c['count'] + gm_fold*k_smooth) / (c['count']+k_smooth)
        fold_df[cat+'_enc'] = fold_df[cat].map(enc_map).fillna(gm_fold)
    Xf = fold_df[FEATS_NO_LEAK].fillna(0).values
    mf = xgb.XGBRegressor(**params)
    mf.fit(Xf[fold_tr], y[fold_tr], verbose=False)
    cv_r2s.append(r2_score(y[fold_te], mf.predict(Xf[fold_te])))

cv_mean = np.mean(cv_r2s)
cv_std  = np.std(cv_r2s)

print("="*60)
print("  TARGET ENCODING: Leakage vs OOF (deployment features)")
print("="*60)
print(f"\n  {'Method':<35} {'R² test':>8}  {'MAE (MXN)':>12}")
print(f"  {'─'*58}")
print(f"  {'Full-dataset encoding (leakage)':<35} {r2_full:>8.3f}  ${mae_full:>10,.0f}")
print(f"  {'OOF encoding (corrected)':<35} {r2_oof:>8.3f}  ${mae_oof:>10,.0f}")
print(f"\n  OOF CV R²: {cv_mean:.3f} ± {cv_std:.3f}")
print(f"\n  ΔR² from fixing encoding: {r2_oof - r2_full:+.3f}")

results = {
    "full_dataset_encoding": {"r2": round(r2_full,3), "mae": round(mae_full,0)},
    "oof_encoding":          {"r2": round(r2_oof,3),  "mae": round(mae_oof,0),
                              "cv_r2": round(cv_mean,3), "cv_std": round(cv_std,3)}
}
json.dump(results, open(RES/"oof_encoding_check.json","w"), indent=2)
print(f"\n✓ Guardado: results/oof_encoding_check.json")
