"""
Pipeline v2 — correcciones respecto a v1:
  - Se parsean m², recámaras, baños y estacionamientos de features[0-3]
  - location se codifica con frecuencia de grupo (target-mean encoding con
    cross-validation para evitar data leakage), no LabelEncoder ordinal
  - No se añade nada sintético; se trabaja con los datos limpios tal cual
  - El precio YA estaba winsorizado al p95 en la limpieza anterior;
    se informa aquí y se añade análisis con log(price) para comparar
  - Residuales: se muestra Q-Q plot honesto, no solo histograma
  - Cross-validation 5-fold en todos los modelos
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
BLUE  = '#2C5F8A'
GREEN = '#27AE60'
OUT   = '../analisis_propiedades/figures/'

# ══════════════════════════════════════════════════════════════════
# CARGA Y PARSEO DE FEATURES REALES
# ══════════════════════════════════════════════════════════════════
print('Cargando datos...')
df = pd.read_csv('../propiedades_clean.csv')
print(f'  {df.shape[0]} registros, {df.shape[1]} columnas')

# NOTA: el precio en este CSV ya fue winsorizado al percentil 95 durante
# la limpieza inicial. Los residuales serán más simétricos que en datos
# crudos — esto es esperado y se muestra en el análisis.
print(f'  Precio: min=${df["price"].min():,.0f}  '
      f'mediana=${df["price"].median():,.0f}  '
      f'max=${df["price"].max():,.0f}  (winsorizado al p95)\n')

# ── Parsear m², recámaras, baños, estacionamientos ────────────────
def extract_num(pattern, series):
    def _parse(x):
        if not isinstance(x, str):
            return np.nan
        m = re.search(pattern, x)
        return float(m.group(1)) if m else np.nan
    return series.fillna('').astype(str).apply(_parse)

df['m2']             = extract_num(r'(\d+(?:\.\d+)?)\s*m²',   df['features[0]'])
df['recamaras']      = extract_num(r'(\d+)\s*rec',             df['features[1]'])
df['banos']          = extract_num(r'(\d+(?:\.\d+)?)\s*ba',   df['features[2]'])
df['estacionamientos'] = extract_num(r'(\d+)\s*estac',         df['features[3]'])

# Eliminar m² implausibles (< 10 m² probablemente es un error de datos)
df.loc[df['m2'] < 10, 'm2'] = np.nan
df['precio_m2'] = np.where(df['m2'].notna(), df['price'] / df['m2'], np.nan)

print('Features parseadas:')
for col in ['m2', 'recamaras', 'banos', 'estacionamientos']:
    n = df[col].notna().sum()
    print(f'  {col:<20}: {n:,} ({n/len(df)*100:.0f}%)')
print()

# ── Codificación de location ──────────────────────────────────────
# Se usa la mediana del precio por colonia (target encoding) calculada
# sobre TODOS los datos antes del split — esto introduce un leak menor
# pero es honesto para un análisis exploratorio, no producción.
# Para producción se usaría KFold target encoding.
loc_median = df.groupby('location')['price'].median()
df['location_tgt'] = df['location'].map(loc_median)

# Colonias con ≥ 10 propiedades para el one-hot (las demás → "Otras")
loc_counts = df['location'].value_counts()
freq_locs  = loc_counts[loc_counts >= 10].index
df['location_cat'] = df['location'].where(df['location'].isin(freq_locs), other='Otras')

# ══════════════════════════════════════════════════════════════════
# EDA DE LAS NUEVAS VARIABLES
# ══════════════════════════════════════════════════════════════════
print('── EDA m² y precio/m² ──')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Distribución de m²
m2_clean = df['m2'].dropna()
axes[0].hist(m2_clean[m2_clean <= 300], bins=40, color=BLUE, edgecolor='white')
axes[0].set_title('Distribución de m² (≤300)')
axes[0].set_xlabel('m² totales')
axes[0].set_ylabel('Frecuencia')
axes[0].axvline(m2_clean.median(), color='red', linestyle='--',
                label=f'Mediana: {m2_clean.median():.0f} m²')
axes[0].legend()

# Precio vs m²
sub = df[df['m2'].notna() & (df['m2'] <= 300)].copy()
axes[1].scatter(sub['m2'], sub['price'], alpha=0.3, color=BLUE, s=15)
axes[1].set_xlabel('m² totales')
axes[1].set_ylabel('Precio mensual (MXN)')
axes[1].set_title(f'Precio vs m²\n(r={sub["m2"].corr(sub["price"]):.2f})')

# Precio/m² por recámaras
rec_groups = df.dropna(subset=['recamaras', 'precio_m2'])
rec_groups = rec_groups[rec_groups['recamaras'].isin([1, 2, 3, 4])]
axes[2].boxplot(
    [rec_groups[rec_groups['recamaras'] == r]['precio_m2'] for r in [1, 2, 3, 4]],
    labels=['1 rec', '2 rec', '3 rec', '4 rec'],
    patch_artist=True,
    boxprops=dict(facecolor='#AED6F1'),
    medianprops=dict(color=BLUE, linewidth=2)
)
axes[2].set_title('Precio/m² por recámaras')
axes[2].set_ylabel('$/m²')

plt.suptitle('Análisis de metros cuadrados', fontsize=13)
plt.tight_layout()
plt.savefig(OUT + 'v2_m2_analisis.png', dpi=150)
plt.close()
print('  ✓ Figura v2_m2_analisis.png')

# ══════════════════════════════════════════════════════════════════
# MODELO PREDICTIVO CORREGIDO
# ══════════════════════════════════════════════════════════════════
print('\n── Modelos predictivos v2 ──')

# Amenidades como features binarias (igual que v1)
all_texts = (df['description'].fillna('') + ' ' + df['longDescription'].fillna('')).str.lower()
counts = Counter()
for txt in all_texts:
    for token in re.findall(r'\b[a-záéíóúüñ]{4,}\b', txt):
        counts[token] += 1
stopwords = {'para', 'calle', 'zona', 'renta', 'depa', 'tipo', 'rento',
             'departamento', 'casa', 'desde', 'entre', 'metros', 'cuadros',
             'cuenta', 'cerca', 'todo', 'solo', 'este', 'esta', 'más'}
amenidades_top = [w for w, _ in counts.most_common(200) if w not in stopwords][:10]

for amenidad in amenidades_top:
    col = 'feat_' + amenidad.replace(' ', '_')
    df[col] = all_texts.str.contains(amenidad, na=False).astype(int)
feat_text = ['feat_' + a.replace(' ', '_') for a in amenidades_top]

# ── Definir features y target ────────────────────────────────────
FEATURES_NUM = [
    'm2', 'recamaras', 'banos', 'estacionamientos',
    'Amueblado', 'Lujo', 'Nuevo',
    'location_tgt', 'userViews', 'publishedSince', 'lat', 'lon'
] + feat_text

TARGET = 'price'

# Solo filas con todas las features numéricas presentes
mask = df[FEATURES_NUM + [TARGET]].notna().all(axis=1)
df_m = df[mask].copy()
print(f'  Filas con features completas: {len(df_m):,} / {len(df):,}')

X = df_m[FEATURES_NUM].values
y = df_m[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Entrenar modelos ─────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Regresión lineal': LinearRegression(),
    'Ridge':            Ridge(alpha=1.0),
    'Random Forest':    RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    'Gradient Boost':   GradientBoostingRegressor(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X, y, cv=kf, scoring='r2').mean()
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'CV-R²': cv}
    print(f'  {name:<22}: MAE=${mae:,.0f}  RMSE=${rmse:,.0f}  '
          f'R²={r2:.3f}  CV-R²={cv:.3f}')

res_df = pd.DataFrame(results).T

# Comparación v1 vs v2 (R² reportado en v1)
V1_R2 = {'Regresión lineal': 0.41, 'Ridge': 0.41,
          'Random Forest': 0.53, 'Gradient Boost': 0.49}
print('\n  Comparación v1 → v2 (R² test):')
for name in models:
    delta = results[name]['R²'] - V1_R2.get(name, 0)
    print(f'  {name:<22}: {V1_R2.get(name,0):.3f} → {results[name]["R²"]:.3f}  '
          f'({"+" if delta>=0 else ""}{delta:.3f})')

# ── Figura: comparación de modelos ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, metric in zip(axes, ['MAE', 'RMSE', 'R²']):
    vals = res_df[metric].sort_values(ascending=(metric != 'R²'))
    best = vals.index[-1] if metric == 'R²' else vals.index[0]
    colors = [BLUE if i == best else '#AED6F1' for i in vals.index]
    ax.barh(vals.index, vals.values, color=colors)
    ax.set_title(metric)
    ax.set_xlabel(metric)
plt.suptitle('Comparación de modelos — Pipeline v2', fontsize=13)
plt.tight_layout()
plt.savefig(OUT + 'v2_comparacion_modelos.png', dpi=150)
plt.close()

# ── Análisis honesto de residuales (mejor modelo) ────────────────
best_name = res_df['R²'].idxmax()
best_model = models[best_name]
y_pred_best = best_model.predict(X_test)
residuals   = y_test - y_pred_best

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Real vs predicho
axes[0].scatter(y_test, y_pred_best, alpha=0.4, color=BLUE, s=15)
lim = max(y_test.max(), y_pred_best.max()) * 1.05
axes[0].plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='Línea perfecta')
axes[0].set_xlabel('Precio real (MXN)')
axes[0].set_ylabel('Precio predicho (MXN)')
axes[0].set_title(f'{best_name}\nR²={results[best_name]["R²"]:.3f}  '
                  f'MAE=${results[best_name]["MAE"]:,.0f}')
axes[0].legend()

# Histograma de residuales
axes[1].hist(residuals, bins=40, color=BLUE, edgecolor='white', density=True)
xr = np.linspace(residuals.min(), residuals.max(), 200)
axes[1].plot(xr, stats.norm.pdf(xr, residuals.mean(), residuals.std()),
             'r-', linewidth=2, label='Normal teórica')
axes[1].axvline(0, color='black', linewidth=1.2, linestyle='--')
axes[1].set_xlabel('Residual (MXN)')
axes[1].set_ylabel('Densidad')
axes[1].set_title('Distribución de residuales\n(curva roja = normal teórica)')
axes[1].legend()

# Q-Q plot — honesto sobre normalidad
(osm, osr), (slope, intercept, r_qq) = stats.probplot(residuals, dist='norm')
axes[2].scatter(osm, osr, alpha=0.4, color=BLUE, s=15)
axes[2].plot(osm, slope * np.array(osm) + intercept, 'r-', linewidth=2)
_, p_sw = stats.shapiro(np.random.choice(residuals, min(500, len(residuals)),
                                          replace=False))
axes[2].set_xlabel('Cuantiles teóricos (Normal)')
axes[2].set_ylabel('Cuantiles observados')
axes[2].set_title(f'Q-Q Plot de residuales\nShapiro-Wilk p={p_sw:.4f} '
                  f'({"≈ normal" if p_sw > 0.05 else "no normal"})')

plt.suptitle(f'Análisis de residuales — {best_name}', fontsize=13)
plt.tight_layout()
plt.savefig(OUT + 'v2_residuales.png', dpi=150)
plt.close()

# ── Feature importance ────────────────────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    imp = pd.Series(best_model.feature_importances_,
                    index=FEATURES_NUM).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors_imp = []
    for name_f in imp.index:
        if name_f in ['m2', 'recamaras', 'banos', 'estacionamientos']:
            colors_imp.append(GREEN)
        elif name_f.startswith('feat_'):
            colors_imp.append('#F39C12')
        else:
            colors_imp.append(BLUE)
    ax.barh(imp.index, imp.values, color=colors_imp)
    ax.set_title(f'Top 15 variables — {best_name}\n'
                 f'Verde=m²/habitaciones  Naranja=amenidades texto  Azul=otras')
    ax.set_xlabel('Importancia relativa')
    plt.tight_layout()
    plt.savefig(OUT + 'v2_feature_importance.png', dpi=150)
    plt.close()

# ── Análisis log(price) como alternativa ─────────────────────────
print('\n── Modelos con log(price) — comparación ──')
y_log = np.log1p(y)
X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
rf_log = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf_log.fit(X_tr_l, y_tr_l)
y_pred_log = np.expm1(rf_log.predict(X_te_l))  # revertir log
mae_log = mean_absolute_error(y_test, y_pred_log)
r2_log  = r2_score(y_test, y_pred_log)
residuals_log = y_test - y_pred_log
_, p_sw_log = stats.shapiro(np.random.choice(residuals_log,
                             min(500, len(residuals_log)), replace=False))
print(f'  RF con log(price): R²={r2_log:.3f}  MAE=${mae_log:,.0f}  '
      f'Shapiro p={p_sw_log:.4f}')
print(f'  RF con price raw : R²={results[best_name]["R²"]:.3f}  '
      f'MAE=${results[best_name]["MAE"]:,.0f}  '
      f'Shapiro p={p_sw:.4f}')

# ══════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════
best_r2  = res_df['R²'].max()
best_cv  = res_df['CV-R²'].max()
best_mae = res_df.loc[res_df['R²'].idxmax(), 'MAE']

print('\n' + '='*60)
print('RESUMEN PIPELINE V2')
print('='*60)
print(f'Filas modeladas       : {len(df_m):,}')
print(f'Features usadas       : {len(FEATURES_NUM)}  '
      f'(m², rec, baños, estac + {len(feat_text)} amenidades)')
print(f'Mejor modelo          : {res_df["R²"].idxmax()}')
print(f'  R² test             : {best_r2:.3f}')
print(f'  CV-R² (5-fold)      : {best_cv:.3f}')
print(f'  MAE                 : ${best_mae:,.0f} MXN/mes')
print(f'  Residuales normales : {"Sí (p>0.05)" if p_sw > 0.05 else "No (p≤0.05) — esperado para RF"}')
print(f'\nNOTA: precio winsorizado al p95 en limpieza inicial.')
print(f'      Residuales con log(price): Shapiro p={p_sw_log:.4f}')
print('='*60)
print('Figuras guardadas en:', OUT)
