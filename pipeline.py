import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import warnings
import re
from collections import Counter

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.decomposition import PCA
from scipy import stats
from wordcloud import WordCloud
import folium
from folium.plugins import HeatMap

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
BLUE = '#2C5F8A'
OUT = '../analisis_propiedades/figures/'

print('Cargando datos...')
df = pd.read_csv('../propiedades_clean.csv')
print(f'  {df.shape[0]} registros, {df.shape[1]} columnas\n')

# ══════════════════════════════════════════════════════════════════
# 1. EDA
# ══════════════════════════════════════════════════════════════════
print('── 1. EDA ──')

# 1a. Distribución de precio
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['price'], bins=40, color=BLUE, edgecolor='white')
axes[0].set_title('Distribución de precio de renta')
axes[0].set_xlabel('Precio mensual (MXN)')
axes[0].set_ylabel('Frecuencia')

axes[1].hist(np.log1p(df['price']), bins=40, color='#5BA4CF', edgecolor='white')
axes[1].set_title('Distribución log(precio)')
axes[1].set_xlabel('log(Precio)')
axes[1].set_ylabel('Frecuencia')

plt.tight_layout()
plt.savefig(OUT + '1a_distribucion_precio.png', dpi=150)
plt.close()

# 1b. Precio por atributos binarios
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, col, label in zip(axes, ['Amueblado', 'Lujo', 'Nuevo'],
                           ['Amueblado', 'Lujo', 'Nuevo']):
    groups = [df[df[col] == 0]['price'], df[df[col] == 1]['price']]
    ax.boxplot(groups, labels=['No', 'Sí'], patch_artist=True,
               boxprops=dict(facecolor='#AED6F1'),
               medianprops=dict(color=BLUE, linewidth=2))
    ax.set_title(f'Precio por {label}')
    ax.set_ylabel('Precio (MXN)')
    t, p = stats.ttest_ind(groups[0].dropna(), groups[1].dropna())
    ax.set_xlabel(f'p-value = {p:.3f}')

plt.tight_layout()
plt.savefig(OUT + '1b_precio_por_atributos.png', dpi=150)
plt.close()

# 1c. Top 15 zonas por precio mediano
top_loc = (df.groupby('location')['price']
             .agg(['median', 'count'])
             .query('count >= 5')
             .sort_values('median', ascending=True)
             .tail(15))

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top_loc.index, top_loc['median'], color=BLUE)
for bar, (_, row) in zip(bars, top_loc.iterrows()):
    ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
            f"n={int(row['count'])}", va='center', fontsize=8)
ax.set_title('Top 15 zonas por precio mediano de renta')
ax.set_xlabel('Precio mediano (MXN)')
plt.tight_layout()
plt.savefig(OUT + '1c_top_zonas.png', dpi=150)
plt.close()

# 1d. Correlación numérica
num_cols = ['price', 'userViews', 'publishedSince', 'Nuevo', 'Amueblado', 'Lujo', 'lat', 'lon']
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            linewidths=0.5, ax=ax)
ax.set_title('Matriz de correlación')
plt.tight_layout()
plt.savefig(OUT + '1d_correlacion.png', dpi=150)
plt.close()

# 1e. userViews vs precio
fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(df['userViews'].dropna(), df.loc[df['userViews'].notna(), 'price'],
                     alpha=0.4, c=BLUE, edgecolors='white', linewidth=0.3)
ax.set_xlabel('Vistas del anuncio (capeadas)')
ax.set_ylabel('Precio (MXN)')
ax.set_title('Relación entre vistas y precio')
plt.tight_layout()
plt.savefig(OUT + '1e_views_vs_precio.png', dpi=150)
plt.close()

print('  ✓ Figuras EDA generadas')

# ══════════════════════════════════════════════════════════════════
# 2. ANÁLISIS GEOGRÁFICO
# ══════════════════════════════════════════════════════════════════
print('── 2. Análisis geográfico ──')

geo = df.dropna(subset=['lat', 'lon']).copy()

# 2a. Scatter geográfico coloreado por precio
fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(geo['lon'], geo['lat'],
                c=geo['price'], cmap='YlOrRd',
                s=25, alpha=0.7, edgecolors='none')
plt.colorbar(sc, ax=ax, label='Precio mensual (MXN)')
ax.set_title(f'Distribución geográfica de precios\n(n={len(geo)} propiedades con coords)')
ax.set_xlabel('Longitud')
ax.set_ylabel('Latitud')
plt.tight_layout()
plt.savefig(OUT + '2a_mapa_precios.png', dpi=150)
plt.close()

# 2b. Mapa interactivo Folium
m = folium.Map(location=[geo['lat'].mean(), geo['lon'].mean()], zoom_start=12)
heat_data = [[row['lat'], row['lon'], row['price']] for _, row in geo.iterrows()]
HeatMap(heat_data, radius=15, blur=20, min_opacity=0.3).add_to(m)

for _, row in geo.sample(min(200, len(geo)), random_state=42).iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=4,
        color='#2C5F8A',
        fill=True,
        fill_opacity=0.6,
        popup=f"${row['price']:,.0f}/mes | {row.get('location','')}"
    ).add_to(m)

map_path = '../analisis_propiedades/mapa_interactivo.html'
m.save(map_path)
print(f'  ✓ Mapa interactivo: {map_path}')
print('  ✓ Scatter geográfico generado')

# ══════════════════════════════════════════════════════════════════
# 3. IMPORTANCIA DE FEATURES (Random Forest)
# ══════════════════════════════════════════════════════════════════
print('── 3. Feature importance ──')

df_model = df.copy()

# Encoding de location
le = LabelEncoder()
df_model['location_enc'] = le.fit_transform(df_model['location'].fillna('Desconocido'))

features = ['Amueblado', 'Lujo', 'Nuevo', 'location_enc', 'publishedSince', 'userViews', 'lat', 'lon']
target = 'price'

mask = df_model[features + [target]].notna().all(axis=1)
X = df_model.loc[mask, features]
y = df_model.loc[mask, target]

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=features).sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#AED6F1' if i < len(importances)-3 else BLUE for i in range(len(importances))]
ax.barh(importances.index, importances.values, color=colors)
ax.set_title('Importancia de variables (Random Forest)')
ax.set_xlabel('Importancia relativa')
plt.tight_layout()
plt.savefig(OUT + '3_feature_importance.png', dpi=150)
plt.close()
print('  ✓ Feature importance calculada')

# ══════════════════════════════════════════════════════════════════
# 4. CLUSTERING (K-Means)
# ══════════════════════════════════════════════════════════════════
print('── 4. Clustering ──')

cluster_features = ['price', 'Amueblado', 'Lujo', 'Nuevo', 'userViews', 'publishedSince']
df_cl = df[cluster_features].dropna().copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cl)

# Método del codo
inertias = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(K_range, inertias, 'o-', color=BLUE, linewidth=2, markersize=7)
ax.set_title('Método del codo para K óptimo')
ax.set_xlabel('Número de clusters (K)')
ax.set_ylabel('Inercia')
plt.tight_layout()
plt.savefig(OUT + '4a_codo.png', dpi=150)
plt.close()

# Ajustar con K=4
K_OPT = 4
km_final = KMeans(n_clusters=K_OPT, random_state=42, n_init=10)
df_cl['cluster'] = km_final.fit_predict(X_scaled)

# PCA 2D para visualizar
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)
df_cl['PC1'] = coords[:, 0]
df_cl['PC2'] = coords[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_cl = ['#2C5F8A', '#E74C3C', '#27AE60', '#F39C12']

for k in range(K_OPT):
    mask_k = df_cl['cluster'] == k
    axes[0].scatter(df_cl.loc[mask_k, 'PC1'], df_cl.loc[mask_k, 'PC2'],
                    label=f'Cluster {k}', color=colors_cl[k], alpha=0.6, s=20)
axes[0].set_title('Clusters en espacio PCA')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].legend()

summary = df_cl.groupby('cluster')[cluster_features].mean().round(0)
summary_T = summary.T
im = axes[1].imshow(summary_T.values, cmap='YlOrRd', aspect='auto')
axes[1].set_xticks(range(K_OPT))
axes[1].set_xticklabels([f'C{k}' for k in range(K_OPT)])
axes[1].set_yticks(range(len(cluster_features)))
axes[1].set_yticklabels(cluster_features)
for i in range(len(cluster_features)):
    for j in range(K_OPT):
        axes[1].text(j, i, f'{summary_T.iloc[i, j]:.0f}', ha='center', va='center', fontsize=9)
axes[1].set_title('Perfil promedio por cluster')
plt.colorbar(im, ax=axes[1])
plt.tight_layout()
plt.savefig(OUT + '4b_clusters.png', dpi=150)
plt.close()

print('  Perfiles de clusters:')
for k in range(K_OPT):
    row = summary.loc[k]
    tag = []
    if row['price'] > df['price'].median() * 1.2: tag.append('precio alto')
    else: tag.append('precio bajo')
    if row['Lujo'] > 0.5: tag.append('lujo')
    if row['Amueblado'] > 0.5: tag.append('amueblado')
    if row['Nuevo'] > 0.5: tag.append('nuevo')
    n = (df_cl['cluster'] == k).sum()
    print(f'  Cluster {k}: ${row["price"]:,.0f}/mes | {", ".join(tag)} | n={n}')

print('  ✓ Clustering completado')

# ══════════════════════════════════════════════════════════════════
# 5. ANÁLISIS DE TEXTO
# ══════════════════════════════════════════════════════════════════
print('── 5. Análisis de texto ──')

AMENIDADES = [
    'alberca', 'gimnasio', 'estacionamiento', 'balcón', 'terraza',
    'jacuzzi', 'pet', 'amueblado', 'bodega', 'vigilancia', 'seguridad',
    'coworking', 'rooftop', 'asador', 'lavadora', 'secadora',
    'aire acondicionado', 'refrigerador', 'microondas', 'elevador',
    'portero', 'cisterna', 'gas', 'internet', 'cancha', 'pádel',
]

def extract_amenidades(text):
    if pd.isna(text):
        return []
    text = text.lower()
    return [a for a in AMENIDADES if a in text]

all_texts = df['description'].fillna('') + ' ' + df['longDescription'].fillna('')
found = all_texts.apply(extract_amenidades)
counts = Counter([a for sublist in found for a in sublist])

# 5a. Barplot de amenidades
amenity_df = pd.DataFrame(counts.most_common(20), columns=['amenidad', 'menciones'])
fig, ax = plt.subplots(figsize=(10, 6))
palette = [BLUE if i < 10 else '#AED6F1' for i in range(len(amenity_df))]
ax.barh(amenity_df['amenidad'][::-1], amenity_df['menciones'][::-1], color=palette[::-1])
ax.set_title('Amenidades más frecuentes en descripciones')
ax.set_xlabel('Número de menciones')
plt.tight_layout()
plt.savefig(OUT + '5a_amenidades.png', dpi=150)
plt.close()

# 5b. Precio promedio por presencia de amenidad (top 10)
top10 = [a for a, _ in counts.most_common(10)]
amenity_price = {}
for amenidad in top10:
    mask_yes = all_texts.str.lower().str.contains(amenidad, na=False)
    price_yes = df.loc[mask_yes, 'price'].mean()
    price_no  = df.loc[~mask_yes, 'price'].mean()
    amenity_price[amenidad] = {'con': price_yes, 'sin': price_no}

ap_df = pd.DataFrame(amenity_price).T.sort_values('con', ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(ap_df))
ax.barh([i - 0.2 for i in x], ap_df['con'], height=0.4, label='Con amenidad', color=BLUE)
ax.barh([i + 0.2 for i in x], ap_df['sin'], height=0.4, label='Sin amenidad', color='#AED6F1')
ax.set_yticks(list(x))
ax.set_yticklabels(ap_df.index)
ax.set_xlabel('Precio promedio (MXN)')
ax.set_title('Precio promedio según presencia de amenidad')
ax.legend()
plt.tight_layout()
plt.savefig(OUT + '5b_amenidad_vs_precio.png', dpi=150)
plt.close()

# 5c. WordCloud
text_corpus = ' '.join(all_texts.dropna().tolist()).lower()
text_corpus = re.sub(r'[^a-záéíóúüñ\s]', ' ', text_corpus)
stopwords_es = {
    'de','la','el','en','y','a','los','las','con','por','para','se',
    'que','un','una','es','su','del','al','lo','más','son','como',
    'todo','esta','este','entre','cada','muy','hay','sus','nos',
    'también','sin','sobre','está','tiene','pero','si','no','le',
    'ya','así','ser','fue','ser','puede','han','ha','o','e',
}
wc = WordCloud(
    width=900, height=450, background_color='white',
    colormap='Blues', stopwords=stopwords_es,
    max_words=120, collocations=False
).generate(text_corpus)

fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(wc, interpolation='bilinear')
ax.axis('off')
ax.set_title('WordCloud — términos más frecuentes en descripciones', fontsize=14)
plt.tight_layout()
plt.savefig(OUT + '5c_wordcloud.png', dpi=150)
plt.close()

print('  ✓ Análisis de texto completado')

# ══════════════════════════════════════════════════════════════════
# 6. MODELO PREDICTIVO DE PRECIO
# ══════════════════════════════════════════════════════════════════
print('── 6. Modelo predictivo ──')

# Agregar features de amenidades como columnas binarias
for amenidad in top10:
    col = 'feat_' + amenidad.replace(' ', '_')
    df_model[col] = all_texts.str.lower().str.contains(amenidad, na=False).astype(int)

feat_text = ['feat_' + a.replace(' ', '_') for a in top10]
all_features = features + feat_text
mask2 = df_model[all_features + [target]].notna().all(axis=1)
X2 = df_model.loc[mask2, all_features]
y2 = df_model.loc[mask2, target]

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X2, y2, cv=5, scoring='r2').mean()
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'CV_R2': cv}
    print(f'  {name}: MAE=${mae:,.0f} | RMSE=${rmse:,.0f} | R²={r2:.3f} | CV-R²={cv:.3f}')

res_df = pd.DataFrame(results).T

# 6a. Comparación de modelos
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, metric in zip(axes, ['MAE', 'RMSE', 'R2']):
    vals = res_df[metric].sort_values()
    ax.barh(vals.index, vals.values,
            color=[BLUE if v == vals.max() else '#AED6F1' for v in vals.values]
                  if metric == 'R2'
                  else [BLUE if v == vals.min() else '#AED6F1' for v in vals.values])
    ax.set_title(metric)
    ax.set_xlabel(metric)
plt.suptitle('Comparación de modelos predictivos', fontsize=13)
plt.tight_layout()
plt.savefig(OUT + '6a_modelos.png', dpi=150)
plt.close()

# 6b. Real vs predicho (mejor modelo = RF)
best_model = models['Random Forest']
y_pred_best = best_model.predict(X_test)
residuals = y_test - y_pred_best

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(y_test, y_pred_best, alpha=0.5, color=BLUE, s=20)
lim = max(y_test.max(), y_pred_best.max())
axes[0].plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='Predicción perfecta')
axes[0].set_xlabel('Precio real (MXN)')
axes[0].set_ylabel('Precio predicho (MXN)')
axes[0].set_title(f'Real vs Predicho — Random Forest\nR²={r2_score(y_test, y_pred_best):.3f}')
axes[0].legend()

axes[1].hist(residuals, bins=35, color=BLUE, edgecolor='white')
axes[1].axvline(0, color='red', linewidth=1.5, linestyle='--')
axes[1].set_xlabel('Residual (MXN)')
axes[1].set_ylabel('Frecuencia')
axes[1].set_title('Distribución de residuales')
plt.tight_layout()
plt.savefig(OUT + '6b_real_vs_predicho.png', dpi=150)
plt.close()

# 6c. Feature importance modelo final
imp_all = pd.Series(best_model.feature_importances_, index=all_features).sort_values().tail(15)
fig, ax = plt.subplots(figsize=(9, 6))
colors_imp = ['#AED6F1' if 'feat_' not in i else '#27AE60' for i in imp_all.index]
ax.barh(imp_all.index, imp_all.values, color=colors_imp)
ax.set_title('Top 15 variables más importantes (Random Forest)\nAzul=estructural | Verde=amenidad')
ax.set_xlabel('Importancia relativa')
plt.tight_layout()
plt.savefig(OUT + '6c_importance_final.png', dpi=150)
plt.close()

print('  ✓ Modelo predictivo completado')

# ══════════════════════════════════════════════════════════════════
# RESUMEN EJECUTIVO
# ══════════════════════════════════════════════════════════════════
best_r2 = res_df['R2'].max()
best_name = res_df['R2'].idxmax()
best_mae = res_df.loc[best_name, 'MAE']

print('\n' + '='*55)
print('RESUMEN EJECUTIVO')
print('='*55)
print(f'Propiedades analizadas : {len(df):,}')
print(f'Precio mediano renta   : ${df["price"].median():,.0f} MXN/mes')
print(f'Zona más cara          : {df.groupby("location")["price"].median().idxmax()}')
print(f'Amenidad más frecuente : {counts.most_common(1)[0][0]} ({counts.most_common(1)[0][1]} menciones)')
print(f'Clusters identificados : {K_OPT}')
print(f'Mejor modelo           : {best_name} (R²={best_r2:.3f}, MAE=${best_mae:,.0f})')
print('='*55)
print('\n✓ Pipeline completo. Figuras en:', OUT)
