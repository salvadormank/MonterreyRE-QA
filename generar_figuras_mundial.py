"""
Genera todas las figuras para el reporte LaTeX del Mundial 2026
usando el CSV enriquecido con GPT-4o-mini.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
BLUE   = '#2C5F8A'
GREEN  = '#27AE60'
ORANGE = '#F39C12'
RED    = '#E74C3C'
OUT    = './figures/'

df = pd.read_csv('./leads_lamudi_enriched_openai.csv')
res = df[df['property_type'].isin(['departamento', 'casa'])].copy()

# ── w26_distribucion.png ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
price_res = res['price'].dropna()
price_clip = price_res[price_res <= price_res.quantile(0.95)]

axes[0].hist(price_clip, bins=40, color=BLUE, edgecolor='white')
axes[0].axvline(price_clip.median(), color=RED, linestyle='--', linewidth=2,
                label=f'Mediana: ${price_clip.median():,.0f}')
axes[0].set_title('Distribución — escala lineal')
axes[0].set_xlabel('Precio mensual (MXN)')
axes[0].set_ylabel('Frecuencia')
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}k'))
axes[0].legend()

log_price = np.log1p(price_clip)
axes[1].hist(log_price, bins=40, color='#5BA4CF', edgecolor='white', density=True)
xr = np.linspace(log_price.min(), log_price.max(), 200)
axes[1].plot(xr, stats.norm.pdf(xr, log_price.mean(), log_price.std()),
             'r-', linewidth=2, label='Normal teórica')
axes[1].set_title('Distribución — escala logarítmica')
axes[1].set_xlabel('log(Precio mensual)')
axes[1].set_ylabel('Densidad')
axes[1].legend()

plt.suptitle(f'Precio de renta residencial — Monterrey ZMM (n={len(price_clip):,})', fontsize=13)
plt.tight_layout()
plt.savefig(OUT + 'w26_distribucion.png', dpi=150)
plt.close()
print('✓ w26_distribucion.png')

# ── w26_tipos.png ────────────────────────────────────────────────────────
stats_tipo = (df.groupby('property_type')['price']
                .agg(['median', 'count'])
                .query('count >= 4')
                .sort_values('median'))

colors_t = [ORANGE if t in ('oficina','local_comercial','bodega_industrial','terreno','edificio')
            else BLUE for t in stats_tipo.index]

labels_map = {
    'departamento': 'Departamento',
    'casa': 'Casa',
    'oficina': 'Oficina',
    'local_comercial': 'Local comercial',
    'bodega_industrial': 'Bodega / Nave',
    'terreno': 'Terreno',
    'edificio': 'Edificio',
    'consultorio': 'Consultorio',
}

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh([labels_map.get(t, t) for t in stats_tipo.index],
               stats_tipo['median'], color=colors_t)
for bar, (_, row) in zip(bars, stats_tipo.iterrows()):
    ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
            f'n={int(row["count"])}   ${row["median"]/1000:.0f}k',
            va='center', fontsize=9)
ax.set_title('Precio mediano por tipo de propiedad\n(Naranja = comercial/industrial  |  Azul = residencial)',
             fontsize=12)
ax.set_xlabel('Precio mediano mensual (MXN)')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}k'))
ax.set_xlim(0, stats_tipo['median'].max() * 1.3)
plt.tight_layout()
plt.savefig(OUT + 'w26_tipos.png', dpi=150)
plt.close()
print('✓ w26_tipos.png')

# ── w26_top_colonias.png ─────────────────────────────────────────────────
top_col = (res.groupby('colonia')['price']
              .agg(['median', 'count'])
              .query('count >= 3')
              .sort_values('median', ascending=True)
              .tail(15))

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top_col.index, top_col['median'], color=BLUE)
ax.axvline(res['price'].median(), color=RED, linestyle='--', linewidth=1.5,
           label=f'Mediana ciudad: ${res["price"].median():,.0f}')
for bar, (_, row) in zip(bars, top_col.iterrows()):
    ax.text(bar.get_width() + 300, bar.get_y() + bar.get_height()/2,
            f'n={int(row["count"])}', va='center', fontsize=8)
ax.set_title('Top 15 colonias — Precio mediano residencial\n(mínimo 3 propiedades)', fontsize=12)
ax.set_xlabel('Precio mediano mensual (MXN)')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}k'))
ax.legend()
plt.tight_layout()
plt.savefig(OUT + 'w26_top_colonias.png', dpi=150)
plt.close()
print('✓ w26_top_colonias.png')

# ── w26_comparacion.png ──────────────────────────────────────────────────
# Dataset 2024 (valores del reporte anterior — Inmuebles24)
data_2024 = {'Departamento': 14000, 'Casa': 20000, 'Mercado\ngeneral': 23000}
data_2026 = {
    'Departamento': int(df[df['property_type']=='departamento']['price'].median()),
    'Casa': int(df[df['property_type']=='casa']['price'].median()),
    'Mercado\ngeneral': int(df['price'].median()),
}

x = np.arange(len(data_2024))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, data_2024.values(), width, label='2024 (Inmuebles24)',
               color='#AED6F1', edgecolor='white')
bars2 = ax.bar(x + width/2, data_2026.values(), width, label='2026 (Lamudi + GPT)',
               color=BLUE, edgecolor='white')

for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f'${bar.get_height()/1000:.0f}k', ha='center', va='bottom', fontsize=9, color=BLUE)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f'${bar.get_height()/1000:.0f}k', ha='center', va='bottom', fontsize=9, color='#2C5F8A')

# Flechas de variación
for i, (k24, k26) in enumerate(zip(data_2024.values(), data_2026.values())):
    pct = (k26 - k24) / k24 * 100
    ax.annotate(f'+{pct:.0f}%', xy=(x[i] + width/2, k26 + 1500),
                fontsize=9, color=RED, ha='center', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(data_2024.keys())
ax.set_ylabel('Precio mediano mensual (MXN)')
ax.set_title('Comparativa de precios medianos 2024 → 2026', fontsize=13)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}k'))
ax.legend()
plt.tight_layout()
plt.savefig(OUT + 'w26_comparacion.png', dpi=150)
plt.close()
print('✓ w26_comparacion.png')

# ── w26_mundial.png ──────────────────────────────────────────────────────
bbva_colonias = ['Santa Maria', 'San Jerónimo', 'Colinas de San Jerónimo',
                 'Miravalle', 'Del Paseo Residencial', 'La Estanzuela',
                 'Cumbres', 'Las Brisas']

bbva_stats = (df[df['colonia'].isin(bbva_colonias)]
              .groupby('colonia')['price']
              .agg(['median', 'count'])
              .query('count >= 5')
              .sort_values('median'))

ciudad_median = df['price'].median()

fig, ax = plt.subplots(figsize=(10, 5))
colors_b = [GREEN if v > ciudad_median else '#AED6F1' for v in bbva_stats['median']]
bars = ax.barh(bbva_stats.index, bbva_stats['median'], color=colors_b)
ax.axvline(ciudad_median, color=RED, linestyle='--', linewidth=2,
           label=f'Mediana ciudad: ${ciudad_median:,.0f}')
for bar, (_, row) in zip(bars, bbva_stats.iterrows()):
    ax.text(bar.get_width() + 300, bar.get_y() + bar.get_height()/2,
            f'n={int(row["count"])}   ${row["median"]/1000:.0f}k',
            va='center', fontsize=9)
ax.set_title('Colonias zona Estadio BBVA — Precio mediano de renta\n(Verde = por encima de la mediana ciudad)', fontsize=12)
ax.set_xlabel('Precio mediano mensual (MXN)')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}k'))
ax.set_xlim(0, bbva_stats['median'].max() * 1.35)
ax.legend()
plt.tight_layout()
plt.savefig(OUT + 'w26_mundial.png', dpi=150)
plt.close()
print('✓ w26_mundial.png')

# ── w26_amenidades.png (nueva figura) ────────────────────────────────────
amenidades = {
    'Seguridad 24h':   'has_security_24h',
    'Minisplit/AC':    'has_minisplit_ac',
    'Terraza':         'has_terrace',
    'Elevador':        'has_elevator',
    'Alberca':         'has_pool',
    'Gimnasio':        'has_gym',
    'Sala de juntas':  'has_sala_juntas',
    'Amueblado':       'furnished',
    'Bodega util.':    'has_bodega',
    'Lobby':           'has_lobby',
    'Coworking':       'has_coworking',
    'Jardín':          'has_garden',
}
pcts_res = {k: (res[v] == 1).sum() / len(res) * 100 for k,v in amenidades.items()}
pcts_com = {k: (df[df['property_type'].isin(['oficina','local_comercial'])][v] == 1).sum() /
               len(df[df['property_type'].isin(['oficina','local_comercial'])]) * 100
            for k,v in amenidades.items()}

x = np.arange(len(amenidades))
width = 0.38
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - width/2, list(pcts_res.values()), width, label='Residencial', color=BLUE)
ax.bar(x + width/2, list(pcts_com.values()), width, label='Comercial', color=ORANGE)
ax.set_xticks(x)
ax.set_xticklabels(list(amenidades.keys()), rotation=35, ha='right')
ax.set_ylabel('% de propiedades')
ax.set_title('Frecuencia de amenidades — Residencial vs Comercial', fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig(OUT + 'w26_amenidades.png', dpi=150)
plt.close()
print('✓ w26_amenidades.png')

print('\nTodas las figuras generadas.')
