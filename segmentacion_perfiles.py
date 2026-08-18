import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.1)

BLUE   = '#2C5F8A'
GREEN  = '#27AE60'
ORANGE = '#E67E22'
GRAY   = '#95A5A6'
LIGHT  = '#EBF5FB'
OUT    = '../analisis_propiedades/figures/'

print('Cargando datos...')
df = pd.read_csv('../propiedades_clean.csv')
texts = (df['description'].fillna('') + ' ' + df['longDescription'].fillna('')).str.lower()

# ══════════════════════════════════════════════════════════════════
# DEFINICIÓN DE SEGMENTOS
# ══════════════════════════════════════════════════════════════════

# Estudiantes: mención explícita de contexto universitario
kw_student = r'estudiante|universitari|tec |tecnológico|tecnologico|itesm|uanl|udem|campus'
mask_student = texts.str.contains(kw_student, na=False)

# Adultos mayores: proxy por accesibilidad + seguridad + tranquilidad
# (el dataset no los menciona directamente; solo 1 propiedad dice "adulto mayor")
kw_access   = r'elevador|ascensor|planta baja|sin escalera|rampa|accesib'
kw_security = r'seguridad 24|vigilancia 24|portero|guardia'
kw_quiet    = r'tranquilo|tranquila|silencioso|silenciosa'
mask_senior = (
    texts.str.contains(kw_access,   na=False) |
    texts.str.contains(kw_security, na=False) |
    texts.str.contains(kw_quiet,    na=False)
)

# Etiqueta de segmento (pueden solaparse; priorizamos estudiante si hay overlap)
df['segmento'] = 'Mercado general'
df.loc[mask_senior,  'segmento'] = 'Adulto mayor (proxy)'
df.loc[mask_student, 'segmento'] = 'Estudiantes'

n_s  = mask_student.sum()
n_am = mask_senior.sum()
n_ov = (mask_student & mask_senior).sum()
n_g  = len(df) - mask_student.sum() - (mask_senior & ~mask_student).sum()

print(f'Estudiantes             : {n_s} propiedades')
print(f'Adultos mayores (proxy) : {n_am} propiedades ({n_am - n_ov} exclusivas)')
print(f'Overlap                 : {n_ov} propiedades')
print(f'Mercado general         : {n_g} propiedades\n')

seg_estudiante = df[mask_student]
seg_mayor      = df[mask_senior & ~mask_student]
seg_general    = df[~mask_student & ~mask_senior]

# ══════════════════════════════════════════════════════════════════
# FIGURA 1: DISTRIBUCIÓN DE PRECIO POR SEGMENTO
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograma superpuesto
for data, label, color in [
    (seg_estudiante['price'], 'Estudiantes',            BLUE),
    (seg_mayor['price'],      'Adulto mayor (proxy)',   GREEN),
    (seg_general['price'],    'Mercado general',        GRAY),
]:
    axes[0].hist(data, bins=30, alpha=0.55, label=label, color=color, edgecolor='white')

axes[0].axvline(seg_estudiante['price'].median(), color=BLUE,  linestyle='--', linewidth=1.5)
axes[0].axvline(seg_mayor['price'].median(),      color=GREEN, linestyle='--', linewidth=1.5)
axes[0].set_title('Distribución de precio por segmento')
axes[0].set_xlabel('Precio mensual (MXN)')
axes[0].set_ylabel('Frecuencia')
axes[0].legend()

# Boxplot comparativo
data_box   = [seg_estudiante['price'], seg_mayor['price'], seg_general['price']]
labels_box = ['Estudiantes', 'Adulto mayor\n(proxy)', 'Mercado\ngeneral']
colors_box = [BLUE, GREEN, GRAY]
bp = axes[1].boxplot(data_box, labels=labels_box, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_title('Boxplot de precio por segmento')
axes[1].set_ylabel('Precio mensual (MXN)')

# Anotaciones de mediana
medians = [seg_estudiante['price'].median(),
           seg_mayor['price'].median(),
           seg_general['price'].median()]
for i, med in enumerate(medians):
    axes[1].text(i + 1, med + 400, f'${med:,.0f}', ha='center', fontsize=9, color='black')

plt.tight_layout()
plt.savefig(OUT + 'S1_precio_por_segmento.png', dpi=150)
plt.close()
print('✓ S1 — Distribución de precio')

# ══════════════════════════════════════════════════════════════════
# FIGURA 2: ATRIBUTOS BINARIOS POR SEGMENTO
# ══════════════════════════════════════════════════════════════════
attrs = ['Amueblado', 'Lujo', 'Nuevo']
seg_labels  = ['Estudiantes', 'Adulto mayor (proxy)', 'Mercado general']
seg_data    = [seg_estudiante, seg_mayor, seg_general]
seg_colors  = [BLUE, GREEN, GRAY]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
x = np.arange(len(seg_labels))
width = 0.6

for ax, attr in zip(axes, attrs):
    vals = [d[attr].mean() * 100 for d in seg_data]
    bars = ax.bar(seg_labels, vals, color=seg_colors, alpha=0.8, width=width, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_title(f'% {attr}')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Porcentaje (%)')
    ax.tick_params(axis='x', labelrotation=15)

plt.suptitle('Atributos binarios por segmento', fontsize=13)
plt.tight_layout()
plt.savefig(OUT + 'S2_atributos_por_segmento.png', dpi=150)
plt.close()
print('✓ S2 — Atributos binarios')

# ══════════════════════════════════════════════════════════════════
# FIGURA 3: TOP ZONAS POR SEGMENTO
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for ax, seg, label, color in [
    (axes[0], seg_estudiante, 'Estudiantes',          BLUE),
    (axes[1], seg_mayor,      'Adulto mayor (proxy)', GREEN),
]:
    top = seg['location'].value_counts().head(10)
    ax.barh(top.index[::-1], top.values[::-1], color=color, alpha=0.8, edgecolor='white')
    ax.set_title(f'Top 10 zonas — {label}')
    ax.set_xlabel('Número de propiedades')
    for i, (loc, val) in enumerate(zip(top.index[::-1], top.values[::-1])):
        ax.text(val + 0.3, i, str(val), va='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUT + 'S3_zonas_por_segmento.png', dpi=150)
plt.close()
print('✓ S3 — Zonas por segmento')

# ══════════════════════════════════════════════════════════════════
# FIGURA 4: MAPA GEOGRÁFICO POR SEGMENTO
# ══════════════════════════════════════════════════════════════════
geo_s = seg_estudiante.dropna(subset=['lat', 'lon'])
geo_a = seg_mayor.dropna(subset=['lat', 'lon'])
geo_g = seg_general.dropna(subset=['lat', 'lon'])

fig, ax = plt.subplots(figsize=(11, 9))

ax.scatter(geo_g['lon'],  geo_g['lat'],  c=GRAY,   s=18, alpha=0.4, label='Mercado general',        zorder=1)
ax.scatter(geo_a['lon'],  geo_a['lat'],  c=GREEN,  s=30, alpha=0.7, label='Adulto mayor (proxy)',   zorder=2)
ax.scatter(geo_s['lon'],  geo_s['lat'],  c=BLUE,   s=30, alpha=0.7, label='Estudiantes',            zorder=3)

ax.set_title('Distribución geográfica por segmento\n(lat/lon — propiedades con coordenadas)', fontsize=13)
ax.set_xlabel('Longitud')
ax.set_ylabel('Latitud')
ax.legend(markerscale=1.5, framealpha=0.9)
plt.tight_layout()
plt.savefig(OUT + 'S4_mapa_segmentos.png', dpi=150)
plt.close()
print('✓ S4 — Mapa geográfico')

# ══════════════════════════════════════════════════════════════════
# FIGURA 5: SEÑALES DE TEXTO — FRECUENCIA DE KEYWORDS
# ══════════════════════════════════════════════════════════════════
kw_estudiante_list = [
    ('estudiante', r'estudiante'),
    ('Tec / Tecnológico', r'tec |tecnológico|tecnologico'),
    ('UANL', r'uanl'),
    ('ITESM', r'itesm'),
    ('UDEM', r'udem'),
    ('universitario/a', r'universitari'),
    ('comparte / roomie', r'comparte'),
    ('coworking', r'coworking'),
    ('cuarto', r'cuarto'),
]
kw_senior_list = [
    ('elevador / ascensor', r'elevador|ascensor'),
    ('planta baja', r'planta baja'),
    ('seguridad 24h', r'seguridad 24|vigilancia 24'),
    ('tranquilo/a', r'tranquilo|tranquila'),
    ('guardia / portero', r'guardia|portero'),
    ('accesible / rampa', r'accesib|rampa'),
    ('silencioso/a', r'silencioso|silenciosa'),
]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Estudiantes
labels_s = [k[0] for k in kw_estudiante_list]
counts_s = [texts.str.contains(k[1], na=False).sum() for k in kw_estudiante_list]
sorted_s = sorted(zip(counts_s, labels_s), reverse=True)
axes[0].barh([l for _, l in sorted_s], [c for c, _ in sorted_s],
             color=BLUE, alpha=0.8, edgecolor='white')
axes[0].set_title('Señales textuales — Estudiantes')
axes[0].set_xlabel('Número de propiedades')
for i, (c, _) in enumerate(sorted_s):
    axes[0].text(c + 1, i, str(c), va='center', fontsize=9)

# Adultos mayores
labels_a = [k[0] for k in kw_senior_list]
counts_a = [texts.str.contains(k[1], na=False).sum() for k in kw_senior_list]
sorted_a = sorted(zip(counts_a, labels_a), reverse=True)
axes[1].barh([l for _, l in sorted_a], [c for c, _ in sorted_a],
             color=GREEN, alpha=0.8, edgecolor='white')
axes[1].set_title('Señales textuales — Adulto mayor (proxy)')
axes[1].set_xlabel('Número de propiedades')
for i, (c, _) in enumerate(sorted_a):
    axes[1].text(c + 1, i, str(c), va='center', fontsize=9)

plt.suptitle('Frecuencia de keywords usados para segmentación', fontsize=13)
plt.tight_layout()
plt.savefig(OUT + 'S5_keywords_segmentos.png', dpi=150)
plt.close()
print('✓ S5 — Keywords de segmentación')

# ══════════════════════════════════════════════════════════════════
# FIGURA 6: PRECIO VS userViews POR SEGMENTO
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
for seg, label, color, marker in [
    (seg_general,    'Mercado general',       GRAY,   'o'),
    (seg_mayor,      'Adulto mayor (proxy)',  GREEN,  's'),
    (seg_estudiante, 'Estudiantes',           BLUE,   '^'),
]:
    valid = seg.dropna(subset=['userViews'])
    ax.scatter(valid['userViews'], valid['price'],
               c=color, alpha=0.5, s=25, marker=marker, label=label)

ax.set_xlabel('Vistas del anuncio')
ax.set_ylabel('Precio mensual (MXN)')
ax.set_title('Interés (vistas) vs Precio por segmento')
ax.legend()
plt.tight_layout()
plt.savefig(OUT + 'S6_views_precio_segmento.png', dpi=150)
plt.close()
print('✓ S6 — Views vs precio')

# ══════════════════════════════════════════════════════════════════
# FIGURA 7: TABLA RESUMEN VISUAL
# ══════════════════════════════════════════════════════════════════
summary = {
    'Métrica': [
        'N propiedades', 'Precio mediano (MXN)', 'Precio mínimo (MXN)',
        'Precio máximo (MXN)', '% Amueblado', '% Lujo', '% Nuevo',
        'Base de identificación'
    ],
    'Estudiantes': [
        n_s,
        f"${seg_estudiante['price'].median():,.0f}",
        f"${seg_estudiante['price'].min():,.0f}",
        f"${seg_estudiante['price'].max():,.0f}",
        f"{seg_estudiante['Amueblado'].mean()*100:.0f}%",
        f"{seg_estudiante['Lujo'].mean()*100:.0f}%",
        f"{seg_estudiante['Nuevo'].mean()*100:.0f}%",
        'Texto explícito'
    ],
    'Adulto mayor (proxy)': [
        n_am - n_ov,
        f"${seg_mayor['price'].median():,.0f}",
        f"${seg_mayor['price'].min():,.0f}",
        f"${seg_mayor['price'].max():,.0f}",
        f"{seg_mayor['Amueblado'].mean()*100:.0f}%",
        f"{seg_mayor['Lujo'].mean()*100:.0f}%",
        f"{seg_mayor['Nuevo'].mean()*100:.0f}%",
        'Proxy (accesib. + seguridad)'
    ],
    'Mercado general': [
        n_g,
        f"${seg_general['price'].median():,.0f}",
        f"${seg_general['price'].min():,.0f}",
        f"${seg_general['price'].max():,.0f}",
        f"{seg_general['Amueblado'].mean()*100:.0f}%",
        f"{seg_general['Lujo'].mean()*100:.0f}%",
        f"{seg_general['Nuevo'].mean()*100:.0f}%",
        '—'
    ],
}
df_summary = pd.DataFrame(summary)

fig, ax = plt.subplots(figsize=(13, 4))
ax.axis('off')
tbl = ax.table(
    cellText=df_summary.values,
    colLabels=df_summary.columns,
    cellLoc='center', loc='center',
    colColours=['#F4F6F8', BLUE, GREEN, GRAY]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.7)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_text_props(color='white', fontweight='bold')
    if col == 0 and row > 0:
        cell.set_facecolor('#F4F6F8')
        cell.set_text_props(fontweight='bold')

plt.title('Tabla comparativa de segmentos', pad=15, fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'S7_tabla_resumen.png', dpi=150, bbox_inches='tight')
plt.close()
print('✓ S7 — Tabla resumen')

# ══════════════════════════════════════════════════════════════════
# RESUMEN CONSOLA
# ══════════════════════════════════════════════════════════════════
print('\n' + '='*55)
print('SEGMENTACIÓN — RESUMEN')
print('='*55)
print(f'Estudiantes         : {n_s} props | mediana ${seg_estudiante["price"].median():,.0f} | {seg_estudiante["Amueblado"].mean()*100:.0f}% amueblado')
print(f'Adulto mayor(proxy) : {n_am-n_ov} props | mediana ${seg_mayor["price"].median():,.0f} | {seg_mayor["Amueblado"].mean()*100:.0f}% amueblado')
print(f'Mercado general     : {n_g} props | mediana ${seg_general["price"].median():,.0f}')
print(f'Overlap ambos       : {n_ov} props')
print('='*55)
print('✓ Todas las figuras S1–S7 guardadas.')
