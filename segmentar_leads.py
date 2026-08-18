"""
Segmentación de leads Lamudi para dos perfiles:
  - Estudiantes:    presupuesto bajo, 1-2 recámaras, keywords universitarios
  - Tercera edad:   accesibilidad, tranquilidad, seguridad, presupuesto moderado

Salida: leads_segmentados.csv  (incluye columna 'segmento')
"""

import pandas as pd
import re

CSV_IN  = './leads_lamudi_renta_enriched.csv'
CSV_OUT = './leads_segmentados.csv'

# ─── Carga ────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_IN)

# Normalizar property_type
df['property_type'] = df['property_type'].str.strip().str.lower()

# Texto combinado para búsqueda de keywords
texto = (df['description'].fillna('') + ' ' + df['address'].fillna('') +
         ' ' + df['colonia'].fillna('')).str.lower()

# ─── Filtro: solo propiedades residenciales ───────────────────────────────────
residenciales = {'casa', 'departamento', 'loft', 'cuarto'}
mask_res = df['property_type'].isin(residenciales)

# ─── Criterios Estudiantes ────────────────────────────────────────────────────
# Presupuesto: ≤ $18,000 MXN/mes
PRECIO_MAX_EST = 18_000

kw_est = (
    r'estudiante|universitari|tec\b|tecnol[oó]gico|itesm|uanl|udem|campus|'
    r'roomie|comparte\s+cuarto|cuarto\s+en\s+renta|coworking|zona\s+estudiantil|'
    r'cerca.*universidad|universidad.*cerca'
)

mask_precio_est  = df['price'] <= PRECIO_MAX_EST
mask_rec_est     = df['bedrooms'].fillna(0).isin([0, 1, 2])
mask_kw_est      = texto.str.contains(kw_est, na=False)

# Score compuesto: precio bajo + (1-2 recámaras OR keyword universitario)
mask_estudiante = mask_res & mask_precio_est & (mask_rec_est | mask_kw_est)

# ─── Criterios Tercera Edad ───────────────────────────────────────────────────
# Presupuesto: $8,000 – $35,000 MXN/mes
PRECIO_MIN_AM = 8_000
PRECIO_MAX_AM = 35_000

kw_acceso  = r'elevador|ascensor|planta\s+baja|sin\s+escalera|rampa|accesib|sin\s+escalon'
kw_segur   = r'seguridad\s+24|vigilancia\s+24|portero|guardia|caseta|c[aá]mara'
kw_tranqui = r'tranquil[ao]|silencios[ao]|privado|residencial\s+privado|zona\s+residencial'
kw_comfort = r'jardín|jardin|patio|área\s+verde|[aá]rea\s+verde'

mask_precio_am = df['price'].between(PRECIO_MIN_AM, PRECIO_MAX_AM)
mask_rec_am    = df['bedrooms'].fillna(0).isin([0, 1, 2, 3])
mask_kw_am     = (
    texto.str.contains(kw_acceso,  na=False) |
    texto.str.contains(kw_segur,   na=False) |
    texto.str.contains(kw_tranqui, na=False) |
    texto.str.contains(kw_comfort, na=False)
)

mask_tercera_edad = mask_res & mask_precio_am & mask_rec_am & mask_kw_am

# Estudiante tiene prioridad en overlap
mask_solo_am = mask_tercera_edad & ~mask_estudiante

# ─── Asignación de segmento ───────────────────────────────────────────────────
df['segmento'] = 'General'
df.loc[mask_solo_am,    'segmento'] = 'Tercera Edad'
df.loc[mask_estudiante, 'segmento'] = 'Estudiantes'

# ─── Razón de asignación (debug / transparencia) ─────────────────────────────
def razon_estudiante(row, txt):
    razones = []
    if row['price'] <= PRECIO_MAX_EST:
        razones.append(f'precio ${row["price"]:,.0f}')
    if re.search(kw_est, txt):
        razones.append('keyword universitario')
    if row.get('bedrooms', 0) in [0, 1, 2]:
        razones.append(f'{int(row["bedrooms"]) if not pd.isna(row.get("bedrooms")) else 0} rec.')
    return ' | '.join(razones)

def razon_am(row, txt):
    razones = []
    if re.search(kw_acceso, txt):  razones.append('accesibilidad')
    if re.search(kw_segur, txt):   razones.append('seguridad 24h')
    if re.search(kw_tranqui, txt): razones.append('zona tranquila')
    if re.search(kw_comfort, txt): razones.append('jardín/patio')
    razones.append(f'precio ${row["price"]:,.0f}')
    return ' | '.join(razones)

razones = []
for i, row in df.iterrows():
    t = texto.loc[i]
    if row['segmento'] == 'Estudiantes':
        razones.append(razon_estudiante(row, t))
    elif row['segmento'] == 'Tercera Edad':
        razones.append(razon_am(row, t))
    else:
        razones.append('')
df['razon_segmento'] = razones

# ─── Guardar CSV ──────────────────────────────────────────────────────────────
cols_out = [
    'name', 'phone', 'email', 'address', 'colonia', 'price',
    'property_type', 'bedrooms', 'bathrooms', 'parking', 'area_m2',
    'description', 'source_url', 'segmento', 'razon_segmento'
]
df_out = df[cols_out].copy()
df_out.to_csv(CSV_OUT, index=False)

# ─── Reporte en consola ───────────────────────────────────────────────────────
est = df[df['segmento'] == 'Estudiantes']
am  = df[df['segmento'] == 'Tercera Edad']
gen = df[df['segmento'] == 'General']

print('=' * 60)
print('SEGMENTACIÓN DE LEADS LAMUDI')
print('=' * 60)
print(f'Total leads           : {len(df):,}')
print(f'  Residenciales       : {mask_res.sum():,}')
print()
print(f'  Segmento Estudiantes  : {len(est):,} propiedades')
if len(est):
    print(f'    Precio mediana      : ${est["price"].median():,.0f} MXN/mes')
    print(f'    Precio mín / máx    : ${est["price"].min():,.0f} / ${est["price"].max():,.0f}')
    print(f'    Top 5 colonias:')
    for col, n in est['colonia'].value_counts().head(5).items():
        print(f'      {col}: {n}')
print()
print(f'  Segmento Tercera Edad : {len(am):,} propiedades')
if len(am):
    print(f'    Precio mediana      : ${am["price"].median():,.0f} MXN/mes')
    print(f'    Precio mín / máx    : ${am["price"].min():,.0f} / ${am["price"].max():,.0f}')
    print(f'    Top 5 colonias:')
    for col, n in am['colonia'].value_counts().head(5).items():
        print(f'      {col}: {n}')
print()
print(f'  General (sin segmento): {len(gen):,} propiedades')
print('=' * 60)
print(f'CSV guardado en: {CSV_OUT}')
