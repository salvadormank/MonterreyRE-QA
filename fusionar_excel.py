"""
Fusiona teléfonos al Excel original y parsea features (m², recámaras, baños,
estacionamientos). Genera propiedades_enriquecido.xlsx con 4 hojas.
"""

import re
import numpy as np
import pandas as pd

EXCEL_ORIG = '../propiedades_websights.xlsx'
CSV_PHONES = '../analisis_propiedades/contactos_segmentos_v2.csv'
EXCEL_OUT  = '../propiedades_enriquecido.xlsx'
BASE_URL   = "https://www.inmuebles24.com"


# ── Helpers de extracción ─────────────────────────────────────────

def extract_num(pattern, text):
    m = re.search(pattern, str(text))
    return float(m.group(1)) if m else np.nan

def parse_features(df):
    f0 = df['features[0]'].astype(str)
    f1 = df['features[1]'].astype(str)
    f2 = df['features[2]'].astype(str)
    f3 = df['features[3]'].astype(str)

    df['m2']            = f0.apply(lambda x: extract_num(r'(\d+(?:\.\d+)?)\s*m²', x))
    df['recamaras']     = f1.apply(lambda x: extract_num(r'(\d+)\s*rec', x))
    df['banos']         = f2.apply(lambda x: extract_num(r'(\d+(?:\.\d+)?)\s*ba', x))
    df['estacionamientos'] = f3.apply(lambda x: extract_num(r'(\d+)\s*estac', x))

    # Precio por m² (solo donde hay m² y precio razonable)
    df['precio_m2'] = np.where(
        df['m2'] > 5,
        (df['price'] / df['m2']).round(1),
        np.nan
    )
    return df

def clean_phone(val):
    if not isinstance(val, str):
        return np.nan
    digits = re.sub(r'\D', '', val)
    return digits[:10] if len(digits) >= 10 else np.nan


# ── Main ──────────────────────────────────────────────────────────

def main():
    df = pd.read_excel(EXCEL_ORIG)
    df_phones = pd.read_csv(CSV_PHONES)

    print(f'Excel original : {len(df):,} filas  ×  {len(df.columns)} cols')
    print(f'CSV teléfonos  : {len(df_phones):,} filas')

    # ── 1. Parsear features ya existentes ────────────────────────
    df = parse_features(df)

    n_m2 = df['m2'].notna().sum()
    print(f'\nCampos parseados:')
    print(f'  m²              : {n_m2:,} ({n_m2/len(df)*100:.0f}%)')
    print(f'  recámaras       : {df["recamaras"].notna().sum():,}')
    print(f'  baños           : {df["banos"].notna().sum():,}')
    print(f'  estacionamientos: {df["estacionamientos"].notna().sum():,}')
    print(f'  precio/m²       : {df["precio_m2"].notna().sum():,}')

    # ── 2. Fusionar teléfonos por URL ────────────────────────────
    df['full_url'] = BASE_URL + df['url'].astype(str)
    df_m = df_phones[['full_url', 'segmento', 'telefono']].copy()
    df_m['telefono'] = df_m['telefono'].apply(clean_phone)

    df = df.merge(df_m, on='full_url', how='left')
    df.drop(columns=['full_url'], inplace=True)

    n_tel = df['telefono'].notna().sum()
    print(f'\nTeléfonos fusionados: {n_tel}')
    print(f'  Estudiantes         : {(df["segmento"]=="Estudiantes").sum()}')
    print(f'  Adulto mayor        : {(df["segmento"]=="Adulto mayor (proxy)").sum()}')

    # ── 3. Columnas de salida ────────────────────────────────────
    col_order = [
        'segmento', 'telefono',
        '_id', 'location', 'address', 'price', 'm2', 'precio_m2',
        'recamaras', 'banos', 'estacionamientos',
        'publisher', 'publisherCode', 'userViews',
        'Nuevo', 'Amueblado', 'Lujo', 'publishedSince',
        'url', 'latlon', 'description',
        'featuresPills[0]', 'featuresPills[1]', 'featuresPills[2]',
        'featuresPills[3]', 'featuresPills[4]',
    ]
    col_order = [c for c in col_order if c in df.columns]
    df_out = df[col_order]

    # ── 4. Guardar Excel con 4 hojas ─────────────────────────────
    fmt_cols_contact = ['segmento', 'location', 'address', 'price', 'm2',
                        'precio_m2', 'recamaras', 'banos', 'estacionamientos',
                        'publisher', 'telefono', 'url']
    fmt_cols_contact = [c for c in fmt_cols_contact if c in df_out.columns]

    with pd.ExcelWriter(EXCEL_OUT, engine='xlsxwriter') as writer:
        wb = writer.book
        hdr = wb.add_format({'bold': True, 'bg_color': '#2C5F8A',
                             'font_color': 'white', 'border': 1, 'align': 'center'})
        num_fmt = wb.add_format({'num_format': '#,##0.0'})

        def write_sheet(df_s, name, cols=None):
            data = df_s[cols] if cols else df_s
            data.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            for i, c in enumerate(data.columns):
                ws.write(0, i, c, hdr)
                ws.set_column(i, i, max(13, len(str(c)) + 2))

        write_sheet(df_out, 'Todas')
        write_sheet(
            df_out[df_out['telefono'].notna()],
            'Contactos', fmt_cols_contact
        )
        write_sheet(
            df_out[df_out['segmento'] == 'Estudiantes'],
            'Estudiantes', fmt_cols_contact
        )
        write_sheet(
            df_out[df_out['segmento'] == 'Adulto mayor (proxy)'],
            'Adulto mayor', fmt_cols_contact
        )

    print(f'\n✓ Excel guardado: {EXCEL_OUT}')

    # ── 5. Estadísticas de precio/m² por segmento ───────────────
    print('\n── Precio/m² por segmento (mediana) ──')
    for seg, label in [('Estudiantes', 'Estudiantes'),
                       ('Adulto mayor (proxy)', 'Adulto mayor')]:
        sub = df_out[(df_out['segmento'] == seg) & df_out['precio_m2'].notna()]
        if len(sub):
            print(f'  {label}: ${sub["precio_m2"].median():.0f}/m²  '
                  f'(rango ${sub["precio_m2"].min():.0f}–${sub["precio_m2"].max():.0f})')

    overall = df_out[df_out['precio_m2'].notna()]
    print(f'  General   : ${overall["precio_m2"].median():.0f}/m²')

    print('\n── Contactos con teléfono ──')
    contactos = df_out[df_out['telefono'].notna()][fmt_cols_contact]
    print(contactos.to_string(index=False))


if __name__ == '__main__':
    main()
