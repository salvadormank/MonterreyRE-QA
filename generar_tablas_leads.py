"""
Genera el archivo leads_segmentos.tex con tablas formateadas
de los top 15 leads por segmento, incluyendo URL clickeable.
"""
import pandas as pd
import numpy as np
import re

def esc(s, maxlen=30):
    if not isinstance(s, str) or s != s:
        return '---'
    s = s[:maxlen]
    for old, new in [('&', r'\&'), ('%', r'\%'), ('#', r'\#'),
                     ('_', r'\_'), ('$', r'\$'), ('{', r'\{'), ('}', r'\}'),
                     ('^', r'\^{}'), ('~', r'\textasciitilde{}'), ('\\', r'\textbackslash{}')]:
        s = s.replace(old, new)
    return s

def fmt_phone(p):
    if p != p or str(p) in ('nan', 'None', ''):
        return '---'
    return str(p).replace('.0', '').strip()

def fmt_price(p):
    try:
        return r'\${:,.0f}'.format(float(p))
    except Exception:
        return '---'

def fmt_m2(m):
    try:
        return '{:.0f}'.format(float(m))
    except Exception:
        return '---'

# ── Cargar datos y recalcular scores ──────────────────────────────────────
df = pd.read_csv('./leads_lamudi_enriched_openai.csv')
desc_lower = df['description'].fillna('').str.lower()

KW_EST = r'estudiante|universitari|tec |tecnol|itesm|uanl|udem|campus'
KW_3RA = r'tranquilo|tranquila|silencio|seguro|segura|privada|privado|adulto mayor'
KW_EXT = r'ejecutivo|corporativo|expat|relocation|internacional|penthouse|premium|exclusivo|lujo'
COLONIAS_PREMIUM = {'Valle Oriente','Obispado','San Pedro Garza García','Del Paseo Residencial',
                    'Colinas de San Jerónimo','Cumbres','Loma Larga','Contry'}

def score_est(r, d):
    s = 0
    p = r.get('price', 0) or 0
    if p <= 12000: s += 3
    elif p <= 18000: s += 2
    elif p <= 25000: s += 1
    if r.get('furnished') == 1: s += 3
    if r.get('property_type') in ('departamento','casa'): s += 2
    if re.search(KW_EST, d): s += 4
    if r.get('has_minisplit_ac') == 1: s += 1
    if r.get('bedrooms') in (1, 2): s += 1
    if r.get('allows_pets') == 1: s += 1
    return s

def score_3ra(r, d):
    s = 0
    if r.get('has_elevator') == 1: s += 4
    if r.get('floor') == 0: s += 3
    if r.get('has_security_24h') == 1: s += 3
    if r.get('has_lobby') == 1: s += 1
    if r.get('property_type') in ('departamento','casa'): s += 2
    if re.search(KW_3RA, d): s += 3
    p = r.get('price', 0) or 0
    if p <= 30000: s += 2
    if r.get('has_gym') == 1: s += 1
    if r.get('has_garden') == 1: s += 1
    if r.get('furnished') == 1: s += 1
    return s

def score_ext(r, d):
    s = 0
    if r.get('furnished') == 1: s += 4
    if r.get('has_security_24h') == 1: s += 3
    if r.get('has_gym') == 1: s += 2
    if r.get('has_pool') == 1: s += 2
    if r.get('has_elevator') == 1: s += 2
    if r.get('has_lobby') == 1: s += 1
    if r.get('has_sala_juntas') == 1: s += 1
    if r.get('has_coworking') == 1: s += 1
    if r.get('property_type') in ('departamento','oficina'): s += 1
    if r.get('colonia') in COLONIAS_PREMIUM: s += 3
    if re.search(KW_EXT, d): s += 3
    p = r.get('price', 0) or 0
    if p >= 25000: s += 2
    if p >= 50000: s += 1
    return s

rows = df.to_dict('records')
df['score_est'] = [score_est(r, d) for r, d in zip(rows, desc_lower)]
df['score_3ra'] = [score_3ra(r, d) for r, d in zip(rows, desc_lower)]
df['score_ext'] = [score_ext(r, d) for r, d in zip(rows, desc_lower)]

# ── Generar LaTeX ────────────────────────────────────────────────────────
SEGMENTS = [
    ('score_est', 'Estudiantes',               r'\textcolor{wsblue}{\textbf{Perfil:}} Precio bajo, amueblado, cerca de campus universitario.'),
    ('score_3ra', 'Tercera Edad',               r'\textcolor{wsblue}{\textbf{Perfil:}} Elevador o planta baja, seguridad 24h, entorno tranquilo.'),
    ('score_ext', 'Extranjeros / Ejecutivos',   r'\textcolor{wsblue}{\textbf{Perfil:}} Amueblado, amenidades premium, colonia exclusiva.'),
]

out_lines = [
    r'\section{Leads por Segmento de Mercado}',
    '',
    r'Los siguientes leads corresponden a las 15 propiedades con mayor score para cada',
    r'perfil de cliente. La columna \textbf{URL} es un hipervínculo clickeable que',
    r'abre directamente el anuncio en Lamudi.com.mx.',
    '',
]

for scol, title, perfil in SEGMENTS:
    top = df.nlargest(15, scol).copy()

    rows_tex = []
    for i, (_, row) in enumerate(top.iterrows()):
        fill = r'\rowcolor{wsrowalt}' if i % 2 == 0 else ''
        name  = esc(row.get('name'), 26)
        phone = fmt_phone(row.get('phone'))
        url   = str(row.get('source_url', ''))
        ptype = esc(row.get('property_type'), 18)
        price = fmt_price(row.get('price'))
        col   = esc(row.get('colonia'), 18)
        m2_val = row.get('area_m2')
        m2    = fmt_m2(m2_val) if (m2_val == m2_val and m2_val is not None) else '---'
        score = int(row[scol])

        url_cell = r'\href{' + url + r'}{\color{wsblue}\underline{Ver}}'
        rows_tex.append(
            f'{fill}\n'
            f'{name} & {phone} & {url_cell} & {ptype} & {price} & {col} & {m2} & {score} \\\\'
        )

    table = '\n'.join([
        '',
        f'\\subsection{{{title}}}',
        '',
        f'\\begin{{insightbox}}{perfil}\\end{{insightbox}}',
        '',
        r'\begin{table}[H]',
        r'\centering',
        r'\footnotesize',
        f'\\caption{{Top 15 leads --- {title}}}',
        r'\begin{tabular}{p{2.8cm}p{2.4cm}p{0.8cm}p{2cm}rp{2cm}rr}',
        r'\toprule',
        r'\rowcolor{wsblue}\color{white}\textbf{Nombre} & \color{white}\textbf{Tel\'{e}fono} & '
        r'\color{white}\textbf{URL} & \color{white}\textbf{Tipo} & \color{white}\textbf{Precio} & '
        r'\color{white}\textbf{Colonia} & \color{white}\textbf{m\textsuperscript{2}} & \color{white}\textbf{Score} \\',
        r'\midrule',
        '\n'.join(rows_tex),
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
        '',
    ])
    out_lines.append(table)

out_path = './leads_segmentos.tex'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out_lines))

print(f'Guardado: {out_path}')
print(f'Lineas generadas: {len(out_lines)}')
