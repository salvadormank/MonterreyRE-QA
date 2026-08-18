"""
Genera leads_segmentados.tex a partir de leads_segmentados.csv
Formateado para compilar en Overleaf.
"""

import pandas as pd
import re

CSV = './leads_segmentados.csv'
TEX = './leads_segmentados.tex'

df = pd.read_csv(CSV)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def parse_address(addr):
    if pd.isna(addr):
        return '', ''
    parts = [p.strip() for p in str(addr).split(',')]
    return parts[0] if parts else '', parts[1] if len(parts) > 1 else ''

def esc(s):
    """Escapa caracteres especiales de LaTeX."""
    if pd.isna(s):
        return '---'
    s = str(s)
    s = s.replace('&', r'\&').replace('%', r'\%').replace('$', r'\$')
    s = s.replace('#', r'\#').replace('_', r'\_').replace('{', r'\{')
    s = s.replace('}', r'\}').replace('~', r'\textasciitilde{}')
    s = s.replace('^', r'\^{}').replace('\\', r'\textbackslash{}')
    return s

def precio(p):
    if pd.isna(p):
        return '---'
    return f'\\${float(p):,.0f}'

def desc_corta(d, max_chars=220):
    if pd.isna(d):
        return '---'
    d = str(d).strip().replace('\n', ' ')
    d = re.sub(r'\s+', ' ', d)
    if len(d) > max_chars:
        d = d[:max_chars].rsplit(' ', 1)[0] + '\\ldots'
    return esc(d)

def int_o_dash(v):
    try:
        return str(int(float(v)))
    except (ValueError, TypeError):
        return '---'

# ─── Preparar datos ────────────────────────────────────────────────────────────
est = df[df['segmento'] == 'Estudiantes'].copy()
am  = df[df['segmento'] == 'Tercera Edad'].copy()

for seg in [est, am]:
    seg[['colonia_p', 'ciudad']] = seg['address'].apply(
        lambda x: pd.Series(parse_address(x))
    )

est = est.sort_values('price').reset_index(drop=True)
am  = am.sort_values('price').reset_index(drop=True)

# ─── Estadísticas resumen ──────────────────────────────────────────────────────
def stats(seg):
    p = seg['price'].dropna()
    return {
        'n':      len(seg),
        'med':    p.median() if len(p) else 0,
        'minp':   p.min()    if len(p) else 0,
        'maxp':   p.max()    if len(p) else 0,
        'top3':   seg['colonia_p'].value_counts().head(3).to_dict()
    }

se = stats(est)
sa = stats(am)

# ─── Función para generar tabla de leads ─────────────────────────────────────
def tabla_leads(seg, max_rows=25):
    if len(seg) == 0:
        return '\\textit{Sin propiedades en este segmento.}\n'

    lineas = []
    lineas.append(r'\begin{longtable}{@{}p{3.2cm} p{3.2cm} r p{0.8cm} p{0.8cm} p{5.5cm}@{}}')
    lineas.append(r'\toprule')
    lineas.append(r'\textbf{Contacto} & \textbf{Colonia} & \textbf{Renta/mes} & \textbf{Rec.} & \textbf{m\textsuperscript{2}} & \textbf{Descripción breve} \\')
    lineas.append(r'\midrule')
    lineas.append(r'\endhead')
    lineas.append(r'\midrule \multicolumn{6}{r}{\footnotesize Continúa\ldots} \\')
    lineas.append(r'\endfoot')
    lineas.append(r'\bottomrule')
    lineas.append(r'\endlastfoot')

    for _, row in seg.head(max_rows).iterrows():
        nombre = esc(str(row['name'])[:32])
        col    = esc(str(row['colonia_p'])[:28])
        pr     = precio(row['price'])
        rec    = int_o_dash(row.get('bedrooms'))
        area   = int_o_dash(row.get('area_m2'))
        desc   = desc_corta(row.get('description'))
        lineas.append(f'{nombre} & {col} & {pr} & {rec} & {area} & {desc} \\\\')
        lineas.append(r'\addlinespace[2pt]')

    lineas.append(r'\end{longtable}')
    return '\n'.join(lineas)

# ─── Top colonias para resumen ────────────────────────────────────────────────
def top3_tex(top3):
    items = [f'{esc(k)} ({v})' for k, v in top3.items()]
    return ', '.join(items) if items else '---'

# ─── DOCUMENTO LaTeX ──────────────────────────────────────────────────────────
doc = r"""\documentclass[11pt,a4paper]{article}

%% ── Codificación y fuentes ──────────────────────────────────────────────────
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish,es-nodecimaldot]{babel}
\usepackage{lmodern}

%% ── Márgenes ────────────────────────────────────────────────────────────────
\usepackage[top=2.4cm, bottom=2.4cm, left=2.5cm, right=2.5cm]{geometry}

%% ── Tablas ───────────────────────────────────────────────────────────────────
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{multirow}
\usepackage{colortbl}

%% ── Color / diseño ───────────────────────────────────────────────────────────
\usepackage[dvipsnames,table]{xcolor}
\usepackage{tikz}
\usepackage{tcolorbox}
\tcbuselibrary{skins, breakable}

%% ── Encabezados ─────────────────────────────────────────────────────────────
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textcolor{gray}{Leads Lamudi — Segmentación de Mercado}}
\fancyhead[R]{\small\textcolor{gray}{Monterrey, NL · 2026}}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

%% ── Hipervínculos ────────────────────────────────────────────────────────────
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=NavyBlue, urlcolor=NavyBlue}

%% ── Misc ─────────────────────────────────────────────────────────────────────
\usepackage{enumitem}
\usepackage{parskip}
\setlength{\parindent}{0pt}

%% ── Colores personalizados ───────────────────────────────────────────────────
\definecolor{azulLamudi}{HTML}{0052CC}
\definecolor{verdeSeg}{HTML}{1A7A4A}
\definecolor{azulEst}{HTML}{1565C0}
\definecolor{verdeAM}{HTML}{2E7D32}
\definecolor{grisClaro}{HTML}{F5F5F5}
\definecolor{grisLinea}{HTML}{BDBDBD}

%% ── Estilos de caja ──────────────────────────────────────────────────────────
\tcbset{
  cajaest/.style={
    enhanced, breakable,
    colback=blue!4, colframe=azulEst, arc=4pt,
    boxrule=0.8pt, left=6pt, right=6pt, top=6pt, bottom=6pt,
    title style={colback=azulEst, colframe=azulEst},
    fonttitle=\bfseries\color{white}
  },
  cajaAM/.style={
    enhanced, breakable,
    colback=green!4, colframe=verdeAM, arc=4pt,
    boxrule=0.8pt, left=6pt, right=6pt, top=6pt, bottom=6pt,
    title style={colback=verdeAM, colframe=verdeAM},
    fonttitle=\bfseries\color{white}
  },
  cajares/.style={
    enhanced, colback=grisClaro, colframe=grisLinea, arc=3pt,
    boxrule=0.6pt, left=5pt, right=5pt, top=5pt, bottom=5pt
  }
}

\begin{document}

%% ═══════════════════════════════════════════════════════════════
%%  PORTADA
%% ═══════════════════════════════════════════════════════════════
\begin{titlepage}
\begin{tikzpicture}[remember picture, overlay]
  \fill[azulLamudi] (current page.north west) rectangle
    ([yshift=-5.5cm] current page.north east);
  \node[anchor=center, white, font=\LARGE\bfseries] at
    ([yshift=-1.8cm] current page.north) {Reporte de Leads Lamudi};
  \node[anchor=center, white, font=\large] at
    ([yshift=-3.0cm] current page.north)
    {Segmentación: Estudiantes y Tercera Edad};
  \node[anchor=center, white, font=\normalsize] at
    ([yshift=-4.2cm] current page.north)
    {Monterrey, Nuevo León · Mayo 2026};
\end{tikzpicture}

\vspace{6.5cm}

\begin{tcolorbox}[cajares]
\textbf{Fuente de datos:} Lamudi.com.mx — Propiedades en renta en Monterrey, NL \\
\textbf{Total de leads:} """ + f"{len(df):,}" + r""" propiedades scraped \\
\textbf{Segmento Estudiantes:} """ + str(se['n']) + r""" propiedades identificadas \\
\textbf{Segmento Tercera Edad:} """ + str(sa['n']) + r""" propiedades identificadas \\[4pt]
\textbf{Metodología:} Las propiedades se filtran primero por tipo residencial
(casa, departamento, loft). La segmentación combina \emph{rangos de precio},
\emph{número de recámaras} y \emph{análisis de palabras clave} en la descripción
del anuncio. El segmento \emph{Estudiantes} tiene prioridad en caso de solapamiento.
\end{tcolorbox}

\vfill

\begin{center}
\textcolor{gray}{\small Generado con Python · datos al 3 de mayo 2026}
\end{center}
\end{titlepage}

%% ═══════════════════════════════════════════════════════════════
%%  SECCIÓN 1: RESUMEN EJECUTIVO
%% ═══════════════════════════════════════════════════════════════
\section*{Resumen Ejecutivo}

\begin{tcolorbox}[cajares]
\rowcolors{2}{white}{grisClaro}
\begin{tabular}{@{} l r r @{}}
\toprule
\textbf{Métrica} & \textbf{Estudiantes} & \textbf{Tercera Edad} \\
\midrule
N\textdegree\ propiedades
  & """ + str(se['n']) + r""" & """ + str(sa['n']) + r""" \\
Precio mediano (MXN/mes)
  & \$""" + f"{se['med']:,.0f}" + r""" & \$""" + f"{sa['med']:,.0f}" + r""" \\
Precio mínimo (MXN/mes)
  & \$""" + f"{se['minp']:,.0f}" + r""" & \$""" + f"{sa['minp']:,.0f}" + r""" \\
Precio máximo (MXN/mes)
  & \$""" + f"{se['maxp']:,.0f}" + r""" & \$""" + f"{sa['maxp']:,.0f}" + r""" \\
\midrule
\multicolumn{3}{@{}l}{\small \textbf{Criterios de inclusión}} \\
Rango de renta
  & \$9{,}000 -- \$18{,}000 & \$8{,}000 -- \$35{,}000 \\
Recámaras
  & 0 -- 2 & 0 -- 3 \\
Keywords
  & Universitario / Tec / UANL & Elevador / Tranquilo / Seguridad \\
\bottomrule
\end{tabular}
\end{tcolorbox}

\vspace{6pt}

\textbf{Colonias más frecuentes — Estudiantes:} """ + top3_tex(se['top3']) + r"""

\textbf{Colonias más frecuentes — Tercera Edad:} """ + top3_tex(sa['top3']) + r"""

%% ═══════════════════════════════════════════════════════════════
%%  SECCIÓN 2: LEADS PARA ESTUDIANTES
%% ═══════════════════════════════════════════════════════════════
\newpage
\begin{tcolorbox}[cajaest, title={Segmento: Estudiantes}]
Propiedades con renta \(\leq\) \$18{,}000 MXN/mes, tipo residencial,
con 1--2 recámaras o con menciones explícitas a universidades
(Tec de Monterrey, UANL, UDEM, etc.) en la descripción del anuncio.
\end{tcolorbox}

""" + tabla_leads(est) + r"""

%% ═══════════════════════════════════════════════════════════════
%%  SECCIÓN 3: LEADS PARA TERCERA EDAD
%% ═══════════════════════════════════════════════════════════════
\newpage
\begin{tcolorbox}[cajaAM, title={Segmento: Tercera Edad}]
Propiedades de tipo residencial en el rango \$8{,}000--\$35{,}000 MXN/mes,
con 1--3 recámaras, que mencionan en su descripción al menos uno de los
siguientes atributos de confort o accesibilidad: \emph{elevador / planta baja,
seguridad 24h / portero, zona tranquila / silenciosa, jardín / patio}.
\end{tcolorbox}

""" + tabla_leads(am) + r"""

%% ═══════════════════════════════════════════════════════════════
%%  APÉNDICE: NOTA METODOLÓGICA
%% ═══════════════════════════════════════════════════════════════
\newpage
\section*{Nota Metodológica}

\begin{tcolorbox}[cajares]
\textbf{Fuente:} Los anuncios se extrajeron de \texttt{lamudi.com.mx} mediante
web scraping con ScraperAPI durante abril--mayo 2026. Se capturó texto, precio,
tipo de inmueble y datos de contacto del anunciante.

\textbf{Segmento Estudiantes}
\begin{itemize}[leftmargin=1.2em]
  \item Tipo: \texttt{casa}, \texttt{departamento}, \texttt{loft}, \texttt{cuarto}
  \item Precio: $\leq$ \$18{,}000 MXN/mes
  \item Al menos una de: recámaras $\leq$ 2 \textbf{o} keyword universitario
    (\textit{estudiante, Tec, UANL, UDEM, ITESM, roomie, comparte cuarto\ldots})
\end{itemize}

\textbf{Segmento Tercera Edad}
\begin{itemize}[leftmargin=1.2em]
  \item Tipo: \texttt{casa}, \texttt{departamento}, \texttt{loft}, \texttt{cuarto}
  \item Precio: \$8{,}000 -- \$35{,}000 MXN/mes
  \item Recámaras $\leq$ 3
  \item Al menos una de las categorías de keyword: accesibilidad
    (\textit{elevador, rampa, planta baja}), seguridad
    (\textit{guardia, portero, vigilancia 24h}), tranquilidad
    (\textit{tranquilo, silencioso, residencial}), confort
    (\textit{jardín, patio, área verde})
\end{itemize}

El segmento \emph{Estudiantes} tiene prioridad en caso de solapamiento con
\emph{Tercera Edad}. Las propiedades no residenciales (bodegas, oficinas,
locales comerciales, etc.) se excluyeron de ambos segmentos.
\end{tcolorbox}

\end{document}
"""

with open(TEX, 'w', encoding='utf-8') as f:
    f.write(doc)

print(f'Archivo generado: {TEX}')
print(f'Leads Estudiantes : {len(est)}')
print(f'Leads Tercera Edad: {len(am)}')
