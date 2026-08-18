"""
Genera leads_lamudi.html — app de búsqueda de leads estática,
moderna y self-contained (sin servidor).
"""
import json, pandas as pd, re, numpy as np

# ── Re-calcular scores ────────────────────────────────────────────
df = pd.read_csv('./leads_lamudi_enriched_openai.csv')
desc_lower = df['description'].fillna('').str.lower()

KW_EST = r'estudiante|universitari|tec |tecnol|itesm|uanl|udem|campus'
KW_3RA = r'tranquilo|tranquila|silencio|seguro|segura|privada|privado|adulto mayor'
KW_EXT = r'ejecutivo|corporativo|expat|relocation|internacional|penthouse|premium|exclusivo|lujo'
COLONIAS_PREMIUM = {'Valle Oriente','Obispado','San Pedro Garza García','Del Paseo Residencial',
                    'Colinas de San Jerónimo','Cumbres','Loma Larga','Contry'}

def score_est(r,d):
    s=0; p=r.get('price',0) or 0
    if p<=12000: s+=3
    elif p<=18000: s+=2
    elif p<=25000: s+=1
    if r.get('furnished')==1: s+=3
    if r.get('property_type') in ('departamento','casa'): s+=2
    if re.search(KW_EST,d): s+=4
    if r.get('has_minisplit_ac')==1: s+=1
    if r.get('bedrooms') in (1,2): s+=1
    if r.get('allows_pets')==1: s+=1
    return s

def score_3ra(r,d):
    s=0
    if r.get('has_elevator')==1: s+=4
    if r.get('floor')==0: s+=3
    if r.get('has_security_24h')==1: s+=3
    if r.get('has_lobby')==1: s+=1
    if r.get('property_type') in ('departamento','casa'): s+=2
    if re.search(KW_3RA,d): s+=3
    p=r.get('price',0) or 0
    if p<=30000: s+=2
    if r.get('has_gym')==1: s+=1
    if r.get('has_garden')==1: s+=1
    if r.get('furnished')==1: s+=1
    return s

def score_ext(r,d):
    s=0
    if r.get('furnished')==1: s+=4
    if r.get('has_security_24h')==1: s+=3
    if r.get('has_gym')==1: s+=2
    if r.get('has_pool')==1: s+=2
    if r.get('has_elevator')==1: s+=2
    if r.get('has_lobby')==1: s+=1
    if r.get('has_sala_juntas')==1: s+=1
    if r.get('has_coworking')==1: s+=1
    if r.get('property_type') in ('departamento','oficina'): s+=1
    if r.get('colonia') in COLONIAS_PREMIUM: s+=3
    if re.search(KW_EXT,d): s+=3
    p=r.get('price',0) or 0
    if p>=25000: s+=2
    if p>=50000: s+=1
    return s

rows = df.to_dict('records')
df['score_est'] = [score_est(r,d) for r,d in zip(rows,desc_lower)]
df['score_3ra'] = [score_3ra(r,d) for r,d in zip(rows,desc_lower)]
df['score_ext'] = [score_ext(r,d) for r,d in zip(rows,desc_lower)]

AMENITY_MAP = {
    'has_pool':'🏊 Alberca','has_gym':'🏋 Gimnasio','has_elevator':'🛗 Elevador',
    'has_terrace':'🌿 Terraza','has_security_24h':'🔒 Seguridad 24h','has_minisplit_ac':'❄️ AC/Minisplit',
    'has_lobby':'🏢 Lobby','has_coworking':'💻 Coworking','has_sala_juntas':'📋 Sala juntas',
    'has_garden':'🌳 Jardín','has_rooftop':'🏙️ Rooftop','has_bodega':'📦 Bodega',
    'has_heating':'🔥 Calefacción','allows_pets':'🐾 Mascotas','has_service_room':'🧹 Cto. servicio',
}
TYPE_LABEL = {
    'departamento':'Departamento','casa':'Casa','oficina':'Oficina',
    'local_comercial':'Local','bodega_industrial':'Bodega Ind.','terreno':'Terreno',
    'edificio':'Edificio','consultorio':'Consultorio',
}
TYPE_ICON = {
    'Departamento':'🏢','Casa':'🏠','Oficina':'🖥️','Local':'🏪',
    'Bodega Ind.':'🏭','Terreno':'🌐','Edificio':'🏗️','Consultorio':'🏥','Otro':'📍',
}

leads = []
for _, row in df.iterrows():
    amenities = [label for col, label in AMENITY_MAP.items() if row.get(col)==1]
    ptype = TYPE_LABEL.get(str(row.get('property_type','')),'Otro')
    leads.append({
        'name': str(row['name'])[:60] if pd.notna(row.get('name')) else 'Sin nombre',
        'phone': str(row['phone']).replace('.0','') if pd.notna(row.get('phone')) else '',
        'url': str(row.get('source_url','')),
        'type': ptype,
        'icon': TYPE_ICON.get(ptype,'📍'),
        'price': int(row['price']) if pd.notna(row.get('price')) else 0,
        'colonia': str(row['colonia'])[:35] if pd.notna(row.get('colonia')) else '',
        'm2': int(row['area_m2']) if pd.notna(row.get('area_m2')) else None,
        'beds': int(row['bedrooms']) if pd.notna(row.get('bedrooms')) else None,
        'baths': float(row['bathrooms']) if pd.notna(row.get('bathrooms')) else None,
        'parking': int(row['parking']) if pd.notna(row.get('parking')) else None,
        'furnished': 1 if row.get('furnished')==1 else 0,
        'amenities': amenities,
        'score_est': int(row['score_est']),
        'score_3ra': int(row['score_3ra']),
        'score_ext': int(row['score_ext']),
        'notes': str(row['notes'])[:100] if pd.notna(row.get('notes')) else '',
    })

leads_json = json.dumps(leads, ensure_ascii=False)

# ── HTML ──────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WebSights — Leads Lamudi Monterrey</title>
<style>
  :root {{
    --blue:    #2C5F8A;
    --blue2:   #1a3d5c;
    --accent:  #27AE60;
    --orange:  #F39C12;
    --red:     #E74C3C;
    --bg:      #0f1923;
    --card:    #162130;
    --card2:   #1e2f42;
    --border:  #243447;
    --text:    #e8edf2;
    --muted:   #7a9ab5;
    --radius:  12px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }}

  /* ── Header ── */
  header {{
    background: linear-gradient(135deg, var(--blue2) 0%, var(--blue) 100%);
    padding: 28px 32px 24px;
    border-bottom: 1px solid var(--border);
  }}
  header h1 {{ font-size: 1.6rem; font-weight: 700; letter-spacing: -0.5px; }}
  header p  {{ color: #adc8e0; font-size: 0.9rem; margin-top: 4px; }}
  .header-stats {{
    display: flex; gap: 24px; margin-top: 16px; flex-wrap: wrap;
  }}
  .stat {{
    background: rgba(255,255,255,0.08);
    border-radius: 8px; padding: 10px 18px; text-align: center;
  }}
  .stat-val {{ font-size: 1.4rem; font-weight: 700; color: #fff; }}
  .stat-lbl {{ font-size: 0.72rem; color: #adc8e0; text-transform: uppercase; letter-spacing: 0.5px; }}

  /* ── Controls ── */
  .controls {{
    padding: 20px 32px;
    background: var(--card);
    border-bottom: 1px solid var(--border);
    display: flex; gap: 12px; flex-wrap: wrap; align-items: center;
  }}
  .seg-btns {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  .seg-btn {{
    padding: 9px 20px; border-radius: 50px; border: 2px solid transparent;
    font-size: 0.85rem; font-weight: 600; cursor: pointer;
    transition: all 0.2s ease; background: var(--card2); color: var(--muted);
  }}
  .seg-btn:hover {{ color: var(--text); border-color: var(--blue); }}
  .seg-btn.active-all   {{ background: var(--blue);   color: #fff; border-color: var(--blue); }}
  .seg-btn.active-est   {{ background: #1a5276;       color: #fff; border-color: #2980b9; }}
  .seg-btn.active-3ra   {{ background: #1d6a47;       color: #fff; border-color: var(--accent); }}
  .seg-btn.active-ext   {{ background: #7d5a0a;       color: #fff; border-color: var(--orange); }}

  .search-wrap {{ flex: 1; min-width: 200px; position: relative; }}
  .search-wrap input {{
    width: 100%; padding: 9px 14px 9px 38px;
    background: var(--card2); border: 1px solid var(--border);
    border-radius: 8px; color: var(--text); font-size: 0.88rem; outline: none;
    transition: border-color 0.2s;
  }}
  .search-wrap input:focus {{ border-color: var(--blue); }}
  .search-icon {{ position: absolute; left: 12px; top: 50%; transform: translateY(-50%); color: var(--muted); font-size: 0.9rem; }}

  .sort-sel {{
    padding: 9px 12px; background: var(--card2); border: 1px solid var(--border);
    border-radius: 8px; color: var(--text); font-size: 0.85rem; outline: none; cursor: pointer;
  }}

  /* ── Counter ── */
  .counter-bar {{
    padding: 10px 32px; background: var(--bg);
    font-size: 0.82rem; color: var(--muted);
    border-bottom: 1px solid var(--border);
  }}
  .counter-bar span {{ color: var(--text); font-weight: 600; }}

  /* ── Grid ── */
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 16px; padding: 24px 32px;
  }}

  /* ── Card ── */
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px;
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    animation: fadeIn 0.3s ease both;
    display: flex; flex-direction: column; gap: 12px;
  }}
  .card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(44,95,138,0.25);
    border-color: var(--blue);
  }}
  @keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}

  .card-top {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; }}
  .card-type {{
    display: flex; align-items: center; gap: 6px;
    font-size: 0.78rem; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.5px;
  }}
  .card-type .icon {{ font-size: 1.1rem; }}

  .score-chips {{ display: flex; gap: 5px; }}
  .chip {{
    font-size: 0.7rem; font-weight: 700; padding: 3px 8px;
    border-radius: 50px; border: 1.5px solid; cursor: default;
    white-space: nowrap;
  }}
  .chip-est {{ color: #5dade2; border-color: #5dade2; background: rgba(93,173,226,0.1); }}
  .chip-3ra {{ color: #58d68d; border-color: #58d68d; background: rgba(88,214,141,0.1); }}
  .chip-ext {{ color: #f5b041; border-color: #f5b041; background: rgba(245,176,65,0.1); }}

  .card-name {{ font-size: 0.95rem; font-weight: 600; color: var(--text); line-height: 1.3; }}
  .card-price {{ font-size: 1.4rem; font-weight: 800; color: #fff; letter-spacing: -0.5px; }}
  .card-price span {{ font-size: 0.8rem; font-weight: 400; color: var(--muted); }}

  .card-meta {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .meta-item {{
    display: flex; align-items: center; gap: 4px;
    font-size: 0.8rem; color: var(--muted);
  }}
  .meta-item strong {{ color: var(--text); }}

  .amenity-list {{ display: flex; flex-wrap: wrap; gap: 5px; }}
  .amenity-tag {{
    font-size: 0.72rem; padding: 3px 9px;
    background: rgba(44,95,138,0.2); border: 1px solid rgba(44,95,138,0.4);
    border-radius: 50px; color: #9cc5e4;
  }}

  .card-notes {{ font-size: 0.78rem; color: var(--muted); font-style: italic; }}

  .card-footer {{
    display: flex; gap: 8px; align-items: center; margin-top: auto; padding-top: 4px;
    border-top: 1px solid var(--border);
  }}
  .btn-phone {{
    display: flex; align-items: center; gap: 5px;
    padding: 7px 14px; border-radius: 8px;
    background: rgba(39,174,96,0.15); border: 1px solid rgba(39,174,96,0.3);
    color: #58d68d; font-size: 0.8rem; font-weight: 600;
    text-decoration: none; transition: background 0.15s;
  }}
  .btn-phone:hover {{ background: rgba(39,174,96,0.28); }}
  .btn-url {{
    display: flex; align-items: center; gap: 5px; flex: 1;
    padding: 7px 14px; border-radius: 8px;
    background: var(--blue); color: #fff;
    font-size: 0.8rem; font-weight: 600;
    text-decoration: none; transition: background 0.15s; text-align: center;
    justify-content: center;
  }}
  .btn-url:hover {{ background: var(--blue2); }}
  .no-phone {{ color: var(--muted); font-size: 0.78rem; }}

  /* ── Empty state ── */
  .empty {{
    grid-column: 1/-1; text-align: center; padding: 80px 20px; color: var(--muted);
  }}
  .empty .big {{ font-size: 3rem; margin-bottom: 12px; }}
  .empty p {{ font-size: 1rem; }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: var(--bg); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
</style>
</head>
<body>

<header>
  <h1>🏙️ WebSights — Leads Inmobiliarios Lamudi · Monterrey</h1>
  <p>Base de datos enriquecida con GPT-4o-mini · Mayo 2026 · 1,029 propiedades</p>
  <div class="header-stats">
    <div class="stat"><div class="stat-val" id="cnt-all">1,029</div><div class="stat-lbl">Total leads</div></div>
    <div class="stat"><div class="stat-val" style="color:#5dade2" id="cnt-est">—</div><div class="stat-lbl">🎓 Estudiantes</div></div>
    <div class="stat"><div class="stat-val" style="color:#58d68d" id="cnt-3ra">—</div><div class="stat-lbl">🧓 Tercera Edad</div></div>
    <div class="stat"><div class="stat-val" style="color:#f5b041" id="cnt-ext">—</div><div class="stat-lbl">✈️ Extranjeros</div></div>
  </div>
</header>

<div class="controls">
  <div class="seg-btns">
    <button class="seg-btn active-all" onclick="setSegment('all',this)">🌐 Todos</button>
    <button class="seg-btn" onclick="setSegment('est',this)">🎓 Estudiantes</button>
    <button class="seg-btn" onclick="setSegment('3ra',this)">🧓 Tercera Edad</button>
    <button class="seg-btn" onclick="setSegment('ext',this)">✈️ Extranjeros / Ejecutivos</button>
  </div>
  <div class="search-wrap">
    <span class="search-icon">🔍</span>
    <input type="text" id="searchBox" placeholder="Buscar por colonia, nombre, tipo..." oninput="render()">
  </div>
  <select class="sort-sel" id="sortBy" onchange="render()">
    <option value="score">Ordenar: Score</option>
    <option value="price_asc">Precio ↑</option>
    <option value="price_desc">Precio ↓</option>
    <option value="m2">Mayor m²</option>
  </select>
</div>

<div class="counter-bar">
  Mostrando <span id="shown">—</span> de <span id="total">1,029</span> propiedades
</div>

<div class="grid" id="grid"></div>

<script>
const DATA = {leads_json};

let currentSegment = 'all';

function setSegment(seg, btn) {{
  currentSegment = seg;
  document.querySelectorAll('.seg-btn').forEach(b => {{
    b.className = 'seg-btn';
  }});
  btn.classList.add('active-' + seg);
  render();
}}

function fmt(n) {{
  return '$' + n.toLocaleString('es-MX');
}}

function scoreKey(seg) {{
  return seg === 'est' ? 'score_est' : seg === '3ra' ? 'score_3ra' : 'score_ext';
}}

function getScore(lead, seg) {{
  if (seg === 'all') return Math.max(lead.score_est, lead.score_3ra, lead.score_ext);
  return lead[scoreKey(seg)];
}}

function scoreBar(val, max, color) {{
  const pct = Math.round((val / max) * 100);
  return `<div style="background:#243447;border-radius:4px;height:4px;width:60px;overflow:hidden">
    <div style="background:${{color}};height:100%;width:${{pct}}%;transition:width 0.4s ease"></div>
  </div>`;
}}

function render() {{
  const q    = document.getElementById('searchBox').value.toLowerCase().trim();
  const sort = document.getElementById('sortBy').value;
  const seg  = currentSegment;
  const maxScores = {{ est: 13, '3ra': 16, ext: 23, all: 23 }};

  let filtered = DATA.filter(d => {{
    if (seg !== 'all' && getScore(d, seg) === 0) return false;
    if (q) {{
      const hay = (d.name + d.colonia + d.type + d.notes).toLowerCase();
      if (!hay.includes(q)) return false;
    }}
    return true;
  }});

  // Sort
  if (sort === 'score') {{
    filtered.sort((a,b) => getScore(b,seg) - getScore(a,seg));
  }} else if (sort === 'price_asc') {{
    filtered.sort((a,b) => a.price - b.price);
  }} else if (sort === 'price_desc') {{
    filtered.sort((a,b) => b.price - a.price);
  }} else if (sort === 'm2') {{
    filtered.sort((a,b) => (b.m2||0) - (a.m2||0));
  }}

  // Update counters
  document.getElementById('shown').textContent = filtered.length.toLocaleString('es-MX');
  const cntEst = DATA.filter(d => d.score_est > 5).length;
  const cnt3ra = DATA.filter(d => d.score_3ra > 5).length;
  const cntExt = DATA.filter(d => d.score_ext > 5).length;
  document.getElementById('cnt-est').textContent = cntEst.toLocaleString('es-MX');
  document.getElementById('cnt-3ra').textContent = cnt3ra.toLocaleString('es-MX');
  document.getElementById('cnt-ext').textContent = cntExt.toLocaleString('es-MX');

  const grid = document.getElementById('grid');

  if (filtered.length === 0) {{
    grid.innerHTML = `<div class="empty"><div class="big">🔍</div><p>Sin resultados para esta búsqueda.</p></div>`;
    return;
  }}

  const cards = filtered.map((d, i) => {{
    const score = getScore(d, seg);
    const maxS  = maxScores[seg] || 23;
    const color = seg==='est'?'#5dade2': seg==='3ra'?'#58d68d': seg==='ext'?'#f5b041':'#5dade2';

    const chips = [];
    if (d.score_est > 5)  chips.push(`<span class="chip chip-est">🎓 Est. ${{d.score_est}}</span>`);
    if (d.score_3ra > 5)  chips.push(`<span class="chip chip-3ra">🧓 3ra ${{d.score_3ra}}</span>`);
    if (d.score_ext > 8)  chips.push(`<span class="chip chip-ext">✈️ Ext ${{d.score_ext}}</span>`);

    const metaItems = [];
    if (d.m2)      metaItems.push(`<span class="meta-item">📐 <strong>${{d.m2}} m²</strong></span>`);
    if (d.beds)    metaItems.push(`<span class="meta-item">🛏 <strong>${{d.beds}} rec</strong></span>`);
    if (d.baths)   metaItems.push(`<span class="meta-item">🚿 <strong>${{d.baths}} baños</strong></span>`);
    if (d.parking) metaItems.push(`<span class="meta-item">🚗 <strong>${{d.parking}} cajones</strong></span>`);
    if (d.furnished) metaItems.push(`<span class="meta-item">🛋️ <strong>Amueblado</strong></span>`);

    const amenHTML = d.amenities.slice(0,6).map(a =>
      `<span class="amenity-tag">${{a}}</span>`).join('');

    const phoneHTML = d.phone
      ? `<a class="btn-phone" href="tel:${{d.phone}}">📞 ${{d.phone}}</a>`
      : `<span class="no-phone">Sin tel.</span>`;

    const delay = (i % 12) * 0.03;

    return `
<div class="card" style="animation-delay:${{delay}}s">
  <div class="card-top">
    <div class="card-type"><span class="icon">${{d.icon}}</span>${{d.type}}</div>
    <div class="score-chips">${{chips.join('')}}</div>
  </div>
  <div class="card-name">${{d.name || 'Sin nombre'}}</div>
  <div style="display:flex;align-items:center;gap:12px">
    <div class="card-price">${{fmt(d.price)}} <span>/mes</span></div>
    <div style="display:flex;flex-direction:column;gap:3px;align-items:flex-start">
      ${{scoreBar(score, maxS, color)}}
      <span style="font-size:0.7rem;color:var(--muted)">score ${{score}}/${{maxS}}</span>
    </div>
  </div>
  ${{metaItems.length ? `<div class="card-meta">${{metaItems.join('')}}</div>` : ''}}
  <div class="meta-item">📍 <strong>${{d.colonia}}</strong></div>
  ${{amenHTML ? `<div class="amenity-list">${{amenHTML}}</div>` : ''}}
  ${{d.notes ? `<div class="card-notes">"${{d.notes}}"</div>` : ''}}
  <div class="card-footer">
    ${{phoneHTML}}
    <a class="btn-url" href="${{d.url}}" target="_blank" rel="noopener">Ver en Lamudi →</a>
  </div>
</div>`;
  }});

  grid.innerHTML = cards.join('');
}}

// Init
render();
</script>
</body>
</html>"""

out = './leads_lamudi.html'
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)

size_kb = len(html.encode('utf-8')) / 1024
print(f'✓ HTML generado: {out}')
print(f'  Tamaño: {size_kb:.0f} KB')
print(f'  Leads embebidos: {len(leads)}')
