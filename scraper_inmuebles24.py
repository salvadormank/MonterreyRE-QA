"""
scraper_inmuebles24.py
─────────────────────────────────────────────────────────────────────────────
Scraper de Inmuebles24.com para propiedades en renta en Monterrey (NL).
Usa ScraperAPI con render=true para obtener el contenido renderizado por JS.

Output: inmuebles24_raw.csv  (mismo formato que leads_lamudi_renta.csv)

Uso:
  export SCRAPERAPI_KEY=tu_key
  python3 scraper_inmuebles24.py --test     # 2 páginas
  python3 scraper_inmuebles24.py            # completo (~30 páginas)
─────────────────────────────────────────────────────────────────────────────
"""

import os, re, time, json, argparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

SCRAPERAPI_KEY = os.environ.get('SCRAPERAPI_KEY', '')
BASE_URL   = 'https://www.inmuebles24.com'
# Monterrey NL — renta de inmuebles — página {page}
SEARCH_URL = 'https://www.inmuebles24.com/propiedades-en-renta-en-nuevo-leon,monterrey.html?pagina={page}'
OUT_CSV    = './inmuebles24_raw.csv'
MAX_PAGES  = 30
SLEEP_SEC  = 2

PHONE_RE = re.compile(
    r'\b(?:\+?52[-.\s]?)?(?:\(?\d{2,3}\)?[-.\s]?)?\d{4}[-.\s]?\d{4}\b'
    r'|\b\d{10}\b'
    r'|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
)


def scraperapi(url: str, render: bool = True) -> requests.Response:
    params = {
        'api_key':      SCRAPERAPI_KEY,
        'url':          url,
        'render':       'true' if render else 'false',
        'country_code': 'mx',
        'premium':      'true',   # bypasa CDN-level blocks
    }
    return requests.get('http://api.scraperapi.com', params=params, timeout=120)


def extract_listings_from_page(html: str) -> list[dict]:
    """Extrae listings del HTML renderizado de Inmuebles24."""
    soup = BeautifulSoup(html, 'lxml')
    results = []

    # Inmuebles24 usa la plataforma Navent (igual que Lamudi MX)
    # Los cards están en <div data-qa="posting PROPERTY"> o similar
    items = (
        soup.find_all('div', attrs={'data-qa': re.compile(r'posting', re.I)}) or
        soup.find_all('div', attrs={'data-postingid': True}) or
        soup.find_all('div', attrs={'data-id': True, 'class': re.compile(r'posting|property|listing', re.I)}) or
        soup.find_all('article', attrs={'data-id': True}) or
        soup.find_all('article', class_=re.compile(r'posting|property|listing|inmueble', re.I)) or
        soup.find_all('section', attrs={'data-qa': re.compile(r'posting', re.I)})
    )

    # Fallback: JSON-LD
    if not items:
        json_lds = soup.find_all('script', type='application/ld+json')
        for jld in json_lds:
            try:
                raw = jld.string or ''
                data = json.loads(raw)
                items_list = data if isinstance(data, list) else [data]
                for item in items_list:
                    t = item.get('@type', '')
                    if t in ('Apartment', 'House', 'RealEstateListing', 'Product', 'Offer'):
                        r = _parse_jsonld(item)
                        if r:
                            results.append(r)
                    elif t == 'ItemList':
                        for el in item.get('itemListElement', []):
                            r = _parse_jsonld(el.get('item', el))
                            if r:
                                results.append(r)
            except Exception:
                pass
        return [r for r in results if r]

    # Fallback 2: buscar en __NEXT_DATA__ o window.__STATE__
    if not items:
        script_tags = soup.find_all('script')
        for sc in script_tags:
            txt = sc.string or ''
            if '__NEXT_DATA__' in txt or 'postings' in txt:
                try:
                    match = re.search(r'__NEXT_DATA__\s*=\s*(\{.*?\});', txt, re.S)
                    if match:
                        nd = json.loads(match.group(1))
                        postings = _dig(nd, ['props', 'pageProps', 'postings']) or []
                        for p in postings:
                            r = _parse_next_data_item(p)
                            if r:
                                results.append(r)
                except Exception:
                    pass
        return [r for r in results if r]

    for item in items:
        try:
            r = _parse_html_item(item)
            if r and r.get('price'):
                results.append(r)
        except Exception:
            pass

    return results


def _dig(obj, keys):
    """Navega un dict anidado por lista de llaves."""
    for k in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(k)
    return obj


def _parse_html_item(item) -> dict:
    text_full = item.get_text(' ', strip=True)

    # Título
    title_el = (
        item.find(attrs={'data-qa': 'posting-title'}) or
        item.find(['h2', 'h3', 'h4'], class_=re.compile(r'title|name|heading', re.I)) or
        item.find('a', class_=re.compile(r'title|name', re.I))
    )
    title = title_el.get_text(strip=True) if title_el else ''

    # Precio
    price_el = (
        item.find(attrs={'data-qa': 'price'}) or
        item.find(attrs={'data-qa': 'PRICE'}) or
        item.find(class_=re.compile(r'price|precio', re.I))
    )
    price_text = price_el.get_text(strip=True) if price_el else ''
    price = _parse_price(price_text)

    # URL
    link_el = item.find('a', href=True)
    url = ''
    if link_el:
        href = link_el['href']
        url = href if href.startswith('http') else BASE_URL + href

    # Descripción
    desc_el = (
        item.find(attrs={'data-qa': 'posting-description'}) or
        item.find(class_=re.compile(r'desc|body|detail|summary', re.I))
    )
    description = desc_el.get_text(strip=True) if desc_el else text_full[:300]

    # Ubicación
    loc_el = (
        item.find(attrs={'data-qa': 'posting-location'}) or
        item.find(attrs={'data-qa': re.compile(r'location|address', re.I)}) or
        item.find(class_=re.compile(r'location|address|colonia|zona|lugar', re.I))
    )
    address = loc_el.get_text(strip=True) if loc_el else ''

    # Poster (propietario vs agencia)
    is_owner = _detect_owner(item)

    # Atributos estructurados
    attrs = _extract_attrs(item, text_full)

    # Teléfono visible en card (raro, pero por si acaso)
    phones = PHONE_RE.findall(text_full)
    phone = phones[0] if phones else np.nan

    return {
        'name':           title,
        'phone':          phone,
        'email':          np.nan,
        'address':        address,
        'colonia':        np.nan,
        'price':          price,
        'operation_type': 'renta',
        'property_type':  _detect_property_type(title + ' ' + description),
        'bedrooms':       attrs.get('bedrooms', np.nan),
        'bathrooms':      attrs.get('bathrooms', np.nan),
        'parking':        attrs.get('parking', np.nan),
        'area_m2':        attrs.get('area_m2', np.nan),
        'terrain_m2':     np.nan,
        'floor':          np.nan,
        'age_years':      np.nan,
        'furnished':      np.nan,
        'has_pool':       np.nan,
        'has_gym':        np.nan,
        'description':    description,
        'is_owner':       is_owner,
        'score':          0,
        'source_url':     url,
        'scraped_at':     pd.Timestamp.now().isoformat(),
        'source':         'inmuebles24',
    }


def _parse_next_data_item(p: dict) -> dict | None:
    """Parsea un posting del JSON __NEXT_DATA__."""
    try:
        price = float(
            re.sub(r'[^\d.]', '', str(p.get('price', '') or p.get('priceTotal', '') or 0)) or 0
        )
        if not price:
            return None
        title = p.get('title', '') or p.get('address', {}).get('addressLocality', '')
        url = p.get('url', '') or p.get('permalink', '')
        if url and not url.startswith('http'):
            url = BASE_URL + url
        address = str(p.get('address', ''))
        desc = p.get('description', '')
        features = p.get('features', {}) or {}
        attrs = {
            'area_m2':   features.get('totalArea') or features.get('area'),
            'bedrooms':  features.get('rooms'),
            'bathrooms': features.get('bathrooms'),
            'parking':   features.get('garages'),
        }
        return {
            'name':           title,
            'phone':          np.nan,
            'email':          np.nan,
            'address':        address,
            'colonia':        np.nan,
            'price':          price,
            'operation_type': 'renta',
            'property_type':  _detect_property_type(title + ' ' + desc),
            'bedrooms':       attrs.get('bedrooms') or np.nan,
            'bathrooms':      attrs.get('bathrooms') or np.nan,
            'parking':        attrs.get('parking') or np.nan,
            'area_m2':        float(attrs.get('area_m2') or 0) or np.nan,
            'terrain_m2':     np.nan,
            'floor':          np.nan,
            'age_years':      np.nan,
            'furnished':      np.nan,
            'has_pool':       np.nan,
            'has_gym':        np.nan,
            'description':    desc,
            'is_owner':       False,
            'score':          0,
            'source_url':     url,
            'scraped_at':     pd.Timestamp.now().isoformat(),
            'source':         'inmuebles24',
        }
    except Exception:
        return None


def _parse_jsonld(item: dict) -> dict | None:
    try:
        price_data = item.get('offers', {})
        raw_price  = price_data.get('price', 0) if price_data else item.get('price', 0)
        price = float(re.sub(r'[^\d.]', '', str(raw_price)) or 0)
        if not price:
            return None
        addr = item.get('address', {})
        address = (
            addr.get('streetAddress', '') + ' ' +
            addr.get('addressLocality', '') + ' ' +
            addr.get('addressRegion', '')
        ).strip() if isinstance(addr, dict) else str(addr)
        url = item.get('url', '')
        return {
            'name':           item.get('name', ''),
            'phone':          np.nan,
            'email':          np.nan,
            'address':        address,
            'colonia':        np.nan,
            'price':          price,
            'operation_type': 'renta',
            'property_type':  item.get('@type', '').lower(),
            'bedrooms':       item.get('numberOfRooms', np.nan),
            'bathrooms':      item.get('numberOfBathroomsTotal', np.nan),
            'parking':        np.nan,
            'area_m2':        item.get('floorSize', {}).get('value', np.nan) if isinstance(item.get('floorSize'), dict) else np.nan,
            'terrain_m2':     np.nan,
            'floor':          np.nan,
            'age_years':      np.nan,
            'furnished':      np.nan,
            'has_pool':       np.nan,
            'has_gym':        np.nan,
            'description':    item.get('description', ''),
            'is_owner':       False,
            'score':          0,
            'source_url':     url,
            'scraped_at':     pd.Timestamp.now().isoformat(),
            'source':         'inmuebles24',
        }
    except Exception:
        return None


def _parse_price(text: str) -> float:
    if not text:
        return np.nan
    # Eliminar MXN, USD, $, puntos de miles, etc.
    nums = re.sub(r'[^\d]', '', text.replace(',', ''))
    return float(nums) if nums else np.nan


def _detect_owner(item) -> bool:
    text = item.get_text().lower()
    owner_kw  = ['propietario', 'particular', 'dueño', 'owner', 'directo']
    agency_kw = ['inmobiliaria', 'agencia', 'bienes raíces', 'realty', 'century',
                 'remax', 'coldwell', 'era ', 'sotheby', 'kw ', 'keller', 'habi ',
                 'lamudi', 'inmuebles24']
    if any(kw in text for kw in owner_kw):
        return True
    if any(kw in text for kw in agency_kw):
        return False
    return False


def _detect_property_type(text: str) -> str:
    text = text.lower()
    if any(w in text for w in ['departamento', 'depto', 'depa', 'apartamento', 'piso']): return 'departamento'
    if any(w in text for w in ['casa', 'residencia', 'chalet']): return 'casa'
    if 'oficina' in text: return 'oficina'
    if any(w in text for w in ['local', 'comercial', 'tienda', 'plaza']): return 'local_comercial'
    if any(w in text for w in ['bodega', 'nave', 'industrial', 'almacén']): return 'bodega_industrial'
    if 'terreno' in text: return 'terreno'
    if 'estudio' in text: return 'estudio'
    return 'otro'


def _extract_attrs(item, text: str = '') -> dict:
    attrs = {}
    if not text:
        text = item.get_text(' ', strip=True)

    # m² — área construida
    m2 = re.search(r'(\d[\d,]*(?:\.\d+)?)\s*m[²2]\s*(?:construidos?|totales?|de\s+construcci[oó]n)?', text, re.I)
    if m2:
        attrs['area_m2'] = float(m2.group(1).replace(',', ''))

    # Recámaras / habitaciones
    rec = re.search(r'(\d+)\s*(?:rec[áa]?maras?|habitaciones?|cuartos?|dormitorios?)', text, re.I)
    if rec:
        attrs['bedrooms'] = int(rec.group(1))

    # Baños
    ban = re.search(r'(\d+(?:\.\d+)?)\s*ba[ñn]os?', text, re.I)
    if ban:
        attrs['bathrooms'] = float(ban.group(1))

    # Estacionamiento / cajones
    est = re.search(r'(\d+)\s*(?:caj[oó]n(?:es)?|estacionamiento|parking|garaje|cochera)', text, re.I)
    if est:
        attrs['parking'] = int(est.group(1))

    # También buscar iconos de features en elementos específicos
    for feat in item.find_all(class_=re.compile(r'feature|amenity|spec', re.I)):
        ft = feat.get_text(' ', strip=True).lower()
        if not attrs.get('bedrooms'):
            rm = re.search(r'(\d+)\s*rec', ft)
            if rm: attrs['bedrooms'] = int(rm.group(1))
        if not attrs.get('bathrooms'):
            bm = re.search(r'(\d+(?:\.\d+)?)\s*ba', ft)
            if bm: attrs['bathrooms'] = float(bm.group(1))
        if not attrs.get('area_m2'):
            am = re.search(r'(\d[\d,]*)\s*m[²2]', ft)
            if am: attrs['area_m2'] = float(am.group(1).replace(',', ''))

    return attrs


def check_credits():
    try:
        info = requests.get(
            f'http://api.scraperapi.com/account?api_key={SCRAPERAPI_KEY}', timeout=10
        ).json()
        used  = info.get('requestCount', '?')
        limit = info.get('requestLimit', '?')
        print(f'  ScraperAPI: {used} / {limit} créditos usados')
    except Exception:
        pass


def diagnose_page(html: str, page_num: int):
    """Imprime diagnóstico cuando no se encuentran listings."""
    soup = BeautifulSoup(html, 'lxml')
    title = soup.find('title')
    print(f'    [debug] título: {title.get_text(strip=True)[:80] if title else "N/A"}')
    # ¿Hay NEXT_DATA?
    for sc in soup.find_all('script'):
        if '__NEXT_DATA__' in (sc.string or ''):
            print('    [debug] __NEXT_DATA__ encontrado')
            break
    # ¿Cuántas <a> hay?
    links = soup.find_all('a', href=re.compile(r'/(propiedad|inmueble|departamento|casa)-', re.I))
    print(f'    [debug] links de propiedad: {len(links)}')
    # Guardar HTML para inspeccion manual (solo página 1)
    if page_num == 1:
        debug_path = OUT_CSV.replace('.csv', '_debug_p1.html')
        with open(debug_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f'    [debug] HTML guardado: {debug_path}')


def scrape_via_links(html: str) -> list[dict]:
    """
    Fallback: extrae URLs de propiedades del HTML y las scrapeá individualmente.
    Se usa cuando el listado no tiene cards parseables.
    Limitado a 10 URLs por página para ahorrar créditos.
    """
    soup = BeautifulSoup(html, 'lxml')
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        full = href if href.startswith('http') else BASE_URL + href
        if re.search(r'/(propiedad|inmueble|departamento|casa|oficina)-\d+', full, re.I):
            links.append(full)
    links = list(dict.fromkeys(links))[:10]  # dedup + limit

    results = []
    for url in links:
        try:
            resp = scraperapi(url, render=False)  # detalle no necesita render
            if resp.status_code == 200:
                r = _parse_detail_page(resp.text, url)
                if r:
                    results.append(r)
            time.sleep(1)
        except Exception:
            pass
    return results


def _parse_detail_page(html: str, url: str) -> dict | None:
    """Parsea una página de detalle de propiedad."""
    soup = BeautifulSoup(html, 'lxml')
    try:
        # JSON-LD
        for jld in soup.find_all('script', type='application/ld+json'):
            data = json.loads(jld.string or '')
            if not isinstance(data, list):
                data = [data]
            for item in data:
                if item.get('@type') in ('Apartment', 'House', 'RealEstateListing', 'Product'):
                    r = _parse_jsonld(item)
                    if r:
                        r['source_url'] = url
                        return r

        # Fallback: título + precio de la página
        title_el = soup.find('h1')
        title = title_el.get_text(strip=True) if title_el else ''
        price_el = soup.find(class_=re.compile(r'price|precio', re.I))
        price_text = price_el.get_text(strip=True) if price_el else ''
        price = _parse_price(price_text)
        if not price:
            return None
        desc_el = soup.find('meta', attrs={'name': 'description'})
        description = desc_el.get('content', '') if desc_el else ''
        text = soup.get_text(' ', strip=True)
        attrs = _extract_attrs(soup, text)

        return {
            'name':           title,
            'phone':          np.nan,
            'email':          np.nan,
            'address':        '',
            'colonia':        np.nan,
            'price':          price,
            'operation_type': 'renta',
            'property_type':  _detect_property_type(title + ' ' + description),
            'bedrooms':       attrs.get('bedrooms', np.nan),
            'bathrooms':      attrs.get('bathrooms', np.nan),
            'parking':        attrs.get('parking', np.nan),
            'area_m2':        attrs.get('area_m2', np.nan),
            'terrain_m2':     np.nan,
            'floor':          np.nan,
            'age_years':      np.nan,
            'furnished':      np.nan,
            'has_pool':       np.nan,
            'has_gym':        np.nan,
            'description':    description,
            'is_owner':       False,
            'score':          0,
            'source_url':     url,
            'scraped_at':     pd.Timestamp.now().isoformat(),
            'source':         'inmuebles24',
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',    action='store_true', help='Solo 2 páginas')
    parser.add_argument('--no-render', action='store_true', help='Sin JS render (más rápido, menos créditos)')
    parser.add_argument('--fallback-links', action='store_true', help='Activar fallback de scraping por links individuales')
    args = parser.parse_args()

    if not SCRAPERAPI_KEY:
        print('✗ SCRAPERAPI_KEY no configurada.')
        print('  Corre: export SCRAPERAPI_KEY=tu_key')
        return

    print(f'✓ ScraperAPI key: {SCRAPERAPI_KEY[:8]}...')
    check_credits()

    use_render = not args.no_render
    max_pages  = 2 if args.test else MAX_PAGES
    all_listings = []

    print(f'\nScrapeando Inmuebles24 — renta Monterrey NL ({max_pages} páginas, render={use_render})...\n')

    for page in range(1, max_pages + 1):
        url = SEARCH_URL.format(page=page)
        print(f'  Página {page:02d}/{max_pages} ... ', end='', flush=True)

        try:
            resp = scraperapi(url, render=use_render)

            if resp.status_code != 200:
                print(f'✗ HTTP {resp.status_code}')
                time.sleep(4)
                continue

            listings = extract_listings_from_page(resp.text)

            if not listings:
                diagnose_page(resp.text, page)
                if args.fallback_links:
                    print('  → intentando fallback por links individuales...')
                    listings = scrape_via_links(resp.text)

            if not listings:
                print(f'✗ Sin listings — posible última página o bloqueo')
                if page > 2:
                    break
                time.sleep(4)
                continue

            # Dedup dentro de la página actual
            seen = {r['source_url'] for r in all_listings}
            new = [r for r in listings if r['source_url'] not in seen]
            all_listings.extend(new)
            print(f'✓ {len(new)} nuevos  (total: {len(all_listings)})')

        except Exception as e:
            print(f'✗ Error: {e}')

        time.sleep(SLEEP_SEC)

    if not all_listings:
        print('\n✗ No se obtuvieron listings.')
        print('  Sugerencias:')
        print('  1. Prueba: python3 scraper_inmuebles24.py --test --no-render')
        print('  2. Inspecciona el HTML: revisa *_debug_p1.html')
        print('  3. Activa fallback: --fallback-links')
        return

    df = pd.DataFrame(all_listings)
    df = df.drop_duplicates(subset=['source_url'])
    df = df[df['price'].notna() & (df['price'] > 0)]
    df.to_csv(OUT_CSV, index=False)

    print(f'\n{"="*55}')
    print('RESUMEN INMUEBLES24')
    print(f'{"="*55}')
    print(f'  Páginas scrapeadas : {page}')
    print(f'  Listings totales   : {len(df):,}')
    print(f'  Con precio         : {df["price"].notna().sum():,}')
    print(f'  Propietarios direc.: {df["is_owner"].sum():,}')
    print(f'  Tipos de propiedad :')
    for t, n in df['property_type'].value_counts().items():
        print(f'    {t:<22}: {n}')
    print(f'\n  ✓ CSV: {OUT_CSV}')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
