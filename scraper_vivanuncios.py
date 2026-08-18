"""
scraper_vivanuncios.py
─────────────────────────────────────────────────────────────────────────────
Scraper de Vivanuncios.com.mx para propiedades en renta en Monterrey.
Usa ScraperAPI con render=true para obtener el contenido renderizado por JS.

Output: vivanuncios_raw.csv  (mismo formato que leads_lamudi_renta.csv)

Uso:
  export SCRAPERAPI_KEY=tu_key
  python3 scraper_vivanuncios.py --test     # 2 páginas
  python3 scraper_vivanuncios.py            # completo (~50 páginas)
─────────────────────────────────────────────────────────────────────────────
"""

import os, re, time, json, argparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

SCRAPERAPI_KEY = os.environ.get('SCRAPERAPI_KEY', '')
BASE_URL   = 'https://www.vivanuncios.com.mx'
SEARCH_URL = 'https://www.vivanuncios.com.mx/s-renta-inmuebles/monterrey/v1c1096l10601p{page}'
OUT_CSV    = './vivanuncios_raw.csv'
MAX_PAGES  = 50
SLEEP_SEC  = 2

PHONE_RE = re.compile(
    r'\b(?:\+?52[-.\s]?)?(?:\(?\d{2,3}\)?[-.\s]?)?\d{4}[-.\s]?\d{4}\b'
    r'|\b\d{10}\b'
    r'|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
)

def scraperapi(url: str, render: bool = True) -> requests.Response:
    params = {
        'api_key':    SCRAPERAPI_KEY,
        'url':        url,
        'render':     'true' if render else 'false',
        'country_code': 'mx',
    }
    return requests.get('http://api.scraperapi.com', params=params, timeout=90)


def extract_listings_from_page(html: str) -> list[dict]:
    """Extrae listings del HTML renderizado de una página de resultados."""
    soup = BeautifulSoup(html, 'lxml')
    results = []

    # Vivanuncios (OLX platform) — selectores principales
    # Los listings están en <li> con data-aut-id="itemBox" o similar
    items = (
        soup.find_all('li',  attrs={'data-aut-id': 'itemBox'}) or
        soup.find_all('li',  attrs={'data-aut-id': 'adItem'}) or
        soup.find_all('div', attrs={'data-aut-id': 'adItem'}) or
        soup.find_all(attrs={'data-testid': 'listing-ad-item'}) or
        soup.find_all('article', class_=re.compile(r'(item|listing|ad)', re.I)) or
        soup.find_all('li',  class_=re.compile(r'(item|listing|ad)', re.I))
    )

    # Fallback: buscar por JSON-LD de listings
    if not items:
        json_lds = soup.find_all('script', type='application/ld+json')
        for jld in json_lds:
            try:
                data = json.loads(jld.string or '')
                if isinstance(data, list):
                    for item in data:
                        if item.get('@type') in ('Apartment','House','RealEstateListing','Product'):
                            results.append(_parse_jsonld(item))
                elif data.get('@type') in ('ItemList',):
                    for elem in data.get('itemListElement', []):
                        results.append(_parse_jsonld(elem.get('item', elem)))
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


def _parse_html_item(item) -> dict:
    """Extrae datos de un elemento HTML de listing."""
    # Título / nombre
    title_el = (
        item.find(attrs={'data-aut-id': 'itemTitle'}) or
        item.find(['h2','h3'], class_=re.compile(r'title|name', re.I)) or
        item.find('a', class_=re.compile(r'title|name', re.I))
    )
    title = title_el.get_text(strip=True) if title_el else ''

    # Precio
    price_el = (
        item.find(attrs={'data-aut-id': 'itemPrice'}) or
        item.find(class_=re.compile(r'price|precio', re.I))
    )
    price_text = price_el.get_text(strip=True) if price_el else ''
    price = _parse_price(price_text)

    # URL del listing
    link_el = item.find('a', href=True)
    url = ''
    if link_el:
        href = link_el['href']
        url = href if href.startswith('http') else BASE_URL + href

    # Descripción
    desc_el = (
        item.find(attrs={'data-aut-id': 'itemDetails'}) or
        item.find(class_=re.compile(r'desc|body|details', re.I))
    )
    description = desc_el.get_text(strip=True) if desc_el else ''

    # Ubicación
    loc_el = (
        item.find(attrs={'data-aut-id': 'item-location'}) or
        item.find(class_=re.compile(r'location|colonia|zona|ubicacion', re.I))
    )
    address = loc_el.get_text(strip=True) if loc_el else ''

    # Tipo de anunciante (propietario vs agencia)
    is_owner = _detect_owner(item)

    # Atributos (m², recámaras, etc.)
    attrs = _extract_attrs(item)

    return {
        'name':          title,
        'phone':         np.nan,
        'email':         np.nan,
        'address':       address,
        'colonia':       np.nan,
        'price':         price,
        'operation_type': 'renta',
        'property_type': _detect_property_type(title + ' ' + description),
        'bedrooms':      attrs.get('bedrooms', np.nan),
        'bathrooms':     attrs.get('bathrooms', np.nan),
        'parking':       attrs.get('parking', np.nan),
        'area_m2':       attrs.get('area_m2', np.nan),
        'terrain_m2':    np.nan,
        'floor':         np.nan,
        'age_years':     np.nan,
        'furnished':     np.nan,
        'has_pool':      np.nan,
        'has_gym':       np.nan,
        'description':   description,
        'is_owner':      is_owner,
        'score':         0,
        'source_url':    url,
        'scraped_at':    pd.Timestamp.now().isoformat(),
        'source':        'vivanuncios',
    }


def _parse_jsonld(item: dict) -> dict | None:
    try:
        price_data = item.get('offers', {})
        price = float(re.sub(r'[^\d.]', '', str(price_data.get('price', 0))) or 0)
        if not price:
            return None
        return {
            'name':          item.get('name', ''),
            'phone':         np.nan,
            'email':         np.nan,
            'address':       str(item.get('address', '')),
            'colonia':       np.nan,
            'price':         price,
            'operation_type': 'renta',
            'property_type': item.get('@type', '').lower(),
            'bedrooms':      item.get('numberOfRooms', np.nan),
            'bathrooms':     item.get('numberOfBathroomsTotal', np.nan),
            'parking':       np.nan,
            'area_m2':       item.get('floorSize', {}).get('value', np.nan),
            'terrain_m2':    np.nan,
            'floor':         np.nan,
            'age_years':     np.nan,
            'furnished':     np.nan,
            'has_pool':      np.nan,
            'has_gym':       np.nan,
            'description':   item.get('description', ''),
            'is_owner':      False,
            'score':         0,
            'source_url':    item.get('url', ''),
            'scraped_at':    pd.Timestamp.now().isoformat(),
            'source':        'vivanuncios',
        }
    except Exception:
        return None


def _parse_price(text: str) -> float:
    if not text:
        return np.nan
    nums = re.sub(r'[^\d]', '', text)
    return float(nums) if nums else np.nan


def _detect_owner(item) -> bool:
    """Detecta si el anunciante es propietario directo."""
    text = item.get_text().lower()
    owner_kw   = ['propietario','particular','dueño','owner','se renta directo']
    agency_kw  = ['inmobiliaria','agencia','bienes raíces','realty','century','remax',
                  'coldwell','era ','sotheby','kw ','keller']
    if any(kw in text for kw in owner_kw):
        return True
    if any(kw in text for kw in agency_kw):
        return False
    # Buscar badge/label específico de OLX
    badge = item.find(class_=re.compile(r'(professional|particular|badge|tag)', re.I))
    if badge:
        badge_text = badge.get_text().lower()
        if 'particular' in badge_text or 'propietario' in badge_text:
            return True
    return False


def _detect_property_type(text: str) -> str:
    text = text.lower()
    if any(w in text for w in ['departamento','depto','depa','apartamento']): return 'departamento'
    if any(w in text for w in ['casa','residencia']): return 'casa'
    if 'oficina' in text: return 'oficina'
    if any(w in text for w in ['local','comercial','tienda']): return 'local_comercial'
    if any(w in text for w in ['bodega','nave','industrial']): return 'bodega_industrial'
    if 'terreno' in text: return 'terreno'
    return 'otro'


def _extract_attrs(item) -> dict:
    attrs = {}
    text = item.get_text()

    # m²
    m2 = re.search(r'(\d+(?:\.\d+)?)\s*m[²2]', text)
    if m2: attrs['area_m2'] = float(m2.group(1))

    # Recámaras
    rec = re.search(r'(\d+)\s*rec[áa]?maras?', text, re.I)
    if rec: attrs['bedrooms'] = int(rec.group(1))

    # Baños
    ban = re.search(r'(\d+(?:\.\d+)?)\s*ba[ñn]os?', text, re.I)
    if ban: attrs['bathrooms'] = float(ban.group(1))

    # Estacionamiento
    est = re.search(r'(\d+)\s*(?:caj[oó]n|estacionamiento|parking)', text, re.I)
    if est: attrs['parking'] = int(est.group(1))

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Solo 2 páginas')
    args = parser.parse_args()

    if not SCRAPERAPI_KEY:
        print('✗ SCRAPERAPI_KEY no configurada.')
        print('  Corre: export SCRAPERAPI_KEY=tu_key')
        return

    print(f'✓ ScraperAPI key: {SCRAPERAPI_KEY[:8]}...')
    check_credits()

    max_pages = 2 if args.test else MAX_PAGES
    all_listings = []

    print(f'\nScrapeando Vivanuncios Monterrey — renta ({max_pages} páginas)...\n')

    for page in range(1, max_pages + 1):
        url = SEARCH_URL.format(page=page)
        print(f'  Página {page:02d}/{max_pages}  {url[-50:]} ... ', end='', flush=True)

        try:
            resp = scraperapi(url, render=True)

            if resp.status_code != 200:
                print(f'✗ HTTP {resp.status_code}')
                time.sleep(3)
                continue

            listings = extract_listings_from_page(resp.text)

            if not listings:
                print(f'✗ Sin listings (posible última página)')
                break

            all_listings.extend(listings)
            print(f'✓ {len(listings)} listings  (total: {len(all_listings)})')

        except Exception as e:
            print(f'✗ Error: {e}')

        time.sleep(SLEEP_SEC)

    if not all_listings:
        print('\n✗ No se obtuvieron listings. Verifica la key o la estructura del sitio.')
        return

    df = pd.DataFrame(all_listings)
    df = df.drop_duplicates(subset=['source_url'])
    df = df[df['price'].notna() & (df['price'] > 0)]
    df.to_csv(OUT_CSV, index=False)

    print(f'\n{"="*55}')
    print('RESUMEN VIVANUNCIOS')
    print(f'{"="*55}')
    print(f'  Páginas scrapeadas : {page}')
    print(f'  Listings totales   : {len(df):,}')
    print(f'  Con precio         : {df["price"].notna().sum():,}')
    print(f'  Propietarios direc.: {df["is_owner"].sum():,}')
    print(f'  Tipos de propiedad :')
    for t, n in df['property_type'].value_counts().items():
        print(f'    {t:<20}: {n}')
    print(f'\n  ✓ CSV: {OUT_CSV}')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
