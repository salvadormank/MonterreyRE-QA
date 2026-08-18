"""
scraper_mercadolibre.py
─────────────────────────────────────────────────────────────────────────────
Scraper de MercadoLibre / MetrosCúbicos para propiedades en renta en Monterrey.
Usa la API oficial de MercadoLibre (no necesita ScraperAPI).

Cómo obtener el access_token:
  1. Crea app en https://developers.mercadolibre.com.mx/devcenter
  2. Guarda Client ID y Client Secret en .env:
        ML_CLIENT_ID=tu_client_id
        ML_CLIENT_SECRET=tu_client_secret
  3. Primera vez — corre con --auth para abrir el flujo OAuth:
        python3 scraper_mercadolibre.py --auth
  4. Después:
        python3 scraper_mercadolibre.py --test
        python3 scraper_mercadolibre.py

Categorías Inmuebles México (MLM):
  MLM1459  Inmuebles (raíz)
  MLM1472  Departamentos en Renta
  MLM1476  Casas en Renta
  MLM1500  Oficinas en Renta
─────────────────────────────────────────────────────────────────────────────
"""

import os, re, time, json, argparse, webbrowser
from urllib.parse import urlencode
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv('./.env')

ML_CLIENT_ID     = os.environ.get('ML_CLIENT_ID', '')
ML_CLIENT_SECRET = os.environ.get('ML_CLIENT_SECRET', '')
ML_REDIRECT_URI  = 'http://localhost:8765'
TOKEN_FILE       = './.ml_token.json'
OUT_CSV          = './mercadolibre_raw.csv'

# Nuevo León state ID en MercadoLibre
NL_STATE_ID = 'TUxNUE5VRTc3MDU'  # Nuevo León

# Categorías a scraper (por separado para mayor cobertura)
CATEGORIES = [
    ('MLM1472', 'departamentos_renta'),
    ('MLM1476', 'casas_renta'),
]

API_BASE    = 'https://api.mercadolibre.com'
LIMIT       = 50   # max por request en ML API
MAX_OFFSET  = 1000  # ML limita a 1000 resultados por búsqueda


# ── OAuth ─────────────────────────────────────────────────────────────────

_auth_code = None

class _OAuthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global _auth_code
        qs = self.path.split('?', 1)[-1] if '?' in self.path else ''
        params = dict(p.split('=') for p in qs.split('&') if '=' in p)
        _auth_code = params.get('code', '')
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'<h2>Autenticacion exitosa. Puedes cerrar esta ventana.</h2>')

    def log_message(self, *args):
        pass


def get_auth_code() -> str:
    """Abre navegador para OAuth y captura el code en servidor local."""
    global _auth_code
    _auth_code = None

    server = HTTPServer(('localhost', 8765), _OAuthHandler)
    t = threading.Thread(target=server.handle_request)
    t.daemon = True
    t.start()

    params = {
        'response_type': 'code',
        'client_id':     ML_CLIENT_ID,
        'redirect_uri':  ML_REDIRECT_URI,
    }
    auth_url = f'https://auth.mercadolibre.com.mx/authorization?{urlencode(params)}'
    print(f'\n  Abriendo navegador para autenticación...')
    print(f'  Si no abre automáticamente, visita:\n  {auth_url}\n')
    webbrowser.open(auth_url)

    t.join(timeout=120)
    return _auth_code or ''


def exchange_code(code: str) -> dict:
    """Intercambia auth code por access_token."""
    resp = requests.post(
        f'{API_BASE}/oauth/token',
        data={
            'grant_type':    'authorization_code',
            'client_id':     ML_CLIENT_ID,
            'client_secret': ML_CLIENT_SECRET,
            'code':          code,
            'redirect_uri':  ML_REDIRECT_URI,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def refresh_token(refresh_tok: str) -> dict:
    """Renueva el access_token usando el refresh_token."""
    resp = requests.post(
        f'{API_BASE}/oauth/token',
        data={
            'grant_type':    'refresh_token',
            'client_id':     ML_CLIENT_ID,
            'client_secret': ML_CLIENT_SECRET,
            'refresh_token': refresh_tok,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def load_token() -> dict | None:
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            return json.load(f)
    return None


def save_token(token_data: dict):
    with open(TOKEN_FILE, 'w') as f:
        json.dump(token_data, f, indent=2)


def get_valid_token() -> str:
    """Retorna access_token válido, renovando si es necesario."""
    tok = load_token()
    if not tok:
        raise RuntimeError('No hay token guardado. Corre con --auth primero.')

    # Intentar renovar (los tokens duran 6h, refresh 6 meses)
    try:
        new_tok = refresh_token(tok['refresh_token'])
        save_token(new_tok)
        return new_tok['access_token']
    except Exception:
        return tok['access_token']


# ── API calls ─────────────────────────────────────────────────────────────

def search_properties(access_token: str, category: str, offset: int = 0) -> dict:
    """Busca propiedades en renta en NL."""
    params = {
        'category':   category,
        'state':      NL_STATE_ID,
        'limit':      LIMIT,
        'offset':     offset,
        # Filtros de renta
        'OPERATION':  '242073',   # 242073 = Alquiler en ML MX
    }
    headers = {'Authorization': f'Bearer {access_token}'}
    resp = requests.get(
        f'{API_BASE}/sites/MLM/search',
        params=params,
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def get_item_details(access_token: str, item_id: str) -> dict:
    """Obtiene detalles completos de un item."""
    headers = {'Authorization': f'Bearer {access_token}'}
    resp = requests.get(
        f'{API_BASE}/items/{item_id}',
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def parse_item(item: dict, details: dict = None) -> dict:
    """Convierte un item de MercadoLibre a nuestro esquema."""
    d = details or {}
    attrs_list = d.get('attributes', item.get('attributes', []))
    attrs = {a['id']: a.get('value_name') or a.get('value_struct', {}) for a in attrs_list}

    def attr_float(key):
        v = attrs.get(key)
        if v is None:
            return np.nan
        if isinstance(v, dict):
            v = v.get('number', v.get('unit', ''))
        try:
            return float(re.sub(r'[^\d.]', '', str(v)) or 0) or np.nan
        except Exception:
            return np.nan

    def attr_int(key):
        v = attr_float(key)
        return int(v) if v == v else np.nan

    # Precio
    price = float(item.get('price') or 0) or np.nan

    # Título y descripción
    title = item.get('title', '')
    desc  = d.get('plain_text', '') or ''

    # Dirección
    address_data = item.get('location', {}) or {}
    neighborhood = (address_data.get('neighborhood') or {}).get('name', '')
    city         = (address_data.get('city') or {}).get('name', '')
    address      = f'{neighborhood}, {city}'.strip(', ')

    # URL del listing
    permalink = item.get('permalink', '')

    # Atributos inmobiliarios conocidos en ML
    bedrooms  = attr_int('BEDROOMS')
    bathrooms = attr_float('FULL_BATHROOMS') or attr_float('BATHROOMS')
    parking   = attr_int('PARKING_LOTS')
    area_m2   = attr_float('COVERED_AREA') or attr_float('TOTAL_AREA')

    # Propietario vs agencia
    seller_type = item.get('seller', {}).get('seller_reputation', {}).get('level_id', '')
    is_owner    = 'not_' not in str(seller_type)  # heurística

    return {
        'name':           title,
        'phone':          np.nan,
        'email':          np.nan,
        'address':        address,
        'colonia':        neighborhood or np.nan,
        'price':          price,
        'operation_type': 'renta',
        'property_type':  _detect_property_type(title + ' ' + desc),
        'bedrooms':       bedrooms,
        'bathrooms':      bathrooms,
        'parking':        parking,
        'area_m2':        area_m2,
        'terrain_m2':     np.nan,
        'floor':          np.nan,
        'age_years':      np.nan,
        'furnished':      np.nan,
        'has_pool':       np.nan,
        'has_gym':        np.nan,
        'description':    desc[:500],
        'is_owner':       is_owner,
        'score':          0,
        'source_url':     permalink,
        'scraped_at':     pd.Timestamp.now().isoformat(),
        'source':         'mercadolibre',
    }


def _detect_property_type(text: str) -> str:
    text = text.lower()
    if any(w in text for w in ['departamento', 'depto', 'depa', 'apartamento']): return 'departamento'
    if any(w in text for w in ['casa', 'residencia']): return 'casa'
    if 'oficina' in text: return 'oficina'
    if any(w in text for w in ['local', 'comercial', 'tienda']): return 'local_comercial'
    if any(w in text for w in ['bodega', 'nave', 'industrial']): return 'bodega_industrial'
    if 'terreno' in text: return 'terreno'
    return 'otro'


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--auth',    action='store_true', help='Iniciar flujo OAuth')
    parser.add_argument('--test',    action='store_true', help='Solo 50 resultados por categoría')
    parser.add_argument('--details', action='store_true', help='Obtener detalles de cada item (más lento, mejor data)')
    args = parser.parse_args()

    if not ML_CLIENT_ID or not ML_CLIENT_SECRET:
        print('✗ ML_CLIENT_ID o ML_CLIENT_SECRET no configurados.')
        print('  Agrega al archivo .env:')
        print('    ML_CLIENT_ID=tu_client_id')
        print('    ML_CLIENT_SECRET=tu_client_secret')
        print('\n  Obtén credenciales en: https://developers.mercadolibre.com.mx/devcenter')
        return

    # ── Autenticación ─────────────────────────────────────────────────────
    if args.auth:
        print('Iniciando flujo OAuth de MercadoLibre...')
        code = get_auth_code()
        if not code:
            print('✗ No se recibió code de autorización.')
            return
        token_data = exchange_code(code)
        save_token(token_data)
        print(f'✓ Token guardado en {TOKEN_FILE}')
        print(f'  Expira en: {token_data.get("expires_in", "?")} segundos')
        return

    # ── Scraping ──────────────────────────────────────────────────────────
    try:
        access_token = get_valid_token()
    except RuntimeError as e:
        print(f'✗ {e}')
        return

    all_listings = []

    for category, cat_label in CATEGORIES:
        print(f'\n  Categoría: {cat_label} ({category})')
        offset = 0
        cat_count = 0

        while True:
            try:
                data   = search_properties(access_token, category, offset)
                items  = data.get('results', [])
                total  = data.get('paging', {}).get('total', 0)

                if not items:
                    break

                print(f'    offset={offset:4d}  items={len(items)}  total={total}', end='  ')

                for item in items:
                    details = {}
                    if args.details:
                        try:
                            details = get_item_details(access_token, item['id'])
                            time.sleep(0.3)
                        except Exception:
                            pass
                    r = parse_item(item, details)
                    if r.get('price'):
                        all_listings.append(r)
                        cat_count += 1

                print(f'✓ acumulado cat: {cat_count}')
                offset += LIMIT

                if args.test or offset >= MAX_OFFSET or offset >= total:
                    break

                time.sleep(0.5)

            except requests.HTTPError as e:
                print(f'\n✗ HTTP {e.response.status_code}: {e.response.text[:200]}')
                if e.response.status_code == 401:
                    print('  Token expirado. Corre: python3 scraper_mercadolibre.py --auth')
                break
            except Exception as e:
                print(f'\n✗ Error: {e}')
                break

    if not all_listings:
        print('\n✗ No se obtuvieron listings.')
        return

    df = pd.DataFrame(all_listings)
    df = df.drop_duplicates(subset=['source_url'])
    df = df[df['price'].notna() & (df['price'] > 0)]
    df.to_csv(OUT_CSV, index=False)

    print(f'\n{"="*55}')
    print('RESUMEN MERCADOLIBRE / METROS CÚBICOS')
    print(f'{"="*55}')
    print(f'  Listings totales   : {len(df):,}')
    print(f'  Con precio         : {df["price"].notna().sum():,}')
    print(f'  Tipos de propiedad :')
    for t, n in df['property_type'].value_counts().items():
        print(f'    {t:<22}: {n}')
    print(f'\n  ✓ CSV: {OUT_CSV}')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
