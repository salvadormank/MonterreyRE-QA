"""
Extrae teléfonos de las 20 propiedades usando ScraperAPI con JS rendering.
"""

import os
import re
import time
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

SCRAPERAPI_KEY = os.environ.get('SCRAPERAPI_KEY', '')
BASE_URL = "https://www.inmuebles24.com"
OUT_CSV  = "../analisis_propiedades/contactos_segmentos.csv"

PHONE_RE = re.compile(
    r'\b(?:\+?52[-.\s]?)?(?:\(?\d{2,3}\)?[-.\s]?)?\d{4}[-.\s]?\d{4}\b'
    r'|\b\d{10}\b'
    r'|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
)


def scraperapi_url(target: str, render_js: bool = True) -> str:
    base = f"http://api.scraperapi.com?api_key={SCRAPERAPI_KEY}&url={target}"
    if render_js:
        base += "&render=true"
    base += "&country_code=mx"
    return base


def get_target_urls():
    df_raw = pd.read_excel('../propiedades_websights.xlsx')
    df = df_raw.copy()
    df['userViews'] = pd.to_numeric(df['userViews'], errors='coerce')
    df['address']   = df['address'].replace('Dirección no informada', np.nan)
    df = df[df['price'] <= 500_000].copy()

    texts = (df['description'].fillna('') + ' ' + df['longDescription'].fillna('')).str.lower()

    mask_s = texts.str.contains(
        r'estudiante|universitari|tec |tecnológico|tecnologico|itesm|uanl|udem|campus', na=False)
    mask_a = texts.str.contains(
        r'elevador|ascensor|planta baja|sin escalera|rampa|accesib|'
        r'seguridad 24|vigilancia 24|portero|guardia|tranquilo|tranquila', na=False) & ~mask_s

    cols = ['_id', 'address', 'location', 'price', 'publisher', 'publisherCode', 'url', 'userViews']
    top_s = df[mask_s][cols].sort_values('userViews', ascending=False).head(10).copy()
    top_s['segmento'] = 'Estudiantes'
    top_a = df[mask_a][cols].sort_values('userViews', ascending=False).head(10).copy()
    top_a['segmento'] = 'Adulto mayor (proxy)'

    combined = pd.concat([top_s, top_a], ignore_index=True)
    combined['full_url'] = BASE_URL + combined['url'].astype(str)
    return combined


def extract_phone(target_url: str) -> str:
    try:
        # Intento 1: con JS rendering (más lento, más completo)
        api_url = scraperapi_url(target_url, render_js=True)
        resp = requests.get(api_url, timeout=60)

        if resp.status_code != 200:
            return f'Error HTTP {resp.status_code}'

        soup = BeautifulSoup(resp.text, 'lxml')

        # Buscar en atributos data- donde suelen esconder el teléfono
        for tag in soup.find_all(attrs={'data-phone': True}):
            m = PHONE_RE.search(tag['data-phone'])
            if m:
                digits = re.sub(r'\D', '', m.group())
                if len(digits) >= 10:
                    return m.group().strip()

        for tag in soup.find_all(attrs={'data-qa': re.compile(r'phone', re.I)}):
            text = tag.get_text()
            m = PHONE_RE.search(text)
            if m:
                digits = re.sub(r'\D', '', m.group())
                if len(digits) >= 10:
                    return m.group().strip()

        # Buscar en todo el HTML renderizado
        full_text = resp.text
        for m in PHONE_RE.finditer(full_text):
            digits = re.sub(r'\D', '', m.group())
            if len(digits) >= 10:
                # Excluir IDs de propiedad (8 dígitos exactos del portal)
                if len(digits) != 8:
                    return m.group().strip()

        return 'No visible sin login'

    except requests.Timeout:
        return 'Error: timeout'
    except Exception as e:
        return f'Error: {str(e)[:50]}'


def main():
    if not SCRAPERAPI_KEY:
        print('✗ SCRAPERAPI_KEY no encontrada.')
        print('  Corre: export SCRAPERAPI_KEY=tu_key')
        return

    print(f'✓ ScraperAPI key: {SCRAPERAPI_KEY[:8]}...')

    # Verificar créditos disponibles
    try:
        info = requests.get(
            f'http://api.scraperapi.com/account?api_key={SCRAPERAPI_KEY}',
            timeout=10
        ).json()
        print(f'  Créditos usados: {info.get("requestCount", "?")} / {info.get("requestLimit", "?")}')
    except Exception:
        pass

    df = get_target_urls()
    print(f'\nPropiedades a visitar: {len(df)} (10 estudiantes + 10 adulto mayor)\n')

    phones = []
    for i, row in df.iterrows():
        idx   = len(phones) + 1
        seg   = row['segmento']
        pub   = row['publisher']
        price = row['price']
        url   = row['full_url']

        print(f'[{idx:02d}/20] {seg[:8]} | {pub[:28]} | ${price:,.0f} ...')
        phone = extract_phone(url)
        phones.append(phone)

        status = '✓' if re.search(r'\d{10}', re.sub(r'\D', '', phone)) else '✗'
        print(f'         {status} {phone}')

        time.sleep(2)

    df['telefono'] = phones
    out_cols = ['segmento', 'address', 'location', 'price',
                'publisher', 'publisherCode', 'telefono', 'full_url']
    df[out_cols].to_csv(OUT_CSV, index=False)

    print('\n' + '='*55)
    for seg in ['Estudiantes', 'Adulto mayor (proxy)']:
        sub = df[df['segmento'] == seg]
        ok  = sub['telefono'].apply(lambda x: bool(re.search(r'\d{10}', re.sub(r'\D','',x)))).sum()
        print(f'{seg}: {ok}/{len(sub)} teléfonos encontrados')

    print(f'\n✓ CSV guardado: {OUT_CSV}')
    print('\n' + df[['segmento','location','price','publisher','telefono']].to_string(index=False))


if __name__ == '__main__':
    main()
