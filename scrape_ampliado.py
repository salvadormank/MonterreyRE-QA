"""
Extrae teléfonos buscando entre los top-30 por segmento hasta conseguir 10 válidos.
Reintentos para HTTP 500 y salta los HTTP 410 (expirados).
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
OUT_CSV  = "../analisis_propiedades/contactos_segmentos_v2.csv"
GOAL     = 10   # teléfonos válidos por segmento
TOP_N    = 40   # candidatos a intentar por segmento

PHONE_RE = re.compile(
    r'\b(?:\+?52[-.\s]?)?(?:\(?\d{2,3}\)?[-.\s]?)?\d{4}[-.\s]?\d{4}\b'
    r'|\b\d{10}\b'
    r'|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
)


def scraperapi_url(target: str) -> str:
    return (
        f"http://api.scraperapi.com?api_key={SCRAPERAPI_KEY}"
        f"&url={target}&render=true&country_code=mx"
    )


def get_candidates():
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
    top_s = df[mask_s][cols].sort_values('userViews', ascending=False).head(TOP_N).copy()
    top_s['segmento'] = 'Estudiantes'
    top_a = df[mask_a][cols].sort_values('userViews', ascending=False).head(TOP_N).copy()
    top_a['segmento'] = 'Adulto mayor (proxy)'

    combined = pd.concat([top_s, top_a], ignore_index=True)
    combined['full_url'] = BASE_URL + combined['url'].astype(str)
    return combined


def extract_phone(target_url: str, retries: int = 2) -> tuple[str, int]:
    """Devuelve (telefono_o_error, http_status)."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(scraperapi_url(target_url), timeout=70)

            if resp.status_code == 410:
                return 'Expirado (410)', 410

            if resp.status_code != 200:
                if attempt < retries:
                    time.sleep(5)
                    continue
                return f'Error HTTP {resp.status_code}', resp.status_code

            soup = BeautifulSoup(resp.text, 'lxml')

            for tag in soup.find_all(attrs={'data-phone': True}):
                m = PHONE_RE.search(tag['data-phone'])
                if m:
                    digits = re.sub(r'\D', '', m.group())
                    if len(digits) >= 10:
                        return m.group().strip(), 200

            for tag in soup.find_all(attrs={'data-qa': re.compile(r'phone', re.I)}):
                text = tag.get_text()
                m = PHONE_RE.search(text)
                if m:
                    digits = re.sub(r'\D', '', m.group())
                    if len(digits) >= 10:
                        return m.group().strip(), 200

            for m in PHONE_RE.finditer(resp.text):
                digits = re.sub(r'\D', '', m.group())
                if 10 <= len(digits) <= 13:
                    return m.group().strip(), 200

            return 'Sin teléfono visible', 200

        except requests.Timeout:
            if attempt < retries:
                time.sleep(5)
                continue
            return 'Error: timeout', -1
        except Exception as e:
            return f'Error: {str(e)[:50]}', -1

    return 'Error: max reintentos', -1


def is_valid_phone(phone: str) -> bool:
    return bool(re.search(r'\d{10}', re.sub(r'\D', '', phone)))


def main():
    if not SCRAPERAPI_KEY:
        print('✗ SCRAPERAPI_KEY no encontrada.')
        print('  Corre: export SCRAPERAPI_KEY=tu_key')
        return

    print(f'✓ ScraperAPI key: {SCRAPERAPI_KEY[:8]}...')
    try:
        info = requests.get(
            f'http://api.scraperapi.com/account?api_key={SCRAPERAPI_KEY}', timeout=10
        ).json()
        print(f'  Créditos usados: {info.get("requestCount","?")} / {info.get("requestLimit","?")}')
    except Exception:
        pass

    df = get_candidates()
    print(f'\nCandidatos disponibles: {len(df)} ({TOP_N} por segmento)')
    print(f'Meta: {GOAL} teléfonos válidos por segmento\n')

    results = []

    for seg in ['Estudiantes', 'Adulto mayor (proxy)']:
        subset = df[df['segmento'] == seg].reset_index(drop=True)
        found  = 0
        seg_results = []

        print(f'\n── {seg} ──')

        for i, row in subset.iterrows():
            if found >= GOAL:
                break

            pub   = row['publisher']
            price = row['price']
            url   = row['full_url']
            n     = i + 1

            print(f'  [{n:02d}/{TOP_N}] {pub[:30]:<30} ${price:>7,.0f}  ', end='', flush=True)

            phone, status = extract_phone(url)

            if status == 410:
                print('✗ expirado')
                seg_results.append({**row.to_dict(), 'telefono': phone})
                time.sleep(1)
                continue

            if is_valid_phone(phone):
                found += 1
                print(f'✓ {phone}  ({found}/{GOAL})')
            else:
                print(f'✗ {phone}')

            seg_results.append({**row.to_dict(), 'telefono': phone})
            time.sleep(2)

        results.extend(seg_results)
        print(f'\n  → Encontrados: {found}/{GOAL}')

    df_out = pd.DataFrame(results)
    out_cols = ['segmento', 'address', 'location', 'price',
                'publisher', 'publisherCode', 'telefono', 'full_url']
    df_out[out_cols].to_csv(OUT_CSV, index=False)

    print('\n' + '='*60)
    print('RESUMEN FINAL')
    print('='*60)
    for seg in ['Estudiantes', 'Adulto mayor (proxy)']:
        sub = df_out[df_out['segmento'] == seg]
        ok  = sub['telefono'].apply(is_valid_phone).sum()
        print(f'{seg}: {ok} teléfonos válidos / {len(sub)} intentados')

    print(f'\n✓ CSV guardado: {OUT_CSV}')

    validos = df_out[df_out['telefono'].apply(is_valid_phone)]
    print('\n── Teléfonos encontrados ──')
    print(validos[['segmento', 'location', 'price', 'publisher', 'telefono']].to_string(index=False))


if __name__ == '__main__':
    main()
