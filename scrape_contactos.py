"""
Extrae teléfono de las 20 propiedades top (10 estudiantes + 10 adultos mayores)
visitando cada página de inmuebles24.com con Playwright y haciendo click en
el botón "Ver teléfono" que lo revela.
"""

import asyncio
import re
import pandas as pd
import numpy as np
from playwright.async_api import async_playwright

BASE_URL = "https://www.inmuebles24.com"
OUT_CSV  = "../analisis_propiedades/contactos_segmentos.csv"

# ── Construir las 20 URLs objetivo ────────────────────────────────
def get_target_urls():
    import pandas as pd
    import numpy as np

    df_raw = pd.read_excel('../propiedades_websights.xlsx')
    df = df_raw.copy()
    df['userViews'] = pd.to_numeric(df['userViews'], errors='coerce')
    df['address']   = df['address'].replace('Dirección no informada', np.nan)
    df = df[df['price'] <= 500_000].copy()

    texts = (df['description'].fillna('') + ' ' + df['longDescription'].fillna('')).str.lower()

    kw_s = r'estudiante|universitari|tec |tecnológico|tecnologico|itesm|uanl|udem|campus'
    kw_a = r'elevador|ascensor|planta baja|sin escalera|rampa|accesib|seguridad 24|vigilancia 24|portero|guardia|tranquilo|tranquila'

    mask_s = texts.str.contains(kw_s, na=False)
    mask_a = texts.str.contains(kw_a, na=False) & ~mask_s

    cols = ['_id', 'address', 'location', 'price', 'publisher', 'publisherCode', 'url', 'userViews']

    top_s = df[mask_s][cols].sort_values('userViews', ascending=False).head(10).copy()
    top_s['segmento'] = 'Estudiantes'

    top_a = df[mask_a][cols].sort_values('userViews', ascending=False).head(10).copy()
    top_a['segmento'] = 'Adulto mayor (proxy)'

    combined = pd.concat([top_s, top_a], ignore_index=True)
    combined['full_url'] = BASE_URL + combined['url'].astype(str)
    return combined


# ── Extracción de teléfono con Playwright ─────────────────────────
PHONE_RE = re.compile(
    r'\b(?:\+?52[-.\s]?)?(?:\(?\d{2,3}\)?[-.\s]?)?\d{4}[-.\s]?\d{4}\b'
    r'|\b\d{10}\b'
    r'|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
)

# Selectores del botón "Ver teléfono" en inmuebles24 (pueden cambiar con updates del sitio)
PHONE_BUTTON_SELECTORS = [
    "[data-qa='phone-button']",
    "button[data-qa*='phone']",
    "a[data-qa*='phone']",
    ".phone-action button",
    "button:has-text('Ver teléfono')",
    "button:has-text('Mostrar teléfono')",
    "button:has-text('teléfono')",
    "[class*='phone'] button",
]

PHONE_TEXT_SELECTORS = [
    "[data-qa='phone-number']",
    "[data-qa='posting-phone']",
    ".phone-value",
    "[class*='phoneNumber']",
    "[class*='phone-number']",
]

async def extract_phone(page, url: str) -> str:
    try:
        await page.goto(url, wait_until='domcontentloaded', timeout=25000)
        await page.wait_for_timeout(2000)

        # Intentar hacer click en el botón de revelar teléfono
        for sel in PHONE_BUTTON_SELECTORS:
            try:
                btn = page.locator(sel).first
                if await btn.count() > 0:
                    await btn.click(timeout=3000)
                    await page.wait_for_timeout(1500)
                    break
            except Exception:
                continue

        # Intentar leer el teléfono de selectores específicos
        for sel in PHONE_TEXT_SELECTORS:
            try:
                el = page.locator(sel).first
                if await el.count() > 0:
                    text = await el.inner_text(timeout=3000)
                    m = PHONE_RE.search(text)
                    if m:
                        return m.group().strip()
            except Exception:
                continue

        # Fallback: buscar en todo el contenido de la página con regex
        content = await page.content()
        m = PHONE_RE.search(content)
        if m:
            return m.group().strip()

        return 'No encontrado'

    except Exception as e:
        return f'Error: {str(e)[:60]}'


# ── Main ──────────────────────────────────────────────────────────
async def main():
    df = get_target_urls()
    print(f'Total propiedades a visitar: {len(df)}')
    print(f'  Estudiantes          : {(df["segmento"]=="Estudiantes").sum()}')
    print(f'  Adulto mayor (proxy) : {(df["segmento"]=="Adulto mayor (proxy)").sum()}\n')

    phones = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()

        for i, row in df.iterrows():
            seg   = row['segmento']
            url   = row['full_url']
            pub   = row['publisher']
            addr  = row['address'] if pd.notna(row['address']) else '(sin dirección)'
            price = row['price']
            idx   = phones.__len__() + 1

            print(f'[{idx:02d}/20] {seg[:8]} | {pub} | ${price:,.0f} ...')
            phone = await extract_phone(page, url)
            print(f'         → Teléfono: {phone}')
            phones.append(phone)

            # Pausa cortés entre requests
            await asyncio.sleep(2.5)

        await browser.close()

    df['telefono'] = phones

    # Reordenar columnas para legibilidad
    out_cols = ['segmento', 'address', 'location', 'price',
                'publisher', 'publisherCode', 'telefono', 'full_url']
    df[out_cols].to_csv(OUT_CSV, index=False)

    print(f'\n✓ Resultados guardados en: {OUT_CSV}')
    print('\n=== RESUMEN ===')
    for seg in ['Estudiantes', 'Adulto mayor (proxy)']:
        sub = df[df['segmento'] == seg]
        encontrados = sub[~sub['telefono'].str.startswith(('No', 'Error'))].shape[0]
        print(f'{seg}: {encontrados}/{len(sub)} teléfonos encontrados')

    print('\n=== TABLA FINAL ===')
    print(df[['segmento','location','price','publisher','telefono']].to_string(index=False))


if __name__ == '__main__':
    asyncio.run(main())
