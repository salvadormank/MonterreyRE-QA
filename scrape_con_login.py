"""
Extrae teléfonos de las 20 propiedades top (estudiantes + adultos mayores).
Paso 1: abre el navegador visible para que el usuario haga login con Google.
Paso 2: script continúa automáticamente con la sesión activa.
"""

import asyncio
import re
import pandas as pd
import numpy as np
from playwright.async_api import async_playwright

BASE_URL = "https://www.inmuebles24.com"
LOGIN_URL = "https://www.inmuebles24.com/login"
OUT_CSV   = "../analisis_propiedades/contactos_segmentos.csv"

PHONE_RE = re.compile(
    r'\b(?:\+?52[-.\s]?)?(?:\(?\d{2,3}\)?[-.\s]?)?\d{4}[-.\s]?\d{4}\b'
    r'|\b\d{10}\b'
    r'|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
)

PHONE_BUTTON_SELECTORS = [
    "[data-qa='phone-button']",
    "button[data-qa*='phone']",
    "a[data-qa*='phone']",
    "button:has-text('Ver teléfono')",
    "button:has-text('Mostrar teléfono')",
    "button:has-text('teléfono')",
    "[class*='phone'] button",
    "[class*='Phone'] button",
]

PHONE_TEXT_SELECTORS = [
    "[data-qa='phone-number']",
    "[data-qa='posting-phone']",
    "[class*='phoneNumber']",
    "[class*='phone-number']",
    "[class*='PhoneNumber']",
]

LOGGED_IN_SELECTORS = [
    "[data-qa='user-menu']",
    "[data-qa='header-user']",
    ".user-menu",
    "a[href*='/mis-avisos']",
    "a[href*='/perfil']",
    "[class*='userLogged']",
    "[class*='user-logged']",
]


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


async def wait_for_login(page):
    print('\n' + '='*55)
    print('  Loguéate con Google en la ventana del navegador.')
    print('  El script continúa automáticamente al detectar')
    print('  que ya iniciaste sesión.')
    print('='*55 + '\n')

    await page.goto(LOGIN_URL, wait_until='domcontentloaded')

    # Espera hasta 3 minutos a que el usuario se loguee
    for _ in range(180):
        for sel in LOGGED_IN_SELECTORS:
            try:
                if await page.locator(sel).count() > 0:
                    print('✓ Sesión detectada — continuando...\n')
                    return True
            except Exception:
                continue
        # También checar que ya no estamos en /login
        if 'login' not in page.url:
            await asyncio.sleep(2)
            for sel in LOGGED_IN_SELECTORS:
                try:
                    if await page.locator(sel).count() > 0:
                        print('✓ Sesión detectada — continuando...\n')
                        return True
                except Exception:
                    continue
        await asyncio.sleep(1)

    print('✗ Tiempo de espera agotado (3 min). Saliendo.')
    return False


async def extract_phone(page, url: str) -> str:
    try:
        await page.goto(url, wait_until='domcontentloaded', timeout=30000)
        await page.wait_for_timeout(2000)

        # Click en botón de teléfono
        clicked = False
        for sel in PHONE_BUTTON_SELECTORS:
            try:
                btn = page.locator(sel).first
                if await btn.count() > 0:
                    await btn.scroll_into_view_if_needed()
                    await btn.click(timeout=4000)
                    await page.wait_for_timeout(1800)
                    clicked = True
                    break
            except Exception:
                continue

        # Leer de selectores de texto específicos
        for sel in PHONE_TEXT_SELECTORS:
            try:
                el = page.locator(sel).first
                if await el.count() > 0:
                    text = await el.inner_text(timeout=3000)
                    m = PHONE_RE.search(text)
                    if m:
                        phone = m.group().strip()
                        # Filtrar falsos positivos (IDs de 8 dígitos del portal)
                        digits = re.sub(r'\D', '', phone)
                        if len(digits) >= 10:
                            return phone
            except Exception:
                continue

        # Fallback: todo el contenido de la página
        if clicked:
            content = await page.content()
            for m in PHONE_RE.finditer(content):
                digits = re.sub(r'\D', '', m.group())
                if len(digits) >= 10:
                    return m.group().strip()

        return 'No encontrado'

    except Exception as e:
        return f'Error: {str(e)[:60]}'


async def main():
    df = get_target_urls()

    async with async_playwright() as p:
        # Navegador VISIBLE para que el usuario haga login
        browser = await p.chromium.launch(headless=False, slow_mo=50)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()

        # Paso 1: login manual
        logged = await wait_for_login(page)
        if not logged:
            await browser.close()
            return

        # Paso 2: scraping automático
        phones = []
        total = len(df)

        for i, row in df.iterrows():
            idx   = len(phones) + 1
            seg   = row['segmento']
            pub   = row['publisher']
            price = row['price']
            url   = row['full_url']

            print(f'[{idx:02d}/{total}] {seg[:8]} | {pub[:30]} | ${price:,.0f}')
            phone = await extract_phone(page, url)
            phones.append(phone)

            status = '✓' if 'No' not in phone and 'Error' not in phone else '✗'
            print(f'         {status} {phone}')

            await asyncio.sleep(3.5)  # pausa cortés

        await browser.close()

    # Guardar resultados
    df['telefono'] = phones
    out_cols = ['segmento', 'address', 'location', 'price',
                'publisher', 'publisherCode', 'telefono', 'full_url']
    df[out_cols].to_csv(OUT_CSV, index=False)

    # Resumen
    print('\n' + '='*55)
    print('RESULTADOS')
    print('='*55)
    for seg in ['Estudiantes', 'Adulto mayor (proxy)']:
        sub = df[df['segmento'] == seg]
        ok  = sub[~sub['telefono'].str.startswith(('No', 'Error'))].shape[0]
        print(f'{seg}: {ok}/{len(sub)} teléfonos encontrados')

    print(f'\n✓ CSV guardado: {OUT_CSV}')
    print('\n' + df[['segmento', 'location', 'price', 'publisher', 'telefono']].to_string(index=False))


if __name__ == '__main__':
    asyncio.run(main())
