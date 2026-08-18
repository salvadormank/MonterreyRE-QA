"""
pipeline_semanal.py
─────────────────────────────────────────────────────────────────────────────
Pipeline completo de actualización semanal:
  1. Scraping  → Lamudi + Inmuebles24 (+ MercadoLibre si hay token)
  2. Merge     → combina nuevos registros con base existente (dedup por URL)
  3. Enrich    → GPT-4o-mini en registros sin enriquecimiento
  4. HTML      → regenera leads_lamudi.html
  5. LaTeX     → regenera leads_segmentos.tex

Uso:
  python3 pipeline_semanal.py --test          # modo rápido (2 páginas/fuente)
  python3 pipeline_semanal.py                 # completo
  python3 pipeline_semanal.py --skip-scrape   # solo enrich + HTML + LaTeX

Variables de entorno requeridas:
  SCRAPERAPI_KEY   → para Lamudi e Inmuebles24
  OPENAI_API_KEY   → para enriquecimiento GPT-4o-mini
  ML_CLIENT_ID / ML_CLIENT_SECRET  → opcional MercadoLibre
─────────────────────────────────────────────────────────────────────────────
"""

import os, subprocess, sys, time
from pathlib import Path
import pandas as pd

BASE_DIR  = Path(__file__).resolve().parent
PYTHON    = sys.executable

ENRICHED_CSV   = BASE_DIR / 'leads_lamudi_enriched_openai.csv'
LAMUDI_RAW     = BASE_DIR / 'leads_lamudi_renta.csv'
INMUEBLES_RAW  = BASE_DIR / 'inmuebles24_raw.csv'
ML_RAW         = BASE_DIR / 'mercadolibre_raw.csv'
MERGED_CSV     = BASE_DIR / 'leads_merged_raw.csv'


def run(cmd: list[str], label: str) -> bool:
    print(f'\n{"─"*55}')
    print(f'  {label}')
    print(f'{"─"*55}')
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f'  ✗ {label} falló (código {result.returncode})')
        return False
    print(f'  ✓ {label} completado')
    return True


def merge_sources(test: bool = False) -> int:
    """Combina todos los CSV raw + base existente, deduplica por source_url."""
    dfs = []

    # Base existente enriquecida (mantiene enriquecimiento previo)
    if ENRICHED_CSV.exists():
        df_base = pd.read_csv(ENRICHED_CSV)
        dfs.append(df_base)
        print(f'  Base existente   : {len(df_base):,} registros')

    # Nuevos raw de cada fuente
    for csv_path, label in [
        (LAMUDI_RAW,    'Lamudi raw'),
        (INMUEBLES_RAW, 'Inmuebles24 raw'),
        (ML_RAW,        'MercadoLibre raw'),
    ]:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            dfs.append(df)
            print(f'  {label:<20}: {len(df):,} registros')

    if not dfs:
        print('  ✗ No hay CSV fuente disponible.')
        return 0

    df_all = pd.concat(dfs, ignore_index=True)
    before = len(df_all)
    df_all = df_all.drop_duplicates(subset=['source_url'], keep='first')
    df_all = df_all[df_all['price'].notna() & (df_all['price'] > 0)]
    after = len(df_all)
    print(f'  Total después de dedup: {after:,}  (eliminados: {before - after:,})')

    df_all.to_csv(MERGED_CSV, index=False)
    return after


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',         action='store_true', help='Modo rápido')
    parser.add_argument('--skip-scrape',  action='store_true', help='Omitir scrapers')
    parser.add_argument('--skip-enrich',  action='store_true', help='Omitir enriquecimiento')
    parser.add_argument('--skip-html',    action='store_true', help='Omitir generación HTML')
    parser.add_argument('--skip-latex',   action='store_true', help='Omitir generación LaTeX')
    args = parser.parse_args()

    print(f'\n{"="*55}')
    print('  PIPELINE SEMANAL — LEADS INMOBILIARIOS MTY')
    print(f'{"="*55}')
    print(f'  Fecha: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'  Modo:  {"TEST (2 pgs)" if args.test else "COMPLETO"}')

    # ── 1. Scraping ───────────────────────────────────────────────────────
    if not args.skip_scrape:
        test_flag = ['--test'] if args.test else []

        if os.environ.get('SCRAPERAPI_KEY'):
            # Inmuebles24
            run([PYTHON, str(BASE_DIR / 'scraper_inmuebles24.py')] + test_flag,
                'Scraping Inmuebles24')

            # Lamudi (usa scrape_scraperapi.py existente)
            lamudi_script = BASE_DIR / 'scrape_scraperapi.py'
            if lamudi_script.exists():
                run([PYTHON, str(lamudi_script)] + test_flag,
                    'Scraping Lamudi')
        else:
            print('\n  ⚠  SCRAPERAPI_KEY no configurada — skip scrapers web')

        # MercadoLibre (opcional)
        ml_token = BASE_DIR / '.ml_token.json'
        if ml_token.exists() and os.environ.get('ML_CLIENT_ID'):
            run([PYTHON, str(BASE_DIR / 'scraper_mercadolibre.py')] + test_flag,
                'Scraping MercadoLibre')
        else:
            print('\n  ⚠  ML token no disponible — skip MercadoLibre')

    # ── 2. Merge ──────────────────────────────────────────────────────────
    print(f'\n{"─"*55}')
    print('  Merge de fuentes')
    print(f'{"─"*55}')
    total = merge_sources(args.test)
    if total == 0:
        print('  ✗ Sin datos para continuar.')
        return

    # ── 3. Enrich ─────────────────────────────────────────────────────────
    if not args.skip_enrich and os.environ.get('OPENAI_API_KEY'):
        # Enriquece solo registros nuevos (los que no tienen enriquecimiento)
        run([PYTHON, str(BASE_DIR / 'enrich_lamudi_openai.py'), '--resume'],
            'Enriquecimiento GPT-4o-mini')
    elif not args.skip_enrich:
        print('\n  ⚠  OPENAI_API_KEY no configurada — skip enriquecimiento')

    # ── 4. HTML ───────────────────────────────────────────────────────────
    if not args.skip_html:
        run([PYTHON, str(BASE_DIR / 'generar_html.py')],
            'Generando HTML interactivo')

    # ── 5. LaTeX ──────────────────────────────────────────────────────────
    if not args.skip_latex:
        run([PYTHON, str(BASE_DIR / 'generar_tablas_leads.py')],
            'Generando tablas LaTeX')

    print(f'\n{"="*55}')
    print('  PIPELINE COMPLETADO')
    print(f'{"="*55}')
    print(f'  Leads en base: {total:,}')
    print(f'  HTML: {BASE_DIR}/leads_lamudi.html')
    print(f'  CSV enriquecido: {ENRICHED_CSV}')
    print(f'{"="*55}\n')


if __name__ == '__main__':
    main()
