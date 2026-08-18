"""
enrich_lamudi_openai.py
─────────────────────────────────────────────────────────────────────────────
Extrae features estructuradas de las descripciones de Lamudi usando OpenAI
(GPT-4o-mini) y produce un CSV limpio listo para regresión.

Columnas nuevas extraídas:
  Numéricas:  area_m2, bedrooms, bathrooms, parking, floor,
              price_per_m2_desc (mencionado en descripción),
              maintenance_fee, price_per_m2_calc (precio / area_m2)
  Binarias:   furnished, has_pool, has_gym, has_elevator, has_terrace,
              has_rooftop, has_garden, has_security_24h, has_lobby,
              has_sala_juntas, has_coworking, has_bodega, has_minisplit_ac,
              has_heating, has_cistern, is_new_construction,
              allows_pets, has_service_room
  Dirección:  colonia, municipio, estado   (parseadas de address)
  URL:        listing_id  (hash de Lamudi extraído de source_url)
  Target:     log_price   (log1p del precio, para modelos log-lineal)

Uso:
  export OPENAI_API_KEY=sk-...
  python enrich_lamudi_openai.py

El script guarda progreso incremental en leads_lamudi_enriched_openai.csv
para no perder trabajo si se corta la conexión.
─────────────────────────────────────────────────────────────────────────────
"""

import os, re, json, time
import numpy as np
import pandas as pd
from openai import OpenAI

# ── Configuración ─────────────────────────────────────────────────────────
INPUT_CSV  = "./leads_lamudi_renta.csv"
OUTPUT_CSV = "./leads_lamudi_enriched_openai.csv"
MODEL      = "gpt-4o-mini"
BATCH_SIZE = 10           # registros por llamada a OpenAI
SLEEP_BETWEEN = 0.5       # segundos entre batches (rate-limit amigable)

import argparse

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

# ── Schema que devolverá OpenAI (JSON) ────────────────────────────────────
SYSTEM_PROMPT = """Eres un asistente que extrae información estructurada de
descripciones de propiedades inmobiliarias en México. Devuelves SOLO JSON
válido, sin texto extra, con exactamente estas claves:

{
  "area_m2":              <número o null>,
  "bedrooms":             <entero o null>,
  "bathrooms":            <número o null>,
  "parking":              <entero o null>,
  "floor":                <entero o null>,
  "furnished":            <true/false/null>,
  "has_pool":             <true/false>,
  "has_gym":              <true/false>,
  "has_elevator":         <true/false>,
  "has_terrace":          <true/false>,
  "has_rooftop":          <true/false>,
  "has_garden":           <true/false>,
  "has_security_24h":     <true/false>,
  "has_lobby":            <true/false>,
  "has_sala_juntas":      <true/false>,
  "has_coworking":        <true/false>,
  "has_bodega":           <true/false>,
  "has_minisplit_ac":     <true/false>,
  "has_heating":          <true/false>,
  "has_cistern":          <true/false>,
  "is_new_construction":  <true/false>,
  "allows_pets":          <true/false>,
  "has_service_room":     <true/false>,
  "price_per_m2_desc":    <número o null>,
  "maintenance_fee":      <número mensual en MXN o null>,
  "notes":                "<observaciones breves o vacío>"
}

Reglas:
- area_m2: metros cuadrados construidos (no terreno). Si hay varios valores
  toma el área rentable/construida principal.
- bathrooms puede ser 1.5, 2.5, etc. (medios baños = 0.5)
- floor: piso del departamento/oficina (no niveles del edificio).
- has_bodega: bodega de servicio dentro de la propiedad (no bodegas industriales).
- price_per_m2_desc: sólo si se menciona explícitamente el precio/m².
- maintenance_fee: cuota mensual de mantenimiento mencionada.
- Para propiedades comerciales (oficinas, locales, bodegas industriales) que
  no apliquen recámaras/baños personales, usa null.
"""

# ── Helpers ───────────────────────────────────────────────────────────────

ESTADOS_MX = {
    "nuevo león", "jalisco", "ciudad de méxico", "cdmx", "estado de méxico",
    "querétaro", "coahuila", "chihuahua", "sonora", "tamaulipas",
}

def parse_address(address: str) -> dict:
    """Descompone 'Colonia, Municipio, Estado' en partes.
    Si solo hay 2 partes y la segunda es un estado conocido, municipio queda null.
    """
    parts = [p.strip() for p in str(address).split(",")]
    if len(parts) == 3:
        return {"colonia": parts[0], "municipio": parts[1], "estado": parts[2]}
    if len(parts) == 2:
        if parts[1].lower() in ESTADOS_MX:
            return {"colonia": None, "municipio": parts[0], "estado": parts[1]}
        return {"colonia": parts[0], "municipio": parts[1], "estado": None}
    return {"colonia": parts[0] if parts else None, "municipio": None, "estado": None}


def extract_listing_id(url: str) -> str | None:
    """Extrae el hash único de la URL de Lamudi."""
    if not isinstance(url, str):
        return None
    m = re.search(r"/detalle/(.+)$", url.strip())
    return m.group(1) if m else None


def clean_url(url: str) -> str:
    """Normaliza la URL: quita espacios, garantiza https."""
    if not isinstance(url, str):
        return url
    url = url.strip()
    if url.startswith("http://"):
        url = "https://" + url[7:]
    return url


def normalize_property_type(pt: str) -> str:
    """Normaliza las variantes de property_type a categorías canónicas."""
    if not isinstance(pt, str):
        return "otro"
    pt = pt.strip().lower()
    mapping = {
        "departamento": "departamento",
        "loft":         "departamento",
        "penthouse":    "departamento",
        "casa":         "casa",
        "oficina":      "oficina",
        "local":        "local_comercial",
        "local comercial":            "local_comercial",
        "tienda / local comercial":   "local_comercial",
        "tienda/local comercial":     "local_comercial",
        "bodega":                     "bodega_industrial",
        "nave industrial / bodega":   "bodega_industrial",
        "nave industrial/bodega":     "bodega_industrial",
        "nave industrial":            "bodega_industrial",
        "terreno":         "terreno",
        "lote / terreno":  "terreno",
        "lote/terreno":    "terreno",
        "edificio":        "edificio",
        "edificio mixto":  "edificio",
        "edificio comercial": "edificio",
        "consultorio médico": "consultorio",
        "consultorio":        "consultorio",
    }
    for key, val in mapping.items():
        if pt == key:
            return val
    return pt


def call_openai_batch(descriptions: list[str]) -> list[dict]:
    """Llama a OpenAI con un batch de descripciones y devuelve lista de dicts."""
    user_msg = "\n\n---\n\n".join(
        f"[{i}] {desc}" for i, desc in enumerate(descriptions)
    )
    user_msg += (
        "\n\nDevuelve una lista JSON con exactamente "
        f"{len(descriptions)} objetos, uno por descripción, en el mismo orden."
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": user_msg},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        parsed = json.loads(raw)

        # OpenAI puede devolver {"results": [...]} o directamente [...]
        if isinstance(parsed, list):
            return parsed
        for key in ("results", "properties", "items", "data"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        # Si devolvió un solo objeto (batch de 1)
        if len(descriptions) == 1:
            return [parsed]
        return [{}] * len(descriptions)

    except Exception as e:
        print(f"    ⚠ Error OpenAI: {e}")
        return [{}] * len(descriptions)


BOOL_COLS = [
    "furnished", "has_pool", "has_gym", "has_elevator", "has_terrace",
    "has_rooftop", "has_garden", "has_security_24h", "has_lobby",
    "has_sala_juntas", "has_coworking", "has_bodega", "has_minisplit_ac",
    "has_heating", "has_cistern", "is_new_construction", "allows_pets",
    "has_service_room",
]
NUM_COLS = [
    "area_m2", "bedrooms", "bathrooms", "parking", "floor",
    "price_per_m2_desc", "maintenance_fee",
]


def safe_float(val, min_val=None, max_val=None):
    try:
        v = float(val)
        if min_val is not None and v < min_val:
            return np.nan
        if max_val is not None and v > max_val:
            return np.nan
        return v
    except (TypeError, ValueError):
        return np.nan


def parse_extracted(row: dict) -> dict:
    """Convierte el dict de OpenAI a tipos correctos con validaciones."""
    out = {}
    for col in NUM_COLS:
        v = row.get(col)
        if col == "area_m2":
            out[col] = safe_float(v, min_val=5, max_val=50_000)
        elif col == "bedrooms":
            val = safe_float(v, min_val=0, max_val=20)
            out[col] = int(val) if not np.isnan(val) else np.nan
        elif col == "bathrooms":
            out[col] = safe_float(v, min_val=0.5, max_val=20)
        elif col == "parking":
            val = safe_float(v, min_val=0, max_val=50)
            out[col] = int(val) if not np.isnan(val) else np.nan
        elif col == "floor":
            val = safe_float(v, min_val=0, max_val=100)
            out[col] = int(val) if not np.isnan(val) else np.nan
        else:
            out[col] = safe_float(v)

    for col in BOOL_COLS:
        v = row.get(col)
        if v is None:
            out[col] = np.nan
        else:
            out[col] = 1 if v else 0

    out["notes"] = str(row.get("notes", ""))[:200]
    return out


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Procesa solo los primeros 10 registros para validar")
    parser.add_argument("--resume", action="store_true",
                        help="Retoma desde el parcial guardado")
    args = parser.parse_args()

    if not client.api_key or not client.api_key.startswith("sk-"):
        print("✗ OPENAI_API_KEY no configurada.")
        print("  Corre:  export OPENAI_API_KEY=sk-...")
        return

    print(f"✓ OpenAI key: {client.api_key[:8]}...")

    # Estimación de costo
    cost_est = (1029 if not args.test else 10) * 0.00015  # ~$0.15 por 1k registros
    print(f"  Costo estimado: ~${cost_est:.2f} USD ({MODEL})\n")

    print(f"\nCargando {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  {len(df):,} registros")

    resume_start = 0
    partial_df   = None

    if args.test:
        df = df.head(10).copy()
        print(f"  Modo TEST: procesando solo {len(df)} registros\n")
    elif args.resume:
        partial_path = OUTPUT_CSV + ".partial"
        if os.path.exists(partial_path):
            partial_df   = pd.read_csv(partial_path)
            resume_start = len(partial_df)
            print(f"  Modo RESUME: retomando desde registro {resume_start + 1}\n")
        else:
            print("  No se encontró archivo parcial, arrancando desde cero.\n")
    else:
        print()

    # ── Limpieza de columnas base ─────────────────────────────────────
    # Eliminar columnas duplicadas del enriquecimiento anterior si existen
    dup_cols = [c for c in df.columns if c.endswith(".1") or c.endswith("_1")]
    if dup_cols:
        df.drop(columns=dup_cols, inplace=True)
        print(f"  Columnas duplicadas eliminadas: {dup_cols}")

    # Normalizar property_type
    df["property_type"] = df["property_type"].apply(normalize_property_type)

    # Parsear address → colonia / municipio / estado
    addr_parsed = df["address"].apply(parse_address).apply(pd.Series)
    df["colonia"]   = addr_parsed["colonia"]
    df["municipio"] = addr_parsed["municipio"]
    df["estado"]    = addr_parsed["estado"]

    # Limpiar y estructurar URL
    df["source_url"]  = df["source_url"].apply(clean_url)
    df["listing_id"]  = df["source_url"].apply(extract_listing_id)

    # ── Enriquecimiento con OpenAI ────────────────────────────────────
    # Inicializar columnas de output (NaN por defecto)
    for col in NUM_COLS + BOOL_COLS + ["notes"]:
        if col not in df.columns:
            df[col] = np.nan

    descriptions = df["description"].fillna("").tolist()
    n = len(descriptions)
    processed = 0

    print(f"Enriqueciendo {n:,} descripciones con {MODEL}...")
    pending = n - resume_start
    print(f"  Batch size: {BATCH_SIZE}  |  Batches pendientes: {(pending + BATCH_SIZE - 1) // BATCH_SIZE}\n")

    extracted_rows = []

    for batch_start in range(resume_start, n, BATCH_SIZE):
        batch_end  = min(batch_start + BATCH_SIZE, n)
        batch_desc = descriptions[batch_start:batch_end]
        batch_num  = batch_start // BATCH_SIZE + 1
        total_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"  Batch {batch_num:03d}/{total_batches}  "
              f"[{batch_start+1:4d}–{batch_end:4d}/{n}] ...",
              end="", flush=True)

        results = call_openai_batch(batch_desc)

        # Asegurar que el número de resultados coincide
        if len(results) < len(batch_desc):
            results += [{}] * (len(batch_desc) - len(results))

        for res in results:
            extracted_rows.append(parse_extracted(res))

        processed += len(batch_desc)
        print(f" ✓  ({processed}/{n})")

        # Guardado incremental cada 5 batches
        if batch_num % 5 == 0:
            _save_partial(df, extracted_rows, OUTPUT_CSV)
            print(f"    ↳ Guardado parcial: {processed} registros")

        time.sleep(SLEEP_BETWEEN)

    # ── Merge de features extraídas ───────────────────────────────────
    ext_df = pd.DataFrame(extracted_rows)

    if args.resume and partial_df is not None:
        # Combinar filas del parcial + filas nuevas
        new_rows_df = df.iloc[resume_start:].copy().reset_index(drop=True)
        for col in NUM_COLS + BOOL_COLS + ["notes"]:
            new_rows_df[col] = ext_df[col].values
        df = pd.concat([partial_df, new_rows_df], ignore_index=True)
    else:
        for col in NUM_COLS + BOOL_COLS + ["notes"]:
            df[col] = ext_df[col].values

    # ── Features derivadas para regresión ────────────────────────────
    df["price_per_m2_calc"] = np.where(
        df["area_m2"].notna() & (df["area_m2"] > 0),
        df["price"] / df["area_m2"],
        np.nan,
    )
    # Winsorizar precio_m2 calc al p95 por tipo de propiedad
    for ptype in df["property_type"].unique():
        mask = df["property_type"] == ptype
        p95  = df.loc[mask, "price_per_m2_calc"].quantile(0.95)
        df.loc[mask & (df["price_per_m2_calc"] > p95), "price_per_m2_calc"] = p95

    df["log_price"]   = np.log1p(df["price"])
    df["log_area_m2"] = np.log1p(df["area_m2"])

    # Amenidades count (score rápido de amenidades)
    amenidad_cols = [
        "has_pool", "has_gym", "has_elevator", "has_terrace", "has_rooftop",
        "has_garden", "has_security_24h", "has_lobby", "has_sala_juntas",
        "has_coworking", "has_bodega", "has_minisplit_ac",
    ]
    df["amenidades_count"] = df[amenidad_cols].fillna(0).sum(axis=1)

    # ── Orden final de columnas ────────────────────────────────────────
    col_order = [
        # Identificación
        "listing_id", "source_url",
        # Contacto
        "name", "phone", "email",
        # Ubicación
        "address", "colonia", "municipio", "estado",
        # Clasificación
        "operation_type", "property_type",
        # Target
        "price", "log_price",
        # Metros cuadrados
        "area_m2", "terrain_m2", "log_area_m2", "price_per_m2_calc", "price_per_m2_desc",
        # Características físicas
        "bedrooms", "bathrooms", "parking", "floor",
        # Financiero
        "maintenance_fee",
        # Binarias condición
        "furnished", "is_new_construction",
        # Amenidades
        "has_pool", "has_gym", "has_elevator", "has_terrace", "has_rooftop",
        "has_garden", "has_security_24h", "has_lobby", "has_sala_juntas",
        "has_coworking", "has_bodega", "has_minisplit_ac", "has_heating",
        "has_cistern", "allows_pets", "has_service_room",
        "amenidades_count",
        # Meta
        "is_owner", "score", "scraped_at",
        # Texto
        "description", "notes",
    ]
    # Agregar columnas no listadas al final
    extras = [c for c in df.columns if c not in col_order]
    df = df[col_order + extras]

    # ── Guardar ───────────────────────────────────────────────────────
    out_path = OUTPUT_CSV if not args.test else OUTPUT_CSV.replace(".csv", "_TEST.csv")
    df.to_csv(out_path, index=False)
    print(f"\n{'='*60}")
    print("RESUMEN DE ENRIQUECIMIENTO")
    print(f"{'='*60}")
    print(f"  Registros procesados : {len(df):,}")
    print(f"  Modelo OpenAI        : {MODEL}")
    print()
    print("  Cobertura de columnas extraídas:")
    for col in NUM_COLS + ["amenidades_count"]:
        n_ok = df[col].notna().sum()
        pct  = n_ok / len(df) * 100
        print(f"    {col:<28}: {n_ok:5,}  ({pct:5.1f}%)")
    print()
    print("  Amenidades detectadas:")
    for col in amenidad_cols:
        n_ok = (df[col] == 1).sum()
        pct  = n_ok / len(df) * 100
        print(f"    {col:<28}: {n_ok:5,}  ({pct:5.1f}%)")
    print()
    print(f"  ✓ CSV guardado en: {out_path}")
    print(f"{'='*60}")


def _save_partial(df_base, extracted_so_far, path):
    """Guarda las filas procesadas hasta el momento como checkpoint."""
    n = len(extracted_so_far)
    ext_df = pd.DataFrame(extracted_so_far)
    df_tmp = df_base.iloc[:n].copy().reset_index(drop=True)
    for col in ext_df.columns:
        df_tmp[col] = ext_df[col].values
    df_tmp.to_csv(path + ".partial", index=False)


if __name__ == "__main__":
    main()
