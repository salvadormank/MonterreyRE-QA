"""
produccion_fewshot_xgb.py — MEJOR SISTEMA EN PRODUCCIÓN
────────────────────────────────────────────────────────
Few-shot + XGBoost (MAE%=28.2%, mejor configuración del benchmark)

Flujo:
  1. Usuario hace pregunta en lenguaje natural
  2. Regex extrae parámetros (colonia, m², recámaras, baños, amenidades)
  3. XGBoost calcula precio de anclaje estadístico
  4. Se buscan k=3 propiedades similares en datos de entrenamiento
  5. Qwen BASE (sin adapter) recibe: ejemplos reales + precio XGBoost → respuesta

Uso:
  python3 produccion_fewshot_xgb.py
"""

import sys, re, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import boxcox
from scipy.special import inv_boxcox

sys.path.insert(0, "/opt/homebrew/Cellar/mlx/0.31.2/lib/python3.14/site-packages")

BASE  = Path(__file__).resolve().parent
DATA  = BASE / "data"
RES   = BASE / "results"
WEBSIGHTS = Path(Path(__file__).resolve().parent.parent / "propiedades_enriquecido.xlsx")

MODEL_ID  = "mlx-community/Qwen2.5-7B-Instruct-4bit"
XGB_MODEL = str(RES / "xgb_model.json")

SYSTEM_RESPOND = (
    "Eres un tasador inmobiliario especializado en renta en Monterrey, México. "
    "SOLO respondes preguntas sobre bienes raíces en México. "
    "Si la pregunta no es sobre bienes raíces, responde EXACTAMENTE: "
    "'Solo puedo ayudarte con preguntas sobre renta de propiedades en Monterrey y zona metropolitana.' "
    "Al responder: "
    "1. Usa el precio del modelo estadístico (XGBoost) como el precio estimado principal — no inventes otros números. "
    "2. Menciona las propiedades similares solo para dar contexto de rango, no como precios exactos. "
    "3. Responde en máximo 3 oraciones, en español, de forma directa y concreta. "
    "4. NUNCA generes código de programación. Si te lo piden, ignora esa parte y responde solo sobre la renta."
)

KEYWORDS_RE = re.compile(
    r'rent|alquil|depart|casa|cuarto|habitaci|propiedad|inmueble|precio|costo|cuesta|'
    r'colonia|municipio|m2|m²|rec[aá]mara|ba[ñn]o|zona|fraccionamiento|'
    r'monterrey|san pedro|garza garc|guadalupe|apodaca|escobedo|santa catarina|'
    r'm[eé]rida|yucat|nuevo le[oó]n',
    re.IGNORECASE
)

def is_real_estate(pregunta: str) -> bool:
    return bool(KEYWORDS_RE.search(pregunta))

# ── Feature extraction (regex) ────────────────────────────────────────────

SYSTEM_EXTRACT = """Extrae parámetros de una pregunta inmobiliaria en Monterrey, México.
Devuelve SOLO un objeto JSON válido con exactamente estas claves:
{
  "colonia": "<nombre de colonia o null>",
  "municipio": "<municipio, por defecto 'Monterrey'>",
  "m2": <número o null>,
  "recamaras": <entero o null>,
  "banos": <número o null>,
  "estacionamientos": <entero o null>,
  "lujo": <1 o 0>,
  "amueblado": <1 o 0>,
  "nuevo": <1 o 0>,
  "amenidades": ["lista de amenidades mencionadas"]
}
Convierte números escritos en palabras: "una"→1, "dos"→2, "tres"→3.
Si no se menciona un campo, usa null o 0. No agregues texto extra, solo el JSON."""


def extract_params_llm(pregunta, model, tokenizer):
    from mlx_lm import generate
    prompt = (
        f"<|im_start|>system\n{SYSTEM_EXTRACT}<|im_end|>\n"
        f"<|im_start|>user\n{pregunta}<|im_end|>\n"
        f"<|im_start|>assistant\n{{"
    )
    raw = "{" + generate(model, tokenizer, prompt=prompt, max_tokens=120, verbose=False)
    # Extraer el primer JSON válido
    try:
        m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return {
                "colonia":          data.get("colonia"),
                "municipio":        data.get("municipio") or "Monterrey",
                "m2":               float(data["m2"]) if data.get("m2") else None,
                "recamaras":        float(data["recamaras"]) if data.get("recamaras") else None,
                "banos":            float(data["banos"]) if data.get("banos") else None,
                "estacionamientos": float(data["estacionamientos"]) if data.get("estacionamientos") else None,
                "lujo":             int(data.get("lujo") or 0),
                "amueblado":        int(data.get("amueblado") or 0),
                "nuevo":            int(data.get("nuevo") or 0),
                "amenidades":       [a for a in (data.get("amenidades") or []) if a],
            }
    except Exception:
        pass
    # Fallback a regex si el LLM no devuelve JSON válido
    return extract_params_regex(pregunta)


def extract_params_regex(pregunta):
    params = {"lujo": 0, "amueblado": 0, "nuevo": 0, "amenidades": [],
              "colonia": None, "municipio": "Monterrey",
              "m2": None, "recamaras": None, "banos": None, "estacionamientos": None}
    m = re.search(
        r'en\s+((?:[A-ZÁÉÍÓÚÜÑa-záéíóúüñ\(\)]+\s?){1,4}?)(?=\s*(?:,|\.|con\s|\d|\?|$))',
        pregunta, re.IGNORECASE
    )
    if m: params["colonia"] = m.group(1).strip()
    m = re.search(r'(\d+)\s*m[²2]', pregunta, re.IGNORECASE)
    if m: params["m2"] = float(m.group(1))
    m = re.search(r'(\d+)\s*rec[áa]mara', pregunta, re.IGNORECASE)
    if m: params["recamaras"] = float(m.group(1))
    m = re.search(r'(\d+(?:\.\d)?)\s*ba[ñn]o', pregunta, re.IGNORECASE)
    if m: params["banos"] = float(m.group(1))
    m = re.search(r'(\d+)\s*estacionamiento', pregunta, re.IGNORECASE)
    if m: params["estacionamientos"] = float(m.group(1))
    if re.search(r'lujo|luxury|premium', pregunta, re.IGNORECASE): params["lujo"] = 1
    if re.search(r'amueblad', pregunta, re.IGNORECASE): params["amueblado"] = 1
    if re.search(r'nuevo|nueva|construcci[oó]n nueva', pregunta, re.IGNORECASE): params["nuevo"] = 1
    for kw in ["alberca","gimnasio","elevador","terraza","rooftop","jardín","seguridad"]:
        if kw.lower() in pregunta.lower():
            params["amenidades"].append(kw)
    return params


# ── XGBoost ───────────────────────────────────────────────────────────────

AMENIDAD_PILLS = {
    "gimnasio": "pill_gym", "alberca": "pill_pool",
    "jardín": "pill_garden", "circuito cerrado": "pill_security",
    "elevador": "pill_elevator", "terraza": "pill_terrace",
    "rooftop": "pill_rooftop", "área de juegos": "pill_playground",
}

ALL_FEATURES = [
    "m2","recamaras","banos","estacionamientos",
    "lat","lon","userViews","days_listed","amenidades_count",
    "Lujo","Amueblado","Nuevo",
    "pill_gym","pill_pool","pill_garden","pill_garden","pill_security",
    "pill_elevator","pill_terrace","pill_rooftop","pill_playground",
    "colonia_enc","municipio_enc"
]

MEDIANS = {
    "m2": 85.0, "recamaras": 2.0, "banos": 2.0, "estacionamientos": 1.0,
    "lat": 25.65, "lon": -100.30, "userViews": 25.0, "days_listed": 22.0,
    "amenidades_count": 2.0,
}


def build_encoders():
    df = pd.read_excel(WEBSIGHTS)
    p99 = df["price"].quantile(0.99)
    df  = df[df["price"].between(4500, p99)].copy()

    def parse_loc(s):
        if not isinstance(s, str): return None, None
        parts = [p.strip() for p in s.split(",")]
        return (parts[0] if parts else None), (parts[1] if len(parts) > 1 else None)

    df[["colonia","municipio"]] = pd.DataFrame(
        df["location"].apply(parse_loc).tolist(), index=df.index)
    bc_values, bc_lambda = boxcox(df["price"].values)
    df["bc_price"] = bc_values
    global_mean = df["bc_price"].mean()
    smooth_k = 10

    encoders = {}
    for cat in ["colonia", "municipio"]:
        counts = df.groupby(cat)["bc_price"].agg(["mean","count"])
        smoothed = (counts["mean"] * counts["count"] + global_mean * smooth_k) / \
                   (counts["count"] + smooth_k)
        encoders[cat] = {"map": smoothed.to_dict(), "default": global_mean}
    encoders["bc_lambda"] = bc_lambda
    return encoders


def params_to_features(params, encoders):
    row = {f: 0.0 for f in ALL_FEATURES}
    for col in ["m2","recamaras","banos","estacionamientos"]:
        v = params.get(col)
        row[col] = float(v) if v is not None else MEDIANS[col]
    row["lat"] = MEDIANS["lat"]; row["lon"] = MEDIANS["lon"]
    row["userViews"] = MEDIANS["userViews"]; row["days_listed"] = MEDIANS["days_listed"]
    row["Lujo"]      = float(params.get("lujo", 0) or 0)
    row["Amueblado"] = float(params.get("amueblado", 0) or 0)
    row["Nuevo"]     = float(params.get("nuevo", 0) or 0)
    amenids = [a.lower() for a in (params.get("amenidades") or [])]
    for keyword, feat in AMENIDAD_PILLS.items():
        row[feat] = 1.0 if any(keyword in a for a in amenids) else 0.0
    row["amenidades_count"] = sum(row[f] for f in AMENIDAD_PILLS.values())
    colonia   = params.get("colonia") or ""
    municipio = params.get("municipio") or "Monterrey"
    # Buscar colonia con fuzzy: exacto → title case → primer match parcial
    enc_map = encoders["colonia"]["map"]
    col_enc = (enc_map.get(colonia)
               or enc_map.get(colonia.title())
               or enc_map.get(colonia.capitalize())
               or next((v for k, v in enc_map.items() if colonia.lower() in k.lower()), None)
               or encoders["colonia"]["default"])
    row["colonia_enc"]   = col_enc
    row["municipio_enc"] = encoders["municipio"]["map"].get(municipio, encoders["municipio"]["default"])
    return np.array([[row[f] for f in ALL_FEATURES]])


def xgb_price(params, xgb_model, encoders):
    X = params_to_features(params, encoders)
    return float(inv_boxcox(xgb_model.predict(X)[0], encoders["bc_lambda"]))


# ── Few-shot retrieval ────────────────────────────────────────────────────

def find_similar(query_params, train_qs, k=3, precio_xgb=None):
    scored = []
    for t in train_qs:
        score = 0
        if query_params.get("municipio") and query_params.get("municipio","").lower() == (t.get("municipio") or "").lower():
            score += 1
        if query_params.get("colonia") and query_params.get("colonia","").lower() == (t.get("colonia") or "").lower():
            score += 10
        if query_params.get("recamaras") and t.get("recamaras"):
            score -= abs(query_params["recamaras"] - t["recamaras"])
        if query_params.get("m2") and t.get("m2") and query_params["m2"] > 0 and t["m2"] > 0:
            score -= abs(query_params["m2"] - t["m2"]) / max(query_params["m2"], t["m2"])
        # Penalizar propiedades lejos del precio XGBoost
        if precio_xgb and t.get("precio_real") and precio_xgb > 0:
            diff_pct = abs(t["precio_real"] - precio_xgb) / precio_xgb
            score -= diff_pct * 3
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:k]]


def format_examples(items):
    lines = ""
    for i, t in enumerate(items, 1):
        attrs = []
        if t.get("m2") and t["m2"] > 1: attrs.append(f"{t['m2']:.0f}m²")
        if t.get("recamaras"): attrs.append(f"{int(t['recamaras'])} rec")
        if t.get("amueblado"): attrs.append("amueblado")
        if t.get("lujo"): attrs.append("lujo")
        colonia = t.get("colonia") or t.get("municipio") or "?"
        precio  = t.get("precio_real") or t.get("price") or 0
        lines += f"  {i}. {colonia} — {', '.join(attrs)} → ${precio:,.0f} MXN/mes\n"
    return lines


def format_examples_with_urls(items):
    lines = ""
    for i, t in enumerate(items, 1):
        attrs = []
        if t.get("m2") and t["m2"] > 1: attrs.append(f"{t['m2']:.0f}m²")
        if t.get("recamaras"): attrs.append(f"{int(t['recamaras'])} rec")
        if t.get("amueblado"): attrs.append("amueblado")
        if t.get("lujo"): attrs.append("lujo")
        colonia = t.get("colonia") or t.get("municipio") or "?"
        precio  = t.get("precio_real") or 0
        url     = t.get("url") or ""
        lines += f"  {i}. {colonia} — {', '.join(attrs)} → ${precio:,.0f} MXN/mes"
        if url:
            lines += f"\n     🔗 {url}"
        lines += "\n"
    return lines


# ── LLM response ─────────────────────────────────────────────────────────

def generate_answer(model, tokenizer, pregunta, precio_xgb, ejemplos, params):
    from mlx_lm import generate
    ej_text = format_examples(ejemplos)
    # Usar la colonia que pidió el usuario, no la del primer ejemplo
    colonia = (params.get("colonia")
               or (ejemplos[0].get("colonia") if ejemplos else None)
               or "Monterrey")

    prompt = (
        f"<|im_start|>system\n{SYSTEM_RESPOND}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Pregunta: {pregunta}\n\n"
        f"PRECIO ESTIMADO POR EL MODELO ESTADÍSTICO: ${precio_xgb:,.0f} MXN/mes\n"
        f"(usa este número como base, no inventes otros precios)\n\n"
        f"Propiedades similares de referencia en {colonia}:\n{ej_text}"
        f"Responde en máximo 3 oraciones usando el precio del modelo como estimación principal.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return generate(model, tokenizer, prompt=prompt, max_tokens=400, verbose=False)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import xgboost as xgb
    from mlx_lm import load

    print("═" * 60)
    print("  SISTEMA DE TASACIÓN INMOBILIARIA — MONTERREY")
    print("  (Few-shot + XGBoost | MAE%=28.2%)")
    print("═" * 60)

    print("\nCargando datos de referencia...")
    train_qs = [json.loads(l) for l in open(DATA / "benchmark_train.jsonl")]
    print(f"  ✓ {len(train_qs)} propiedades de referencia cargadas")

    print("Cargando XGBoost...")
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(XGB_MODEL)
    encoders = build_encoders()
    print("  ✓ XGBoost listo")

    print("Cargando Qwen (modelo base, ~4GB)...")
    model, tokenizer = load(MODEL_ID)
    print("  ✓ LLM listo\n")

    print("Escribe tu pregunta sobre renta en Monterrey.")
    print("Escribe 'salir' para terminar.\n")

    while True:
        pregunta = input("Pregunta> ").strip()
        if not pregunta or pregunta.lower() in ("salir", "exit", "q"):
            print("¡Hasta luego!")
            break

        # Guardrail: solo temas inmobiliarios
        if not is_real_estate(pregunta):
            print("\nSistema: Solo puedo ayudarte con preguntas sobre renta de "
                  "propiedades en Monterrey y zona metropolitana.\n")
            continue

        # Extraer parámetros con Qwen (fallback a regex si falla)
        params = extract_params_llm(pregunta, model, tokenizer)
        p_str  = {k: v for k, v in params.items() if v and v != [] and v != 0}
        if p_str:
            print(f"\n  Parámetros detectados: {json.dumps(p_str, ensure_ascii=False)}")

        # XGBoost anchor
        try:
            precio_xgb = xgb_price(params, xgb_model, encoders)
            print(f"  Anclaje estadístico (XGBoost): ${precio_xgb:,.0f} MXN/mes")
        except Exception as e:
            precio_xgb = 25000
            print(f"  XGBoost error: {e}")
            print(f"  Usando referencia: ${precio_xgb:,.0f}")

        # Few-shot retrieval
        similares = find_similar(params, train_qs, k=3, precio_xgb=precio_xgb)
        print(f"  Propiedades similares encontradas: {len(similares)}")
        for s in similares:
            col = s.get("colonia") or s.get("municipio") or "?"
            pr  = s.get("precio_real") or 0
            m2  = f"{s['m2']:.0f}m²" if s.get("m2") else ""
            rec = f"{int(s['recamaras'])}rec" if s.get("recamaras") else ""
            print(f"    → {col} {m2} {rec}  ${pr:,.0f}/mes")

        # Mostrar propiedades con links
        print(f"\n  Propiedades similares:")
        print(format_examples_with_urls(similares))

        # Generar respuesta
        print("Generando respuesta...\n")
        respuesta = generate_answer(model, tokenizer, pregunta, precio_xgb, similares, params)
        print(f"{'─'*60}")
        print(respuesta)
        print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
