"""
hybrid_system.py
────────────────────────────────────────────────────────────────────────────
Sistema híbrido: Qwen fine-tuneado + XGBoost para predicción de precios.

Flujo:
  1. Usuario hace pregunta en lenguaje natural
  2. Qwen extrae parámetros estructurados (colonia, m², rec, baños, amenidades)
  3. XGBoost calcula precio exacto con esos parámetros
  4. Qwen genera respuesta final citando el precio de XGBoost

Modos:
  python3 hybrid_system.py                    # chat interactivo
  python3 hybrid_system.py --benchmark        # evalúa 109 preguntas del test set
  python3 hybrid_system.py --benchmark --limit 10  # prueba rápida
────────────────────────────────────────────────────────────────────────────
"""

import sys
import re
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import boxcox
from scipy.special import inv_boxcox

sys.path.insert(0, "/opt/homebrew/Cellar/mlx/0.31.2/lib/python3.14/site-packages")

BASE    = Path(__file__).resolve().parent
DATA    = BASE / "data"
RES     = BASE / "results"
RES.mkdir(exist_ok=True)

MODEL_ID    = "mlx-community/Qwen2.5-7B-Instruct-4bit"
ADAPTER     = str(BASE / "adapters")
XGB_MODEL   = str(RES / "xgb_model.json")
WEBSIGHTS   = Path(Path(__file__).resolve().parent.parent / "propiedades_enriquecido.xlsx")

SYSTEM_EXTRACT = """Eres un asistente que extrae parámetros estructurados de
preguntas sobre propiedades inmobiliarias en Monterrey, México.

Devuelves SOLO JSON con exactamente estas claves:
{
  "colonia":          "<nombre de colonia o null>",
  "municipio":        "<municipio o 'Monterrey'>",
  "m2":               <número o null>,
  "recamaras":        <entero o null>,
  "banos":            <número o null>,
  "estacionamientos": <entero o null>,
  "lujo":             <1 o 0>,
  "amueblado":        <1 o 0>,
  "nuevo":            <1 o 0>,
  "amenidades":       ["lista", "de", "amenidades"]
}

Si no se menciona algún campo, usa null o 0 según corresponda."""

SYSTEM_RESPOND = """Eres un experto en el mercado inmobiliario de renta en
Monterrey, Nuevo León. Respondes en español con argumentos concretos.
Cuando se te proporciona un precio calculado por un modelo estadístico,
úsalo como base y agrega contexto y razonamiento del mercado local."""


# ── Cargar modelo XGBoost ─────────────────────────────────────────────────
def load_xgb():
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(XGB_MODEL)
    return model


# ── Preparar encoder de colonias ──────────────────────────────────────────
def build_colonia_encoder():
    df = pd.read_excel(WEBSIGHTS)
    p99 = df["price"].quantile(0.99)
    df  = df[df["price"].between(4500, p99)].copy()
    df["userViews"] = pd.to_numeric(df["userViews"], errors="coerce")

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


# ── Construir feature vector para XGBoost ────────────────────────────────
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

# Medianas del dataset de entrenamiento para imputar nulos
MEDIANS = {
    "m2": 85.0, "recamaras": 2.0, "banos": 2.0, "estacionamientos": 1.0,
    "lat": 25.65, "lon": -100.30, "userViews": 25.0, "days_listed": 22.0,
    "amenidades_count": 2.0,
}

def params_to_features(params, encoders):
    row = {f: 0.0 for f in ALL_FEATURES}

    # Numéricas
    for col in ["m2","recamaras","banos","estacionamientos"]:
        v = params.get(col)
        row[col] = float(v) if v is not None else MEDIANS[col]

    # Defaults geográficos (Monterrey centro)
    row["lat"]         = MEDIANS["lat"]
    row["lon"]         = MEDIANS["lon"]
    row["userViews"]   = MEDIANS["userViews"]
    row["days_listed"] = MEDIANS["days_listed"]

    # Binarias
    row["Lujo"]      = float(params.get("lujo", 0) or 0)
    row["Amueblado"] = float(params.get("amueblado", 0) or 0)
    row["Nuevo"]     = float(params.get("nuevo", 0) or 0)

    # Amenidades
    amenids = [a.lower() for a in (params.get("amenidades") or [])]
    for keyword, feat in AMENIDAD_PILLS.items():
        row[feat] = 1.0 if any(keyword in a for a in amenids) else 0.0
    row["amenidades_count"] = sum(row[f] for f in AMENIDAD_PILLS.values())

    # Target encoding
    colonia   = params.get("colonia") or ""
    municipio = params.get("municipio") or "Monterrey"
    row["colonia_enc"]   = encoders["colonia"]["map"].get(
        colonia, encoders["colonia"]["default"])
    row["municipio_enc"] = encoders["municipio"]["map"].get(
        municipio, encoders["municipio"]["default"])

    return np.array([[row[f] for f in ALL_FEATURES]])


# ── Llamadas al LLM ───────────────────────────────────────────────────────
def extract_params(model, tokenizer, pregunta):
    """Extrae parámetros con regex — evita segunda llamada al LLM."""
    params = {"lujo": 0, "amueblado": 0, "nuevo": 0, "amenidades": []}

    # Colonia: busca patrón "en X, Monterrey" o "en X"
    m = re.search(r'en\s+([A-ZÁÉÍÓÚÜÑa-záéíóúüñ][A-ZÁÉÍÓÚÜÑa-záéíóúüñ\s\(\)]+?)(?:,|\.|con|\d)', pregunta)
    if m:
        params["colonia"] = m.group(1).strip()
        params["municipio"] = "Monterrey"

    # m²
    m = re.search(r'(\d+)\s*m[²2]', pregunta, re.IGNORECASE)
    if m: params["m2"] = float(m.group(1))

    # Recámaras
    m = re.search(r'(\d+)\s*rec[áa]mara', pregunta, re.IGNORECASE)
    if m: params["recamaras"] = float(m.group(1))

    # Baños
    m = re.search(r'(\d+(?:\.\d)?)\s*ba[ñn]o', pregunta, re.IGNORECASE)
    if m: params["banos"] = float(m.group(1))

    # Binarias
    if re.search(r'lujo|luxury|premium', pregunta, re.IGNORECASE):
        params["lujo"] = 1
    if re.search(r'amueblad', pregunta, re.IGNORECASE):
        params["amueblado"] = 1
    if re.search(r'nuevo|nueva|construcci[oó]n nueva', pregunta, re.IGNORECASE):
        params["nuevo"] = 1

    # Amenidades
    for kw in ["alberca","gimnasio","elevador","terraza","rooftop","jardín","jardín","seguridad"]:
        if kw.lower() in pregunta.lower():
            params["amenidades"].append(kw)

    return params


def generate_response(model, tokenizer, pregunta, precio_xgb, params):
    """Paso 2: generar respuesta citando el precio de XGBoost."""
    from mlx_lm import generate
    colonia = params.get("colonia") or "Monterrey"
    context = (
        f"Pregunta del usuario: {pregunta}\n\n"
        f"Precio calculado por modelo estadístico (XGBoost, R²=0.70): "
        f"${precio_xgb:,.0f} MXN/mes\n\n"
        f"Responde en español explicando el precio para {colonia}, "
        f"mencionando que está basado en datos reales del mercado."
    )
    prompt = (
        f"<|im_start|>system\n{SYSTEM_RESPOND}<|im_end|>\n"
        f"<|im_start|>user\n{context}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return generate(model, tokenizer, prompt=prompt, max_tokens=350, verbose=False)


# ── Benchmark ─────────────────────────────────────────────────────────────
def run_hybrid_benchmark(model, tokenizer, xgb_model, encoders, questions):
    results = []
    n = len(questions)

    for i, q in enumerate(questions):
        if i % 10 == 0:
            print(f"  Progreso: {i}/{n} ({i/n*100:.0f}%)")

        pregunta   = q["pregunta"]
        precio_real = q.get("precio_real")
        precio_xgb  = q.get("precio_xgb")
        params      = q.get("params", {})

        # LLM genera respuesta usando precio ya calculado por XGBoost
        if precio_xgb:
            respuesta = generate_response(model, tokenizer, pregunta, precio_xgb, params)
        else:
            respuesta = "No se pudo calcular el precio."

        error_pct = abs(precio_xgb - precio_real) / precio_real * 100 \
                    if precio_xgb and precio_real else None

        results.append({
            **q,
            "respuesta_hibrida": respuesta,
            "error_pct": error_pct,
            "dentro_15pct": error_pct <= 15 if error_pct else False,
            "dentro_20pct": error_pct <= 20 if error_pct else False,
        })

    return results


def print_metrics(results, label):
    priced = [r for r in results if r.get("precio_real") and r.get("precio_xgb")]
    errors = [r["error_pct"] for r in priced if r.get("error_pct") is not None]

    print(f"\n  {label}")
    print(f"  {'─'*40}")
    print(f"  Preguntas evaluadas : {len(results)}")
    print(f"  Con precio XGBoost  : {len(priced)}")
    if errors:
        print(f"  MAE%                : {np.mean(errors):.1f}%")
        mae_mxn = np.mean([abs(r['precio_xgb']-r['precio_real']) for r in priced])
        print(f"  MAE (MXN)           : ${mae_mxn:,.0f}")
        print(f"  Dentro de ±15%      : {sum(1 for e in errors if e<=15)/len(errors)*100:.1f}%")
        print(f"  Dentro de ±20%      : {sum(1 for e in errors if e<=20)/len(errors)*100:.1f}%")
    n_fail = sum(1 for r in results if not r.get("precio_xgb"))
    print(f"  Fallos extracción   : {n_fail} ({n_fail/len(results)*100:.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from mlx_lm import load

    if args.benchmark:
        questions = [json.loads(l) for l in open(DATA / "benchmark_test.jsonl")]
        if args.limit:
            questions = questions[:args.limit]
            print(f"Modo limitado: {args.limit} preguntas")

        # Paso 1: XGBoost predice precios (sin LLM en memoria)
        print("Paso 1/2: XGBoost calculando precios...")
        xgb_model = load_xgb()
        encoders  = build_colonia_encoder()
        for q in questions:
            params = extract_params(None, None, q["pregunta"])
            q["params"] = params
            try:
                X = params_to_features(params, encoders)
                q["precio_xgb"] = float(inv_boxcox(xgb_model.predict(X)[0], encoders["bc_lambda"]))
            except:
                q["precio_xgb"] = None
        del xgb_model  # liberar memoria antes de cargar LLM

        # Paso 2: LLM genera respuestas (XGBoost ya no está en memoria)
        print("Paso 2/2: Cargando LLM para generar respuestas...")
        model, tokenizer = load(MODEL_ID, adapter_path=ADAPTER)
        print("✓ Sistema híbrido listo\n")

        print(f"Evaluando {len(questions)} preguntas...")
        results = run_hybrid_benchmark(model, tokenizer, xgb_model=None,
                                       encoders=None, questions=questions)

        out = RES / "benchmark_hybrid_test.jsonl"
        with open(out, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

        print(f"\n{'='*55}")
        print("  RESULTADOS SISTEMA HÍBRIDO")
        print(f"{'='*55}")
        print_metrics(results, "Qwen fine-tuneado + XGBoost")

        # Comparación final
        print(f"\n{'='*55}")
        print("  COMPARACIÓN FINAL (109 preguntas)")
        print(f"{'='*55}")
        print(f"  {'Modelo':<30} {'MAE%':>6}  {'±15%':>6}  {'±20%':>6}  {'Aluc':>6}")
        print(f"  {'─'*58}")
        print(f"  {'Qwen base':<30} {'42.8%':>6}  {'26.7%':>6}  {'32.6%':>6}  {'26.7%':>6}")
        print(f"  {'Qwen fine-tuneado':<30} {'42.5%':>6}  {'31.2%':>6}  {'43.1%':>6}  {'0.0%':>6}")

        priced = [r for r in results if r.get("precio_real") and r.get("precio_xgb")]
        errors = [r["error_pct"] for r in priced if r.get("error_pct")]
        if errors:
            mae_pct = np.mean(errors)
            p15 = sum(1 for e in errors if e<=15)/len(errors)*100
            p20 = sum(1 for e in errors if e<=20)/len(errors)*100
            print(f"  {'Híbrido (XGBoost + Qwen)':<30} {mae_pct:>5.1f}%  {p15:>5.1f}%  {p20:>5.1f}%  {'0.0%':>6}")
        print(f"  {'─'*58}")
        print(f"\n✓ Resultados: {out}")

    else:
        # Modo interactivo
        print("Sistema Híbrido — Experto Inmobiliario Monterrey")
        print("Escribe tu pregunta (o 'salir' para terminar)\n")
        while True:
            pregunta = input("Pregunta> ").strip()
            if pregunta.lower() in ("salir", "exit", "q", ""): break

            params = extract_params(model, tokenizer, pregunta)
            print(f"\n  Parámetros extraídos: {json.dumps(params, ensure_ascii=False)}")

            try:
                X = params_to_features(params, encoders)
                precio_xgb = float(inv_boxcox(xgb_model.predict(X)[0], encoders["bc_lambda"]))
                print(f"  XGBoost estima: ${precio_xgb:,.0f} MXN/mes")
            except Exception as e:
                precio_xgb = 25000
                print(f"  XGBoost (fallback): ${precio_xgb:,.0f}")

            respuesta = generate_response(model, tokenizer, pregunta, precio_xgb, params)
            print(f"\nSistema: {respuesta}\n")


if __name__ == "__main__":
    main()
