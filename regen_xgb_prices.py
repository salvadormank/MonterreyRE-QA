"""
regen_xgb_prices.py — Regenera precios XGBoost con Box-Cox para las 109 preguntas de test.
Actualiza benchmark_base_hybrid_test.jsonl preservando todos los demás campos.
"""
import json, re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import xgboost as xgb

BASE      = Path(__file__).resolve().parent
WEBSIGHTS = Path(Path(__file__).resolve().parent.parent / "propiedades_enriquecido.xlsx")
RES       = BASE / "results"

# ── Encoder con Box-Cox ────────────────────────────────────────────────────
df = pd.read_excel(WEBSIGHTS)
p99 = df["price"].quantile(0.99)
df  = df[df["price"].between(4500, p99)].copy()

def parse_loc(s):
    if not isinstance(s, str): return None, None
    parts = [x.strip() for x in s.split(",")]
    return (parts[0] if parts else None), (parts[1] if len(parts) > 1 else None)

df[["colonia","municipio"]] = pd.DataFrame(df["location"].apply(parse_loc).tolist(), index=df.index)
bc_values, bc_lambda = boxcox(df["price"].values)
df["bc_price"] = bc_values
global_mean = df["bc_price"].mean()
k = 10
encoders = {}
for cat in ["colonia", "municipio"]:
    c = df.groupby(cat)["bc_price"].agg(["mean", "count"])
    sm = (c["mean"] * c["count"] + global_mean * k) / (c["count"] + k)
    encoders[cat] = {"map": sm.to_dict(), "default": global_mean}
print(f"  Box-Cox lambda: {bc_lambda:.4f}")

# ── Modelo ────────────────────────────────────────────────────────────────
model = xgb.XGBRegressor()
model.load_model(str(RES / "xgb_model.json"))
print(f"  Modelo cargado ({model.n_features_in_} features)")

MEDIANS = {"m2": 85.0, "recamaras": 2.0, "banos": 2.0, "estacionamientos": 1.0,
           "lat": 25.65, "lon": -100.30, "userViews": 25.0, "days_listed": 22.0,
           "amenidades_count": 2.0}

ALL_FEATURES = [
    "m2","recamaras","banos","estacionamientos",
    "lat","lon","userViews","days_listed","amenidades_count",
    "Lujo","Amueblado","Nuevo",
    "pill_gym","pill_pool","pill_garden","pill_garden","pill_security",
    "pill_elevator","pill_terrace","pill_rooftop","pill_playground",
    "colonia_enc","municipio_enc"
]

def params_to_features(params):
    row = {f: 0.0 for f in ALL_FEATURES}
    for col in ["m2", "recamaras", "banos", "estacionamientos"]:
        v = params.get(col)
        row[col] = float(v) if v else MEDIANS[col]
    row.update({k: MEDIANS[k] for k in ["lat","lon","userViews","days_listed"]})
    row["Lujo"]      = float(params.get("lujo", 0) or 0)
    row["Amueblado"] = float(params.get("amueblado", 0) or 0)
    row["Nuevo"]     = float(params.get("nuevo", 0) or 0)
    row["amenidades_count"] = 0.0
    colonia   = params.get("colonia") or ""
    municipio = params.get("municipio") or "Monterrey"
    enc_map = encoders["colonia"]["map"]
    row["colonia_enc"] = (enc_map.get(colonia)
                          or enc_map.get(colonia.title())
                          or enc_map.get(colonia.capitalize())
                          or next((v for kk, v in enc_map.items() if colonia.lower() in kk.lower()), None)
                          or encoders["colonia"]["default"])
    row["municipio_enc"] = encoders["municipio"]["map"].get(municipio, encoders["municipio"]["default"])
    return np.array([[row[f] for f in ALL_FEATURES]])

# ── Leer y actualizar benchmark_base_hybrid_test.jsonl ────────────────────
records = [json.loads(l) for l in open(RES / "benchmark_base_hybrid_test.jsonl")]
updated = 0
for r in records:
    params = r.get("params", {})
    # También usar colonia/municipio del benchmark directamente
    if not params.get("colonia"):
        test_q = next((json.loads(l) for l in open(BASE/"data/benchmark_test.jsonl")
                       if json.loads(l)["id"] == r["id"]), None)
        if test_q:
            params["colonia"]   = test_q.get("colonia")
            params["municipio"] = test_q.get("municipio")
            params["m2"]        = test_q.get("m2")
            params["recamaras"] = test_q.get("recamaras")
            params["amueblado"] = test_q.get("amueblado", 0)
            params["lujo"]      = test_q.get("lujo", 0)
    try:
        X = params_to_features(params)
        precio_nuevo = float(inv_boxcox(model.predict(X)[0], bc_lambda))
        precio_viejo = r["precio_xgb"]
        r["precio_xgb"] = precio_nuevo
        updated += 1
    except Exception as e:
        print(f"  Error en {r['id']}: {e}")

with open(RES / "benchmark_base_hybrid_test.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

print(f"\n✓ {updated}/{len(records)} precios actualizados con Box-Cox")
print(f"  Archivo: {RES}/benchmark_base_hybrid_test.jsonl")

# ── Muestra algunas diferencias ────────────────────────────────────────────
print("\nEjemplos (precio_real | precio_xgb_nuevo):")
for r in records[:5]:
    print(f"  {r['id']}: real=${r['precio_real']:,.0f}  xgb=${r['precio_xgb']:,.0f}")
