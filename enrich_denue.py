"""
enrich_denue.py — Enriquece propiedades con features del DENUE
Calcula densidad de negocios en radio 500m y 1km para cada propiedad,
agrega las features al dataset y re-entrena XGBoost.

Uso:
  python3 enrich_denue.py
Salida:
  data/propiedades_denue.csv          — dataset enriquecido
  results/xgb_denue_metrics.json      — métricas del modelo mejorado
  results/xgb_denue_model.json        — modelo XGBoost con features DENUE
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RES  = BASE / "results"

DENUE_CSV = DATA / "denue_nl/conjunto_de_datos/denue_inegi_19_.csv"
PROPS_XLS = Path(Path(__file__).resolve().parent.parent / "propiedades_enriquecido.xlsx")


# ── Haversine: distancia en km entre dos puntos GPS ─────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ── Categorías de negocios relevantes para precio de renta ──────────────────

CATEGORIAS = {
    # Proxy de zona premium / NSE alto
    "gym":           ["Centros de acondicionamiento físico"],
    "spa_belleza":   ["Salones y clínicas de belleza"],
    "banco":         ["Banca", "Casas de bolsa", "Instituciones de crédito"],
    "hotel":         ["Hoteles", "Moteles"],
    "clinica_priv":  ["Consultorios de medicina", "sector privado"],

    # Proxy de zona viva / servicios
    "restaurante":   ["Restaurantes", "Cafeterías", "fuentes de sodas"],
    "cafe":          ["Cafeterías", "fuentes de sodas", "neverías"],
    "supermercado":  ["Supermercados", "minisupers", "abarrotes"],
    "farmacia":      ["Farmacias"],

    # Proxy de NSE (educación privada)
    "escuela_priv":  ["sector privado", "Escuelas de educación"],
    "universidad":   ["Instituciones de educación superior", "Universidades"],

    # Proxy de zona comercial
    "plaza_comercial": ["Centros comerciales", "Plazas comerciales"],
    "tienda_depto":    ["Tiendas departamentales"],
}


def match_categoria(nombre_act: str, keywords: list) -> bool:
    n = str(nombre_act).lower()
    return all(kw.lower() in n for kw in keywords)


# ── Carga datos ──────────────────────────────────────────────────────────────

print("Cargando DENUE...")
denue = pd.read_csv(DENUE_CSV, encoding="latin1", low_memory=False,
                    usecols=["nombre_act", "latitud", "longitud"])
denue["latitud"]  = pd.to_numeric(denue["latitud"],  errors="coerce")
denue["longitud"] = pd.to_numeric(denue["longitud"], errors="coerce")
denue = denue.dropna(subset=["latitud", "longitud"])
denue_lat = denue["latitud"].values
denue_lon = denue["longitud"].values
denue_act = denue["nombre_act"].values
print(f"  {len(denue):,} negocios con coordenadas")

print("Cargando propiedades...")
df = pd.read_excel(PROPS_XLS)
p99 = df["price"].quantile(0.99)
df  = df[df["price"].between(4500, p99)].copy()
df["userViews"] = pd.to_numeric(df["userViews"], errors="coerce")

# Extraer lat/lon de la columna "latlon"
latlon_split = df["latlon"].str.split(",", expand=True)
df["lat"] = pd.to_numeric(latlon_split[0], errors="coerce")
df["lon"] = pd.to_numeric(latlon_split[1], errors="coerce")
df = df.dropna(subset=["lat", "lon"])
print(f"  {len(df)} propiedades con coordenadas")

# ── Clasificar negocios DENUE por categoría ──────────────────────────────────

print("Clasificando negocios por categoría...")
cat_masks = {}
for cat, keywords in CATEGORIAS.items():
    mask = np.array([
        any(all(kw.lower() in act.lower() for kw in kws_alt.split(","))
            for kws_alt in [",".join(keywords)])
        for act in denue_act
    ])
    # Más simple: cualquier keyword en el nombre
    mask = np.array([
        any(kw.lower() in str(act).lower() for kw in keywords)
        for act in denue_act
    ])
    cat_masks[cat] = mask
    print(f"  {cat}: {mask.sum():,} negocios")


# ── Calcular features por propiedad ─────────────────────────────────────────

RADIOS = [0.5, 1.0]   # km

print(f"\nCalculando features para {len(df)} propiedades...")
feature_cols = {}

for radio in RADIOS:
    for cat in CATEGORIAS:
        col = f"denue_{cat}_{int(radio*1000)}m"
        feature_cols[col] = []

for i, row in df.iterrows():
    if i % 100 == 0:
        print(f"  {df.index.get_loc(i)}/{len(df)}", end="\r")

    plat, plon = row["lat"], row["lon"]

    # Distancias vectorizadas a todos los negocios
    dlat = np.radians(denue_lat - plat)
    dlon = np.radians(denue_lon - plon)
    a    = np.sin(dlat/2)**2 + np.cos(np.radians(plat)) * np.cos(np.radians(denue_lat)) * np.sin(dlon/2)**2
    dist = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    for radio in RADIOS:
        in_radio = dist <= radio
        for cat, mask in cat_masks.items():
            col = f"denue_{cat}_{int(radio*1000)}m"
            feature_cols[col].append(int((in_radio & mask).sum()))

print(f"\n  Listo.")

# Agregar columnas al dataframe
for col, vals in feature_cols.items():
    df[col] = vals

out_csv = DATA / "propiedades_denue.csv"
df.to_csv(out_csv, index=False)
print(f"✓ Dataset enriquecido: {out_csv} ({len(df)} propiedades, {len(feature_cols)} features nuevas)")

# ── Re-entrenar XGBoost con features DENUE ───────────────────────────────────

print("\nRe-entrenando XGBoost con features DENUE...")
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

# Target encoding colonias
df[["colonia", "municipio"]] = pd.DataFrame(
    df["location"].apply(
        lambda s: [p.strip() for p in str(s).split(",")][:2]
        if isinstance(s, str) else [None, None]
    ).tolist(), index=df.index
)
df["log_price"]  = np.log1p(df["price"])
global_mean      = df["log_price"].mean()
k                = 10
encoders = {}
for cat in ["colonia", "municipio"]:
    c = df.groupby(cat)["log_price"].agg(["mean", "count"])
    encoders[cat] = ((c["mean"] * c["count"] + global_mean * k) / (c["count"] + k)).to_dict()

df["colonia_enc"]   = df["colonia"].map(encoders["colonia"]).fillna(global_mean)
df["municipio_enc"] = df["municipio"].map(encoders["municipio"]).fillna(global_mean)

# Extraer amenidades pills
PILL_MAP = {
    "pill_gym": "gym", "pill_pool": "alberca", "pill_garden": "jardin",
    "pill_security": "seguridad", "pill_elevator": "elevador",
    "pill_terrace": "terraza", "pill_rooftop": "rooftop",
    "pill_playground": "juegos",
}
for pill in PILL_MAP:
    if pill not in df.columns:
        pill_cols = [c for c in df.columns if "featuresPills" in c]
        df[pill] = df[pill_cols].apply(
            lambda row: int(any(pill in str(v).lower() for v in row)), axis=1
        )

BASE_FEATURES = [
    "m2", "recamaras", "banos", "estacionamientos",
    "lat", "lon", "userViews", "amenidades_count",
    "Lujo", "Amueblado", "Nuevo",
    "pill_gym", "pill_pool", "pill_garden", "pill_garden",   # garden duplicado = feature original
    "pill_security", "pill_elevator", "pill_terrace", "pill_rooftop", "pill_playground",
    "colonia_enc", "municipio_enc",
]
DENUE_FEATURES = list(feature_cols.keys())
ALL_FEATURES   = BASE_FEATURES + DENUE_FEATURES

# Calcular amenidades_count si no existe
pill_cols_existing = [c for c in BASE_FEATURES if c.startswith("pill_") and c in df.columns]
df["amenidades_count"] = df[pill_cols_existing].sum(axis=1)

# Calcular días en mercado si existe publishedSince
if "publishedSince" in df.columns:
    df["days_listed"] = pd.to_numeric(
        pd.to_datetime("today") - pd.to_datetime(df["publishedSince"], errors="coerce"),
        errors="coerce"
    ) // 10**9 // 86400
    df["days_listed"] = df["days_listed"].fillna(22)
    BASE_FEATURES.insert(BASE_FEATURES.index("amenidades_count"), "days_listed")

# Preparar matriz
available = [f for f in ALL_FEATURES if f in df.columns]
missing   = [f for f in ALL_FEATURES if f not in df.columns]
if missing:
    print(f"  Features faltantes (se rellenan con 0): {missing}")
    for f in missing:
        df[f] = 0.0

X = df[ALL_FEATURES].fillna(0).values
y = df["log_price"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

model = xgb.XGBRegressor(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1,
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

y_pred_test  = model.predict(X_test)
r2_test      = r2_score(y_test, y_pred_test)
mae_test     = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_test))
rmse_test    = np.sqrt(np.mean((np.expm1(y_test) - np.expm1(y_pred_test))**2))

cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

print(f"\n{'='*50}")
print("  RESULTADOS XGBoost + DENUE")
print(f"{'='*50}")
print(f"  R² test          : {r2_test:.3f}")
print(f"  CV R² (5-fold)   : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"  MAE              : ${mae_test:,.0f} MXN/mes")
print(f"  RMSE             : ${rmse_test:,.0f} MXN/mes")
print(f"  Features totales : {len(ALL_FEATURES)}")
print(f"  Features DENUE   : {len(DENUE_FEATURES)}")

# Feature importance top 15
importance = pd.Series(model.feature_importances_, index=ALL_FEATURES)
print(f"\n  Top 15 features más importantes:")
for feat, imp in importance.nlargest(15).items():
    tag = " ← DENUE" if feat.startswith("denue_") else ""
    print(f"    {feat:<35} {imp:.4f}{tag}")

# Guardar modelo y métricas
model.save_model(str(RES / "xgb_denue_model.json"))
metrics = {
    "r2_test": round(r2_test, 3),
    "cv_r2_mean": round(float(cv_scores.mean()), 3),
    "cv_r2_std":  round(float(cv_scores.std()), 3),
    "mae_mxn":    round(float(mae_test), 0),
    "rmse_mxn":   round(float(rmse_test), 0),
    "n_features":     len(ALL_FEATURES),
    "n_denue_features": len(DENUE_FEATURES),
    "feature_importance": importance.nlargest(20).to_dict(),
}
json.dump(metrics, open(RES / "xgb_denue_metrics.json", "w"), indent=2, ensure_ascii=False)

print(f"\n✓ Modelo guardado: results/xgb_denue_model.json")
print(f"✓ Métricas: results/xgb_denue_metrics.json")

# Guardar encoders para uso en predicción
import pickle
with open(RES / "denue_encoders.pkl", "wb") as f:
    pickle.dump({"encoders": encoders, "global_mean": global_mean,
                 "all_features": ALL_FEATURES, "denue_features": DENUE_FEATURES}, f)
print(f"✓ Encoders: results/denue_encoders.pkl")
