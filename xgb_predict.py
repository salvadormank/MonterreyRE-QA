"""
xgb_predict.py  — Paso 1 del híbrido
Lee benchmark_test.jsonl, extrae parámetros con regex,
predice precios con XGBoost y guarda en xgb_predictions.json
"""
import re, json, sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import boxcox
from scipy.special import inv_boxcox

BASE  = Path(__file__).resolve().parent
DATA  = BASE / "data"
RES   = BASE / "results"

import xgboost as xgb

model = xgb.XGBRegressor()
model.load_model(str(RES / "xgb_model.json"))

# Encoder de colonias
df = pd.read_excel(Path(__file__).resolve().parent.parent / "propiedades_enriquecido.xlsx")
p99 = df["price"].quantile(0.99)
df  = df[df["price"].between(4500, p99)].copy()
df["userViews"] = pd.to_numeric(df["userViews"], errors="coerce")
df[["colonia","municipio"]] = pd.DataFrame(
    df["location"].apply(lambda s: [p.strip() for p in str(s).split(",")][:2]
                         if isinstance(s,str) else [None,None]).tolist(), index=df.index)
bc_values, bc_lambda = boxcox(df["price"].values)
df["bc_price"] = bc_values
global_mean = df["bc_price"].mean()
k = 10
encoders = {}
for cat in ["colonia","municipio"]:
    c = df.groupby(cat)["bc_price"].agg(["mean","count"])
    encoders[cat] = ((c["mean"]*c["count"]+global_mean*k)/(c["count"]+k)).to_dict()

MEDIANS = {"m2":85,"recamaras":2,"banos":2,"estacionamientos":1,
           "lat":25.65,"lon":-100.30,"userViews":25,"days_listed":22,"amenidades_count":2}
PILLS   = ["pill_gym","pill_pool","pill_garden","pill_garden",  # pill_garden duplicado (bug original)
           "pill_security","pill_elevator","pill_terrace","pill_rooftop","pill_playground"]
ALL_F   = ["m2","recamaras","banos","estacionamientos","lat","lon","userViews",
           "days_listed","amenidades_count","Lujo","Amueblado","Nuevo"]+PILLS+["colonia_enc","municipio_enc"]

def extract(text):
    p = {"lujo":0,"amueblado":0,"nuevo":0,"amenidades":[]}
    m = re.search(r'en\s+([A-ZÁÉÍÓÚÜÑa-záéíóúüñ][A-ZÁÉÍÓÚÜÑa-záéíóúüñ\s\(\)]+?)(?:,|\.|con|\d)', text)
    if m: p["colonia"] = m.group(1).strip()
    m = re.search(r'(\d+)\s*m[²2]', text, re.IGNORECASE)
    if m: p["m2"] = float(m.group(1))
    m = re.search(r'(\d+)\s*rec[áa]mara', text, re.IGNORECASE)
    if m: p["recamaras"] = float(m.group(1))
    m = re.search(r'(\d+(?:\.\d)?)\s*ba[ñn]o', text, re.IGNORECASE)
    if m: p["banos"] = float(m.group(1))
    if re.search(r'lujo|luxury|premium', text, re.IGNORECASE): p["lujo"]=1
    if re.search(r'amueblad', text, re.IGNORECASE): p["amueblado"]=1
    if re.search(r'nuevo|nueva', text, re.IGNORECASE): p["nuevo"]=1
    for kw in ["alberca","gimnasio","elevador","terraza","rooftop","jardín","seguridad"]:
        if kw.lower() in text.lower(): p["amenidades"].append(kw)
    return p

def to_features(p):
    r = {f:0.0 for f in ALL_F}
    for c in ["m2","recamaras","banos","estacionamientos"]:
        r[c] = float(p[c]) if p.get(c) else MEDIANS[c]
    r["lat"]=MEDIANS["lat"]; r["lon"]=MEDIANS["lon"]
    r["userViews"]=MEDIANS["userViews"]; r["days_listed"]=MEDIANS["days_listed"]
    r["Lujo"]=float(p.get("lujo",0)); r["Amueblado"]=float(p.get("amueblado",0)); r["Nuevo"]=float(p.get("nuevo",0))
    kw_map = {"alberca":"pill_pool","gimnasio":"pill_gym","elevador":"pill_elevator",
              "terraza":"pill_terrace","rooftop":"pill_rooftop","jardín":"pill_garden","seguridad":"pill_security"}
    for a in p.get("amenidades",[]):
        if a.lower() in kw_map: r[kw_map[a.lower()]]=1.0
    r["amenidades_count"] = sum(r[f] for f in PILLS)
    col = p.get("colonia",""); mun = p.get("municipio","Monterrey")
    r["colonia_enc"]   = encoders["colonia"].get(col, global_mean)
    r["municipio_enc"] = encoders["municipio"].get(mun, global_mean)
    return np.array([[r[f] for f in ALL_F]])

limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
questions = [json.loads(l) for l in open(DATA/"benchmark_test.jsonl")]
if limit: questions = questions[:limit]

out = []
for q in questions:
    p = extract(q["pregunta"])
    try:
        precio = float(inv_boxcox(model.predict(to_features(p))[0], bc_lambda))
    except:
        precio = None
    out.append({"id": q["id"], "precio_xgb": precio, "precio_real": q.get("precio_real"),
                "params": p, "pregunta": q["pregunta"]})

json.dump(out, open(RES/"xgb_predictions.json","w"), ensure_ascii=False, indent=2)
print(f"✓ {len(out)} predicciones guardadas en results/xgb_predictions.json")
