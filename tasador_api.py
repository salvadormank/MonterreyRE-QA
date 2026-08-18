"""
tasador_api.py — API de datos para el skill de Hermes
Uso: python3 tasador_api.py "cuanto cuesta en cumbres con 2 recamaras"
Salida: JSON con precio XGBoost + propiedades similares
"""
import sys, re, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import boxcox
from scipy.special import inv_boxcox

BASE      = Path(__file__).resolve().parent
WEBSIGHTS = Path(Path(__file__).resolve().parent.parent / "propiedades_enriquecido.xlsx")
XGB_MODEL = str(BASE / "results/xgb_model.json")

AMENIDAD_PILLS = {
    "gimnasio": "pill_gym", "alberca": "pill_pool",
    "jardín": "pill_garden", "jardín ": "pill_garden",
    "circuito cerrado": "pill_security", "elevador": "pill_elevator",
    "terraza": "pill_terrace", "rooftop": "pill_rooftop",
    "área de juegos": "pill_playground",
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
    "m2":85.0,"recamaras":2.0,"banos":2.0,"estacionamientos":1.0,
    "lat":25.65,"lon":-100.30,"userViews":25.0,"days_listed":22.0,"amenidades_count":2.0,
}
WORD_NUM = {"una":1,"un":1,"dos":2,"tres":3,"cuatro":4,"cinco":5}


def extract_params(text):
    t = text.lower()
    p = {"colonia": None, "municipio": "Monterrey", "m2": None,
         "recamaras": None, "banos": None, "estacionamientos": None,
         "lujo": 0, "amueblado": 0, "nuevo": 0, "amenidades": []}

    m = re.search(r'en\s+((?:[a-záéíóúüña-z\(\)]+\s?){1,4}?)(?=\s*(?:,|\.|con\s|\d|\?|$))', text, re.IGNORECASE)
    if m: p["colonia"] = m.group(1).strip()

    for field, patterns in [
        ("m2",               [r'(\d+)\s*m[²2]']),
        ("recamaras",        [r'(\d+)\s*rec[áa]mara', r'(una|un|dos|tres|cuatro)\s*rec[áa]mara']),
        ("banos",            [r'(\d+(?:\.\d)?)\s*ba[ñn]o']),
        ("estacionamientos", [r'(\d+)\s*estacionamiento']),
    ]:
        for pat in patterns:
            m2 = re.search(pat, text, re.IGNORECASE)
            if m2:
                val = m2.group(1)
                p[field] = float(WORD_NUM.get(val.lower(), val))
                break

    if re.search(r'lujo|luxury|premium', t): p["lujo"] = 1
    if re.search(r'amueblad', t): p["amueblado"] = 1
    if re.search(r'nuevo|nueva', t): p["nuevo"] = 1
    for kw in ["alberca","gimnasio","elevador","terraza","rooftop","jardín","seguridad"]:
        if kw in t: p["amenidades"].append(kw)
    return p


def build_encoders():
    df = pd.read_excel(WEBSIGHTS)
    p99 = df["price"].quantile(0.99)
    df  = df[df["price"].between(4500, p99)].copy()
    def parse_loc(s):
        if not isinstance(s, str): return None, None
        parts = [x.strip() for x in s.split(",")]
        return (parts[0] if parts else None), (parts[1] if len(parts)>1 else None)
    df[["colonia","municipio"]] = pd.DataFrame(df["location"].apply(parse_loc).tolist(), index=df.index)
    bc_values, bc_lambda = boxcox(df["price"].values)
    df["bc_price"] = bc_values
    gm = df["bc_price"].mean()
    enc = {}
    for cat in ["colonia","municipio"]:
        counts = df.groupby(cat)["bc_price"].agg(["mean","count"])
        sm = (counts["mean"]*counts["count"] + gm*10) / (counts["count"]+10)
        enc[cat] = {"map": sm.to_dict(), "default": gm}
    enc["bc_lambda"] = bc_lambda
    return enc


def params_to_features(p, enc):
    row = {f: 0.0 for f in ALL_FEATURES}
    for col in ["m2","recamaras","banos","estacionamientos"]:
        row[col] = float(p[col]) if p.get(col) else MEDIANS[col]
    row.update({k: MEDIANS[k] for k in ["lat","lon","userViews","days_listed"]})
    row["Lujo"] = float(p.get("lujo",0)); row["Amueblado"] = float(p.get("amueblado",0)); row["Nuevo"] = float(p.get("nuevo",0))
    amenids = [a.lower() for a in (p.get("amenidades") or [])]
    for kw, feat in AMENIDAD_PILLS.items():
        row[feat] = 1.0 if any(kw in a for a in amenids) else 0.0
    row["amenidades_count"] = sum(row[f] for f in set(AMENIDAD_PILLS.values()))
    col = p.get("colonia") or ""
    mun = p.get("municipio") or "Monterrey"
    em  = enc["colonia"]["map"]
    row["colonia_enc"] = (em.get(col) or em.get(col.title()) or em.get(col.capitalize())
                          or next((v for k,v in em.items() if col.lower() in k.lower()), None)
                          or enc["colonia"]["default"])
    row["municipio_enc"] = enc["municipio"]["map"].get(mun, enc["municipio"]["default"])
    return np.array([[row[f] for f in ALL_FEATURES]])


def find_similar(params, train_qs, k=3):
    scored = []
    for t in train_qs:
        s = 0
        if params.get("colonia") and params["colonia"].lower() == (t.get("colonia") or "").lower(): s += 10
        if params.get("municipio","").lower() == (t.get("municipio") or "").lower(): s += 1
        if params.get("recamaras") and t.get("recamaras"): s -= abs(params["recamaras"] - t["recamaras"])
        if params.get("m2") and t.get("m2") and params["m2"]>0 and t["m2"]>0:
            s -= abs(params["m2"] - t["m2"]) / max(params["m2"], t["m2"])
        scored.append((s, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:k]]


if __name__ == "__main__":
    pregunta = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    if not pregunta:
        print(json.dumps({"error": "Sin pregunta"}, ensure_ascii=False))
        sys.exit(1)

    import xgboost as xgb
    params  = extract_params(pregunta)
    enc     = build_encoders()
    model   = xgb.XGBRegressor(); model.load_model(XGB_MODEL)
    X       = params_to_features(params, enc)
    precio  = float(inv_boxcox(model.predict(X)[0], enc["bc_lambda"]))

    train_qs = [json.loads(l) for l in open(BASE / "data/benchmark_train.jsonl")]
    similares = find_similar(params, train_qs, k=3)

    output = {
        "pregunta":    pregunta,
        "params":      params,
        "precio_xgb":  round(precio, 0),
        "similares": [
            {
                "colonia":     s.get("colonia","?"),
                "m2":          s.get("m2"),
                "recamaras":   s.get("recamaras"),
                "amueblado":   s.get("amueblado",0),
                "lujo":        s.get("lujo",0),
                "precio_real": s.get("precio_real",0),
                "url":         s.get("url")
            } for s in similares
        ],
        "nota": "precio_xgb es la estimación estadística basada en datos reales de Monterrey (R²=0.719)"
    }
    print(json.dumps(output, ensure_ascii=False, default=str))
