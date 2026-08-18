"""
run_multi_seed.py — Evaluación multi-semilla para robustez estadística.

Corre las 4 estrategias cabecera con 5 semillas distintas:
  base_zeroshot   : Qwen2.5-7B base, sin contexto
  ft_zeroshot     : Qwen2.5-7B fine-tuned (LoRA), sin contexto
  fewshot_xgb     : Few-shot (k=3) + anchor XGBoost
  hybrid          : Fine-tuned + anchor XGBoost

Por cada semilla se re-ajusta TODO lo que aprende del train:
  Box-Cox lambda, target encoding, XGBoost, pool few-shot, adapter LoRA.

Salidas:
  results/multi_seed/seed_{s}/  → métricas por semilla
  results/multi_seed_results.json → media ± sd entre semillas
  results/multi_seed_results.csv  → tabla lista para LaTeX

Tiempo estimado: ~8 h (dominado por LoRA ~90 min × 5 semillas).
Para reanudar tras interrupción: los adaptadores y métricas ya guardados
se saltan automáticamente.
"""

import json, re, sys, subprocess, yaml, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Rutas ─────────────────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parent
INPUT  = BASE.parent / "propiedades_enriquecido.xlsx"
RES    = BASE / "results"
MS_DIR = RES / "multi_seed"
MS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"

SEEDS = [42, 7, 123, 2024, 31415]

SYSTEM_BASE = (
    "Eres un experto en el mercado inmobiliario de renta en Monterrey, Nuevo León. "
    "Conoces a fondo las colonias, rangos de precios por zona, y el efecto de "
    "amenidades y características sobre el valor de renta. Siempre respondes en español, "
    "con argumentos concretos basados en el mercado local."
)
SYSTEM_HYBRID = (
    "Eres un experto en el mercado inmobiliario de renta en Monterrey, Nuevo León. "
    "Respondes en español con argumentos concretos basados en el mercado local. "
    "Cuando se te proporciona un precio calculado por modelo estadístico, úsalo como base."
)

ALL_FEATURES = [
    "m2", "recamaras", "banos", "estacionamientos",
    "lat", "lon", "userViews", "days_listed", "amenidades_count",
    "Lujo", "Amueblado", "Nuevo",
    "pill_gym", "pill_pool", "pill_garden", "pill_security",
    "pill_elevator", "pill_terrace", "pill_rooftop", "pill_playground",
    "colonia_enc", "municipio_enc",
]

AMENIDAD_MAP = {
    "gimnasio": "pill_gym", "alberca": "pill_pool",
    "jardín": "pill_garden", "jardín ": "pill_garden",
    "circuito cerrado": "pill_security", "elevador": "pill_elevator",
    "terraza": "pill_terrace", "rooftop": "pill_rooftop",
    "área de juegos": "pill_playground",
}

XGB_PARAMS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1,
)

LORA_PARAMS = dict(
    model=MODEL_ID, train=True, fine_tune_type="lora",
    num_layers=8, batch_size=4, iters=600, val_batches=20,
    learning_rate=2e-5, steps_per_report=10, steps_per_eval=50,
    save_every=100, max_seq_length=1024, mask_prompt=True,
)


# ══════════════════════════════════════════════════════════════════════════
# 1. CARGA Y PREPROCESAMIENTO BASE (una sola vez)
# ══════════════════════════════════════════════════════════════════════════

def load_raw():
    df = pd.read_excel(INPUT)
    p99 = df["price"].quantile(0.99)
    df = df[df["price"].between(4_500, p99)].copy()
    df["userViews"] = pd.to_numeric(df["userViews"], errors="coerce")

    # lat/lon
    if "latlon" in df.columns:
        latlon = df["latlon"].str.split(",", expand=True)
        df["lat"] = pd.to_numeric(latlon[0], errors="coerce")
        df["lon"] = pd.to_numeric(latlon[1], errors="coerce")

    # colonia / municipio
    def parse_loc(s):
        if not isinstance(s, str): return None, None
        p = [x.strip() for x in s.split(",")]
        return (p[0] if p else None), (p[1] if len(p) > 1 else None)
    df[["colonia", "municipio"]] = pd.DataFrame(
        df["location"].apply(parse_loc).tolist(), index=df.index)

    # amenidades
    pill_cols = [c for c in df.columns if c.startswith("featuresPills")]
    for feat_col in dict.fromkeys(AMENIDAD_MAP.values()):
        df[feat_col] = 0
    for _, row in df.iterrows():
        pills = [str(row[c]).lower().strip() for c in pill_cols if pd.notna(row[c])]
        for keyword, feat_col in AMENIDAD_MAP.items():
            if any(keyword in p for p in pills):
                df.at[row.name, feat_col] = 1
    df["amenidades_count"] = df[list(dict.fromkeys(AMENIDAD_MAP.values()))].sum(axis=1)

    # days_listed
    def parse_days(s):
        if pd.isna(s): return np.nan
        s = str(s).lower()
        if "hoy" in s or "today" in s: return 0
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else np.nan
    df["days_listed"] = (df["publishedSince"].apply(parse_days)
                         if "publishedSince" in df.columns else 22)

    df = df.reset_index(drop=True)
    print(f"  Datos cargados: {len(df):,} propiedades")
    return df


# ══════════════════════════════════════════════════════════════════════════
# 2. FUNCIONES POR SEMILLA
# ══════════════════════════════════════════════════════════════════════════

def split_data(df, seed):
    idx = df.index.tolist()
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.20, random_state=seed)
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, random_state=seed)
    return df.loc[idx_tr], df.loc[idx_va], df.loc[idx_te]


def fit_xgboost(train_df, seed):
    """Ajusta Box-Cox + target encoding + XGBoost sobre train_df."""
    bc_values, lam = boxcox(train_df["price"].values)
    train_df = train_df.copy()
    train_df["bc_price"] = bc_values
    gm = train_df["bc_price"].mean()
    k = 10
    enc = {}
    for cat in ["colonia", "municipio"]:
        c = train_df.groupby(cat)["bc_price"].agg(["mean", "count"])
        enc[cat] = ((c["mean"] * c["count"] + gm * k) / (c["count"] + k))

    def encode(df):
        df = df.copy()
        df["bc_price"] = boxcox(df["price"].values, lmbda=lam)
        for cat in ["colonia", "municipio"]:
            df[cat + "_enc"] = df[cat].map(enc[cat]).fillna(gm)
        feat = df[ALL_FEATURES].copy().astype(float)
        med = feat.median()
        return feat.fillna(med).values, df["bc_price"].values

    X_tr, y_tr = encode(train_df)

    model = xgb.XGBRegressor(**XGB_PARAMS, random_state=seed)
    model.fit(X_tr, y_tr, verbose=False)
    return model, lam, enc, gm


def xgb_predict(model, lam, enc, gm, df):
    """Predice precio (MXN/mes) para un DataFrame."""
    df = df.copy()
    for cat in ["colonia", "municipio"]:
        df[cat + "_enc"] = df[cat].map(enc[cat]).fillna(gm)
    feat = df[ALL_FEATURES].copy().astype(float)
    feat = feat.fillna(feat.median())
    bc_pred = model.predict(feat.values)
    return inv_boxcox(bc_pred, lam)


def build_jsonl(df, path, seed):
    """Genera train.jsonl / valid.jsonl / test.jsonl para LoRA."""
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    def notnan(v):
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    def row_to_text(row):
        attrs = []
        if notnan(row.get("m2")) and row["m2"] > 1:
            attrs.append(f"{row['m2']:.0f} m²")
        if notnan(row.get("recamaras")) and row["recamaras"] > 0:
            attrs.append(f"{int(row['recamaras'])} recámaras")
        if notnan(row.get("banos")) and row["banos"] > 0:
            attrs.append(f"{int(row['banos'])} baños")
        if notnan(row.get("estacionamientos")) and row["estacionamientos"] > 0:
            attrs.append(f"{int(row['estacionamientos'])} estacionamientos")
        amenidades = []
        for kw, col in AMENIDAD_MAP.items():
            if row.get(col, 0) == 1:
                amenidades.append(kw)
        flags = []
        if row.get("Lujo"): flags.append("lujo")
        if row.get("Amueblado"): flags.append("amueblado")
        if row.get("Nuevo"): flags.append("nuevo")
        colonia = row.get("colonia", "zona desconocida")
        mun = row.get("municipio", "Monterrey")
        desc = f"Propiedad en {colonia}, {mun}. " + ", ".join(attrs)
        if flags:
            desc += f". Características: {', '.join(flags)}"
        if amenidades:
            desc += f". Amenidades: {', '.join(amenidades)}"
        return desc

    records = []
    for _, row in df.iterrows():
        desc = row_to_text(row)
        precio = row["price"]
        user_msg = f"{desc}. ¿Cuál sería el precio de renta mensual estimado?"
        asst_msg = (f"Basándome en las características de la propiedad y el mercado "
                    f"de {row.get('colonia', 'Monterrey')}, estimo un precio de renta "
                    f"de ${precio:,.0f} MXN/mes.")
        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_BASE},
                {"role": "user",   "content": user_msg},
                {"role": "assistant", "content": asst_msg},
            ]
        })

    rng.shuffle(records)
    splits = {"train.jsonl": records, "valid.jsonl": [], "test.jsonl": []}
    with open(path / "train.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_lora(data_dir, adapter_dir, seed):
    """Lanza fine-tuning LoRA; omite si el adaptador ya existe."""
    adapter_dir = Path(adapter_dir)
    if (adapter_dir / "adapters.safetensors").exists():
        print(f"    ✓ Adaptador seed={seed} ya existe, omitiendo LoRA")
        return True

    adapter_dir.mkdir(parents=True, exist_ok=True)
    cfg = {**LORA_PARAMS,
           "data": str(data_dir),
           "adapter_path": str(adapter_dir),
           "seed": seed}
    cfg_path = adapter_dir / "lora_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    print(f"    Lanzando LoRA seed={seed} (~90 min)...")
    result = subprocess.run(
        [sys.executable, "-m", "mlx_lm", "lora", "-c", str(cfg_path)],
        cwd=BASE
    )
    if result.returncode != 0:
        print(f"    ✗ LoRA falló para seed={seed}")
        return False
    print(f"    ✓ LoRA completado seed={seed}")
    return True


# ══════════════════════════════════════════════════════════════════════════
# 3. EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════════

def extract_price(text):
    for pat in [
        r'estimado\S*\s+es\s+\$\s*([\d,]+)', r'precio\s+\$\s*([\d,]+)',
        r'\$\s*([\d,]+(?:\.\d+)?)\s*MXN',
        r'([\d,]+(?:\.\d+)?)\s*(?:MXN|pesos)',
        r'\b(\d{4,6})\b',
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            v = float(m.group(1).replace(",", ""))
            if 1_000 <= v <= 500_000:
                return v
    return None


def build_test_questions(test_df, xgb_model, lam, enc, gm):
    """Construye lista de dicts con pregunta + precio_real + precio_xgb."""
    xgb_prices = xgb_predict(xgb_model, lam, enc, gm, test_df)
    questions = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        def _ok(v): return v is not None and not (isinstance(v, float) and np.isnan(v))
        attrs = []
        if _ok(row.get("m2")) and row["m2"] > 1: attrs.append(f"{row['m2']:.0f} m²")
        if _ok(row.get("recamaras")) and row["recamaras"] > 0: attrs.append(f"{int(row['recamaras'])} recámaras")
        if _ok(row.get("banos")) and row["banos"] > 0: attrs.append(f"{int(row['banos'])} baños")
        colonia = row.get("colonia", "zona desconocida")
        mun     = row.get("municipio", "Monterrey")
        pregunta = (f"Propiedad en {colonia}, {mun}. {', '.join(attrs)}. "
                    f"¿Cuál sería el precio de renta mensual?")
        questions.append({
            "colonia":    colonia,
            "municipio":  mun,
            "recamaras":  row.get("recamaras"),
            "m2":         row.get("m2"),
            "amueblado":  row.get("Amueblado", 0),
            "lujo":       row.get("Lujo", 0),
            "precio_real": row["price"],
            "precio_xgb": float(xgb_prices[i]),
            "pregunta":   pregunta,
        })
    return questions


def find_similar(q, pool, k=3):
    scored = []
    for t in pool:
        s = 0
        if q.get("municipio") and q.get("municipio") == t.get("municipio"): s += 1
        if q.get("colonia")   and q.get("colonia")   == t.get("colonia"):   s += 10
        if q.get("recamaras") and t.get("recamaras"):
            s -= abs(q["recamaras"] - t["recamaras"])
        if q.get("m2") and t.get("m2") and q["m2"] > 0 and t["m2"] > 0:
            s -= abs(q["m2"] - t["m2"]) / max(q["m2"], t["m2"])
        scored.append((s, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:k]]


def metricas(preds, reales):
    preds  = np.array(preds, dtype=float)
    reales = np.array(reales, dtype=float)
    errs   = np.abs(preds - reales) / reales * 100
    return {
        "mae_pct":  float(np.mean(errs)),
        "acc_15":   float(np.mean(errs <= 15) * 100),
        "acc_20":   float(np.mean(errs <= 20) * 100),
        "n":        len(preds),
    }


def evaluate_seed(seed, test_qs, train_pool, xgb_model, lam, enc, gm,
                  adapter_dir, model_base, tokenizer_base):
    from mlx_lm import load, generate

    results = {}

    # ── Base zero-shot ────────────────────────────────────────────────────
    print("    Evaluando base_zeroshot...")
    preds, reales = [], []
    for q in test_qs:
        prompt = (f"<|im_start|>system\n{SYSTEM_BASE}<|im_end|>\n"
                  f"<|im_start|>user\n{q['pregunta']}<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        resp = generate(model_base, tokenizer_base, prompt=prompt,
                        max_tokens=400, verbose=False)
        pred = extract_price(resp)
        if pred:
            preds.append(pred)
            reales.append(q["precio_real"])
    results["base_zeroshot"] = metricas(preds, reales)
    results["base_zeroshot"]["non_response"] = len(test_qs) - len(preds)

    # ── Few-shot + XGBoost ────────────────────────────────────────────────
    print("    Evaluando fewshot_xgb...")
    preds, reales = [], []
    for q in test_qs:
        ejemplos = find_similar(q, train_pool, k=3)
        ej_text = ""
        def _v(v): return v is not None and not (isinstance(v, float) and np.isnan(v))
        for i, e in enumerate(ejemplos, 1):
            attrs = []
            if _v(e.get("m2")) and e["m2"] > 1: attrs.append(f"{e['m2']:.0f}m²")
            if _v(e.get("recamaras")) and e["recamaras"] > 0: attrs.append(f"{int(e['recamaras'])} rec")
            if e.get("amueblado"): attrs.append("amueblado")
            if e.get("lujo"): attrs.append("lujo")
            ej_text += (f"  {i}. {e.get('colonia','?')}, {e.get('municipio','?')} "
                        f"— {', '.join(attrs)} → ${e['precio_real']:,.0f} MXN/mes\n")
        prompt = (
            f"<|im_start|>system\n{SYSTEM_BASE}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Propiedades similares en el mercado de Monterrey:\n{ej_text}\n"
            f"Precio estadístico (XGBoost): ${q['precio_xgb']:,.0f} MXN/mes\n\n"
            f"Basándote en los ejemplos y el precio estadístico, "
            f"responde SOLO con el número: {q['pregunta']}"
            f"<|im_end|>\n<|im_start|>assistant\n"
        )
        resp = generate(model_base, tokenizer_base, prompt=prompt,
                        max_tokens=20, verbose=False)
        pred = extract_price(resp) or q["precio_xgb"]
        preds.append(pred)
        reales.append(q["precio_real"])
    results["fewshot_xgb"] = metricas(preds, reales)

    # ── Fine-tuned zero-shot + Hybrid ────────────────────────────────────
    adapter_path = str(adapter_dir)
    if (adapter_dir / "adapters.safetensors").exists():
        print("    Cargando modelo fine-tuned...")
        model_ft, tokenizer_ft = load(MODEL_ID, adapter_path=adapter_path)

        # ft zero-shot
        print("    Evaluando ft_zeroshot...")
        preds, reales = [], []
        for q in test_qs:
            prompt = (f"<|im_start|>system\n{SYSTEM_BASE}<|im_end|>\n"
                      f"<|im_start|>user\n{q['pregunta']}<|im_end|>\n"
                      f"<|im_start|>assistant\n")
            resp = generate(model_ft, tokenizer_ft, prompt=prompt,
                            max_tokens=400, verbose=False)
            pred = extract_price(resp)
            if pred:
                preds.append(pred)
                reales.append(q["precio_real"])
        results["ft_zeroshot"] = metricas(preds, reales)
        results["ft_zeroshot"]["non_response"] = len(test_qs) - len(preds)

        # hybrid
        print("    Evaluando hybrid...")
        preds, reales = [], []
        for q in test_qs:
            prompt = (
                f"<|im_start|>system\n{SYSTEM_HYBRID}<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Pregunta: {q['pregunta']}\n\n"
                f"Precio estimado por modelo estadístico (XGBoost): "
                f"${q['precio_xgb']:,.0f} MXN/mes\n\n"
                f"Responde explicando este precio para {q['colonia']}, "
                f"citando que es basado en datos reales."
                f"<|im_end|>\n<|im_start|>assistant\n"
            )
            resp = generate(model_ft, tokenizer_ft, prompt=prompt,
                            max_tokens=300, verbose=False)
            pred = extract_price(resp) or q["precio_xgb"]
            preds.append(pred)
            reales.append(q["precio_real"])
        results["hybrid"] = metricas(preds, reales)

        del model_ft, tokenizer_ft
    else:
        print(f"    ⚠ Adaptador no encontrado para seed={seed}, omitiendo ft/hybrid")
        results["ft_zeroshot"] = None
        results["hybrid"]      = None

    return results


# ══════════════════════════════════════════════════════════════════════════
# 4. LOOP PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  EVALUACIÓN MULTI-SEMILLA")
    print(f"  Seeds: {SEEDS}")
    print("=" * 60)

    print("\nCargando datos base...")
    df_raw = load_raw()

    # Cargar modelo base una sola vez (sin adaptador)
    print("\nCargando modelo base (sin adaptador)...")
    from mlx_lm import load
    model_base, tokenizer_base = load(MODEL_ID)
    print("  ✓ Modelo base listo")

    all_results = {}

    for seed in SEEDS:
        seed_dir     = MS_DIR / f"seed_{seed}"
        metrics_file = seed_dir / "metrics.json"
        adapter_dir  = BASE / "adapters" / f"seed_{seed}"
        data_dir     = BASE / "data"     / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─'*60}")
        print(f"  SEED = {seed}")
        print(f"{'─'*60}")

        # Saltar si ya evaluamos esta semilla completa
        if metrics_file.exists():
            print(f"  ✓ Métricas ya existen para seed={seed}, cargando...")
            all_results[seed] = json.loads(metrics_file.read_text())
            continue

        # 1. Split
        train_df, val_df, test_df = split_data(df_raw, seed)
        print(f"  Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        # 2. Ajustar XGBoost sobre train
        print("  Ajustando XGBoost...")
        xgb_model, lam, enc, gm = fit_xgboost(train_df, seed)
        print(f"  ✓ XGBoost listo (λ={lam:.4f})")

        # 3. Generar JSONL para LoRA (solo train; val/test son preguntas de benchmark)
        print("  Generando datos LoRA...")
        build_jsonl(train_df, data_dir, seed)
        # valid.jsonl necesario para mlx-lm — usamos val_df
        with open(data_dir / "valid.jsonl", "w") as f:
            for _, row in val_df.iterrows():
                f.write(json.dumps({"messages": [
                    {"role": "system",    "content": SYSTEM_BASE},
                    {"role": "user",      "content": f"Propiedad en {row.get('colonia','?')}. ¿Precio de renta?"},
                    {"role": "assistant", "content": f"${row['price']:,.0f} MXN/mes"},
                ]}, ensure_ascii=False) + "\n")

        # 4. Fine-tuning LoRA
        run_lora(data_dir, adapter_dir, seed)

        # 5. Construir pool few-shot (solo desde train)
        train_pool = build_test_questions(train_df, xgb_model, lam, enc, gm)

        # 6. Preguntas de test con precio_xgb
        test_qs = build_test_questions(test_df, xgb_model, lam, enc, gm)
        print(f"  Test: {len(test_qs)} preguntas")

        # 7. Evaluar
        print("  Evaluando estrategias...")
        metrics = evaluate_seed(
            seed, test_qs, train_pool, xgb_model, lam, enc, gm,
            adapter_dir, model_base, tokenizer_base
        )
        metrics["n_train"] = len(train_df)
        metrics["n_test"]  = len(test_df)

        # Guardar por semilla
        metrics_file.write_text(json.dumps(metrics, indent=2))
        all_results[seed] = metrics
        print(f"  ✓ Guardado: {metrics_file}")

    # ── Agregar resultados ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RESULTADOS AGREGADOS (media ± sd)")
    print(f"{'='*60}")

    estrategias = ["base_zeroshot", "ft_zeroshot", "fewshot_xgb", "hybrid"]
    metricas_k  = ["mae_pct", "acc_15", "acc_20"]
    summary = {}

    for est in estrategias:
        vals = {mk: [] for mk in metricas_k}
        for seed in SEEDS:
            m = all_results.get(seed, {}).get(est)
            if m:
                for mk in metricas_k:
                    vals[mk].append(m[mk])
        summary[est] = {
            mk: {"mean": round(np.mean(v), 2), "sd": round(np.std(v), 2)}
            for mk, v in vals.items() if v
        }
        # Cuántas semillas preservan el orden fewshot_xgb < hybrid en acc_15
        if est == "hybrid":
            order_ok = sum(
                1 for seed in SEEDS
                if (all_results.get(seed, {}).get("hybrid") and
                    all_results.get(seed, {}).get("fewshot_xgb") and
                    all_results[seed]["hybrid"]["acc_15"] >=
                    all_results[seed]["fewshot_xgb"]["acc_15"])
            )
            summary["order_preserved_hybrid_ge_fewshot"] = f"{order_ok}/{len(SEEDS)}"

    # Imprimir tabla
    print(f"\n  {'Estrategia':<20} {'MAE%':>12}  {'Acc±15%':>12}  {'Acc±20%':>12}")
    print(f"  {'─'*60}")
    for est in estrategias:
        s = summary.get(est, {})
        def fmt(mk):
            if mk not in s: return "  —"
            return f"{s[mk]['mean']:5.1f}±{s[mk]['sd']:.1f}"
        print(f"  {est:<20} {fmt('mae_pct'):>12}  {fmt('acc_15'):>12}  {fmt('acc_20'):>12}")

    if "order_preserved_hybrid_ge_fewshot" in summary:
        print(f"\n  Orden hybrid ≥ fewshot_xgb preservado en "
              f"{summary['order_preserved_hybrid_ge_fewshot']} semillas")

    # Guardar JSON
    out_json = RES / "multi_seed_results.json"
    out_json.write_text(json.dumps(
        {"seeds": SEEDS, "per_seed": {str(k): v for k, v in all_results.items()},
         "summary": summary}, indent=2))
    print(f"\n✓ Guardado: {out_json}")

    # Guardar CSV para LaTeX
    rows = []
    for est in estrategias:
        s = summary.get(est, {})
        rows.append({
            "estrategia": est,
            "mae_pct_mean": s.get("mae_pct", {}).get("mean"),
            "mae_pct_sd":   s.get("mae_pct", {}).get("sd"),
            "acc_15_mean":  s.get("acc_15",  {}).get("mean"),
            "acc_15_sd":    s.get("acc_15",  {}).get("sd"),
            "acc_20_mean":  s.get("acc_20",  {}).get("mean"),
            "acc_20_sd":    s.get("acc_20",  {}).get("sd"),
        })
    pd.DataFrame(rows).to_csv(RES / "multi_seed_results.csv", index=False)
    print(f"✓ Guardado: {RES / 'multi_seed_results.csv'}")


if __name__ == "__main__":
    main()
