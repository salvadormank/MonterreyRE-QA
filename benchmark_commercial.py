"""
benchmark_commercial.py — Commercial APIs: 3 models × 3 configs
Models : GPT-4o-mini, Claude Haiku 4.5, Claude Sonnet 4.6
Configs: zero-shot | few-shot (k=3) | few-shot + XGBoost anchor
Output : results/benchmark_{model_tag}_{config}_test.jsonl  (9 files total)
"""
import os, json, re, time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

BASE = Path(__file__).resolve().parent
RES  = BASE / "results"

SYSTEM = (
    "Eres un tasador inmobiliario experto en Monterrey, México. "
    "Responde ÚNICAMENTE con el precio de renta estimado en MXN/mes. "
    "Formato obligatorio: solo el número, sin explicaciones ni texto adicional. "
    "Ejemplo: 22500"
)

MODELS = [
    {"tag": "gpt4o_mini",      "label": "GPT-4o-mini",        "provider": "openai",    "id": "gpt-4o-mini"},
    {"tag": "claude_haiku",    "label": "Claude Haiku 4.5",   "provider": "anthropic", "id": "claude-haiku-4-5-20251001"},
    {"tag": "claude_sonnet",   "label": "Claude Sonnet 4.6",  "provider": "anthropic", "id": "claude-sonnet-4-6"},
]

# ── Utilities ─────────────────────────────────────────────────────────────────
def extract_price(text: str):
    patterns = [
        r'^\s*\$?\s*([\d,]+(?:\.\d+)?)\s*$',
        r'\$\s*([\d,]+(?:\.\d+)?)\s*(?:MXN|pesos)?',
        r'([\d,]+(?:\.\d+)?)\s*(?:MXN|pesos)',
        r'\b([\d]{4,6})\b',
    ]
    for p in patterns:
        m = re.search(p, text.strip(), re.IGNORECASE | re.MULTILINE)
        if m:
            val = float(m.group(1).replace(',', ''))
            if 3000 <= val <= 150000:
                return val
    return None

def find_similar(q, train_qs, k=3):
    scored = []
    for t in train_qs:
        score = 0
        if q.get('municipio') and q.get('municipio') == t.get('municipio'):
            score += 1
        if q.get('colonia') and q.get('colonia') == t.get('colonia'):
            score += 10
        if q.get('recamaras') and t.get('recamaras'):
            score -= abs(q['recamaras'] - t['recamaras'])
        if q.get('m2') and t.get('m2') and q['m2'] > 0 and t['m2'] > 0:
            score -= abs(q['m2'] - t['m2']) / max(q['m2'], t['m2'])
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:k]]

def examples_text(items):
    lines = ""
    for i, t in enumerate(items, 1):
        attrs = []
        if t.get('m2') and t['m2'] > 1:
            attrs.append(f"{t['m2']:.0f}m²")
        if t.get('recamaras'):
            attrs.append(f"{t['recamaras']:.0f} rec")
        if t.get('amueblado'):
            attrs.append("amueblado")
        if t.get('lujo'):
            attrs.append("lujo")
        attr_str = ", ".join(attrs) if attrs else "sin datos"
        lines += f"  {i}. {t.get('colonia','?')} — {attr_str} → ${t['precio_real']:,.0f} MXN/mes\n"
    return lines

def call_api(provider, model_id, system, user, retries=3):
    for attempt in range(retries):
        try:
            if provider == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "system", "content": system},
                              {"role": "user",   "content": user}],
                    max_tokens=20, temperature=0,
                )
                return resp.choices[0].message.content.strip()

            elif provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                resp = client.messages.create(
                    model=model_id,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    max_tokens=20,
                )
                return resp.content[0].text.strip()

        except Exception as e:
            print(f"    Error (intento {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return ""

def metricas(results, nombre):
    errs   = [r["error_pct"] for r in results]
    p_resp = [r for r in results if r["precio_predicho"] is not None]
    print(f"\n{'='*60}")
    print(f"  {nombre}")
    print(f"{'='*60}")
    print(f"  MAE% (penalizado)  : {np.mean(errs):.1f}%")
    print(f"  Dentro de ±15%     : {sum(1 for e in errs if e<=15)/len(errs)*100:.1f}%")
    print(f"  Dentro de ±20%     : {sum(1 for e in errs if e<=20)/len(errs)*100:.1f}%")
    print(f"  Non-response rate  : {sum(1 for r in results if r['precio_predicho'] is None)/len(results)*100:.1f}%")
    print(f"  Respondidas        : {len(p_resp)}/{len(results)}")
    return {
        "model": nombre, "mae_pct": np.mean(errs),
        "p15": sum(1 for e in errs if e<=15)/len(errs)*100,
        "p20": sum(1 for e in errs if e<=20)/len(errs)*100,
        "nr_rate": sum(1 for r in results if r['precio_predicho'] is None)/len(results)*100,
    }

# ── Load data ─────────────────────────────────────────────────────────────────
test_qs   = [json.loads(l) for l in open(BASE / "data/benchmark_test.jsonl")]
train_qs  = [json.loads(l) for l in open(BASE / "data/benchmark_train.jsonl")]
xgb_prices = {
    r['id']: r['precio_xgb']
    for r in [json.loads(l) for l in open(RES / "benchmark_base_hybrid_test.jsonl")]
}
print(f"Test: {len(test_qs)} preguntas | Train: {len(train_qs)} ejemplos\n")

# ── Main loop ─────────────────────────────────────────────────────────────────
summary = []

for cfg in MODELS:
    tag      = cfg["tag"]
    provider = cfg["provider"]
    model_id = cfg["id"]
    label    = cfg["label"]

    env_key = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    if not os.environ.get(env_key):
        print(f"⚠  {env_key} no encontrada, saltando {label}")
        continue

    print(f"\n{'#'*60}")
    print(f"  {label}  ({model_id})")
    print(f"{'#'*60}")

    n = len(test_qs)

    # ── Config 1: Zero-shot ────────────────────────────────────────────────────
    print(f"\n[{tag}] Config 1/3: Zero-shot...")
    results_zs = []
    for i, q in enumerate(test_qs):
        if i % 20 == 0:
            print(f"  {i}/{n}")
        raw  = call_api(provider, model_id, SYSTEM, q['pregunta'])
        pred = extract_price(raw)
        pr   = q.get('precio_real')
        err  = abs(pred - pr) / pr * 100 if pred and pr else 100.0
        results_zs.append({
            "id": q.get("id"), "precio_real": pr,
            "precio_predicho": pred, "respuesta_raw": raw,
            "error_pct": err, "modelo": tag, "config": "zeroshot",
        })
        time.sleep(0.1)
    m = metricas(results_zs, f"{label} — ZERO-SHOT")
    summary.append({**m, "config": "zeroshot"})

    # ── Config 2: Few-shot (k=3) ───────────────────────────────────────────────
    print(f"\n[{tag}] Config 2/3: Few-shot (k=3)...")
    results_fs = []
    for i, q in enumerate(test_qs):
        if i % 20 == 0:
            print(f"  {i}/{n}")
        ejemplos = find_similar(q, train_qs, k=3)
        ej_text  = examples_text(ejemplos)
        user_msg = (
            f"Propiedades similares en el mercado de Monterrey:\n{ej_text}\n"
            f"Basándote en estos ejemplos, responde:\n{q['pregunta']}"
        )
        raw  = call_api(provider, model_id, SYSTEM, user_msg)
        pred = extract_price(raw)
        pr   = q.get('precio_real')
        err  = abs(pred - pr) / pr * 100 if pred and pr else 100.0
        results_fs.append({
            "id": q.get("id"), "precio_real": pr,
            "precio_predicho": pred, "respuesta_raw": raw,
            "error_pct": err, "modelo": tag, "config": "fewshot",
        })
        time.sleep(0.1)
    m = metricas(results_fs, f"{label} — FEW-SHOT (k=3)")
    summary.append({**m, "config": "fewshot"})

    # ── Config 3: Few-shot + XGBoost anchor ────────────────────────────────────
    print(f"\n[{tag}] Config 3/3: Few-shot + XGBoost anchor...")
    results_xgb = []
    for i, q in enumerate(test_qs):
        if i % 20 == 0:
            print(f"  {i}/{n}")
        ejemplos   = find_similar(q, train_qs, k=3)
        ej_text    = examples_text(ejemplos)
        precio_xgb = xgb_prices.get(q['id'], 0)
        user_msg = (
            f"Propiedades similares en el mercado:\n{ej_text}\n"
            f"Precio calculado por modelo estadístico (XGBoost): ${precio_xgb:,.0f} MXN/mes\n\n"
            f"Basándote en los ejemplos y el precio estadístico, responde:\n{q['pregunta']}"
        )
        raw  = call_api(provider, model_id, SYSTEM, user_msg)
        pred = extract_price(raw)
        pr   = q.get('precio_real')
        err  = abs(pred - pr) / pr * 100 if pred and pr else 100.0
        results_xgb.append({
            "id": q.get("id"), "precio_real": pr, "precio_xgb": precio_xgb,
            "precio_predicho": pred, "respuesta_raw": raw,
            "error_pct": err, "modelo": tag, "config": "fewshot_xgb",
        })
        time.sleep(0.1)
    m = metricas(results_xgb, f"{label} — FEW-SHOT + XGBOOST ANCHOR")
    summary.append({**m, "config": "fewshot_xgb"})

    # ── Save results ───────────────────────────────────────────────────────────
    for results, config_name in [
        (results_zs,  "zeroshot"),
        (results_fs,  "fewshot"),
        (results_xgb, "fewshot_xgb"),
    ]:
        out = RES / f"benchmark_{tag}_{config_name}_test.jsonl"
        with open(out, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"✓ Guardado: {out}")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n\n{'='*80}")
print("  RESUMEN COMMERCIAL BENCHMARK")
print(f"{'='*80}")
print(f"  {'Model':<30} {'Config':<15} {'MAE%':>6} {'±15%':>6} {'±20%':>6} {'NR%':>6}")
print(f"  {'-'*30} {'-'*15} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
for s in summary:
    print(f"  {s['model'][:30]:<30} {s['config']:<15} "
          f"{s['mae_pct']:>6.1f} {s['p15']:>6.1f} {s['p20']:>6.1f} {s['nr_rate']:>6.1f}")
print(f"{'='*80}")

out = RES / "benchmark_commercial_summary.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\n✓ Resumen guardado: {out}")
