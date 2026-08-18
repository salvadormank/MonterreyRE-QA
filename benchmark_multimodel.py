"""
benchmark_multimodel.py — 3 models × 3 configs generalizability benchmark
Models : Llama 3.1 8B 4-bit, Mistral 7B Instruct v0.3 4-bit, Salamandra 7B 4-bit
Configs: zero-shot | few-shot (k=3) | few-shot + XGBoost anchor
Output : results/benchmark_{model_tag}_{config}_test.jsonl  (9 files total)
"""
import sys, json, re, gc
import numpy as np
from pathlib import Path

sys.path.insert(0, "/opt/homebrew/Cellar/mlx/0.31.2/lib/python3.14/site-packages")
import mlx.core as mx
from mlx_lm import load, generate

BASE = Path(__file__).resolve().parent
RES  = BASE / "results"
SAL_4BIT = BASE / "models/salamandra-7b-4bit"

# ── Model registry ────────────────────────────────────────────────────────────
# family: "chatml"  → <|im_start|> (Qwen / Salamandra)
#         "llama3"  → <|start_header_id|>
#         "mistral" → [INST] (no system role)
MODELS = [
    {
        "tag":    "llama31_8b",
        "id":     "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "family": "llama3",
        "label":  "Llama 3.1 8B Instruct 4-bit",
    },
    {
        "tag":    "mistral_7b",
        "id":     "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "family": "mistral",
        "label":  "Mistral 7B Instruct v0.3 4-bit",
    },
    {
        "tag":    "salamandra_7b",
        "id":     str(SAL_4BIT) if SAL_4BIT.exists() else "BSC-LT/salamandraTA-7b",
        "family": "chatml",
        "label":  "Salamandra 7B 4-bit" if SAL_4BIT.exists() else "Salamandra 7B bf16 (conversion pending)",
    },
]

SYSTEM = (
    "Eres un tasador inmobiliario experto en Monterrey, México. "
    "Responde ÚNICAMENTE con el precio de renta estimado en MXN/mes. "
    "Formato obligatorio: solo el número, sin explicaciones ni texto adicional. "
    "Ejemplo: 22500"
)

# ── Chat templates ─────────────────────────────────────────────────────────────
def prompt_chatml(system: str, user: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def prompt_llama3(system: str, user: str) -> str:
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def prompt_mistral(system: str, user: str) -> str:
    # Mistral v0.3 has no system role — inject system as preamble in user turn
    return f"<s>[INST] {system}\n\n{user} [/INST]"

TEMPLATE = {
    "chatml":  prompt_chatml,
    "llama3":  prompt_llama3,
    "mistral": prompt_mistral,
}

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
        "mae_pct":   np.mean(errs),
        "p15":       sum(1 for e in errs if e<=15)/len(errs)*100,
        "p20":       sum(1 for e in errs if e<=20)/len(errs)*100,
        "nr_rate":   sum(1 for r in results if r['precio_predicho'] is None)/len(results)*100,
        "n":         len(results),
    }

# ── Load data ─────────────────────────────────────────────────────────────────
test_qs   = [json.loads(l) for l in open(BASE / "data/benchmark_test.jsonl")]
train_qs  = [json.loads(l) for l in open(BASE / "data/benchmark_train.jsonl")]
xgb_prices = {
    r['id']: r['precio_xgb']
    for r in [json.loads(l) for l in open(RES / "benchmark_base_hybrid_test.jsonl")]
}
print(f"Test: {len(test_qs)} preguntas | Train: {len(train_qs)} ejemplos")

# ── Main loop ─────────────────────────────────────────────────────────────────
summary = []

for cfg in MODELS:
    tag     = cfg["tag"]
    model_id = cfg["id"]
    family  = cfg["family"]
    label   = cfg["label"]
    build   = TEMPLATE[family]

    # Skip Salamandra if still converting and only bf16 available (too large)
    if tag == "salamandra_7b" and not SAL_4BIT.exists():
        print(f"\n⚠  Salamandra 4-bit not ready at {SAL_4BIT}")
        print("   Run: python -m mlx_lm.convert --hf-path BSC-LT/salamandraTA-7b \\")
        print(f"        --mlx-path {SAL_4BIT} --q-bits 4")
        print("   Then re-run this script.\n")
        continue

    print(f"\n{'#'*60}")
    print(f"  Cargando: {label}")
    print(f"  ID: {model_id}")
    print(f"{'#'*60}")
    model, tokenizer = load(model_id)
    print("✓ Modelo listo\n")

    n = len(test_qs)

    # ── Config 1: Zero-shot ────────────────────────────────────────────────────
    print(f"[{tag}] Config 1/3: Zero-shot...")
    results_zs = []
    for i, q in enumerate(test_qs):
        if i % 20 == 0:
            print(f"  {i}/{n}")
        prompt = build(SYSTEM, q['pregunta'])
        resp   = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
        pred   = extract_price(resp)
        pr     = q.get('precio_real')
        err    = abs(pred - pr) / pr * 100 if pred and pr else 100.0
        results_zs.append({
            "id": q.get("id"), "precio_real": pr,
            "precio_predicho": pred, "respuesta_raw": resp.strip(),
            "error_pct": err, "modelo": tag, "config": "zeroshot",
        })
    m = metricas(results_zs, f"{label} — ZERO-SHOT")
    summary.append({"model": label, "config": "zeroshot", **m})

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
        prompt = build(SYSTEM, user_msg)
        resp   = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
        pred   = extract_price(resp)
        pr     = q.get('precio_real')
        err    = abs(pred - pr) / pr * 100 if pred and pr else 100.0
        results_fs.append({
            "id": q.get("id"), "precio_real": pr,
            "precio_predicho": pred, "respuesta_raw": resp.strip(),
            "error_pct": err, "modelo": tag, "config": "fewshot",
        })
    m = metricas(results_fs, f"{label} — FEW-SHOT (k=3)")
    summary.append({"model": label, "config": "fewshot", **m})

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
        prompt = build(SYSTEM, user_msg)
        resp   = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
        pred   = extract_price(resp)
        pr     = q.get('precio_real')
        err    = abs(pred - pr) / pr * 100 if pred and pr else 100.0
        results_xgb.append({
            "id": q.get("id"), "precio_real": pr, "precio_xgb": precio_xgb,
            "precio_predicho": pred, "respuesta_raw": resp.strip(),
            "error_pct": err, "modelo": tag, "config": "fewshot_xgb",
        })
    m = metricas(results_xgb, f"{label} — FEW-SHOT + XGBOOST ANCHOR")
    summary.append({"model": label, "config": "fewshot_xgb", **m})

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

    # ── Unload model to free memory ────────────────────────────────────────────
    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()
    print(f"\n✓ {label} — done, memoria liberada")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n\n{'='*80}")
print("  RESUMEN MULTI-MODEL BENCHMARK")
print(f"{'='*80}")
print(f"  {'Model':<35} {'Config':<15} {'MAE%':>6} {'±15%':>6} {'±20%':>6} {'NR%':>6}")
print(f"  {'-'*35} {'-'*15} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
for s in summary:
    print(f"  {s['model'][:35]:<35} {s['config']:<15} "
          f"{s['mae_pct']:>6.1f} {s['p15']:>6.1f} {s['p20']:>6.1f} {s['nr_rate']:>6.1f}")
print(f"{'='*80}")

# Save summary JSON
out = RES / "benchmark_multimodel_summary.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\n✓ Resumen guardado: {out}")
