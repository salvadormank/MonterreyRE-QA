"""
baseline_fewshot.py — Few-shot prompting baseline
Para cada pregunta de test, busca los 3 ejemplos más similares del train
(misma colonia > m² cercano > recámaras) y los inyecta en el prompt.
"""
import sys, json, re
import numpy as np
from pathlib import Path

sys.path.insert(0, "/opt/homebrew/Cellar/mlx/0.31.2/lib/python3.14/site-packages")
from mlx_lm import load, generate

BASE = Path(__file__).resolve().parent
RES  = BASE / "results"

MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"

SYSTEM = (
    "Eres un tasador inmobiliario experto en Monterrey, México. "
    "Responde ÚNICAMENTE con el precio de renta estimado en MXN/mes. "
    "Formato obligatorio: solo el número, sin explicaciones ni texto adicional. "
    "Ejemplo: 22500"
)

def extract_price(text):
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
        # mismo municipio
        if q.get('municipio') and q.get('municipio') == t.get('municipio'):
            score += 1
        # misma colonia
        if q.get('colonia') and q.get('colonia') == t.get('colonia'):
            score += 10
        # recamaras similares
        if q.get('recamaras') and t.get('recamaras'):
            score -= abs(q['recamaras'] - t['recamaras'])
        # m2 similares (normalizado)
        if q.get('m2') and t.get('m2') and q['m2'] > 0 and t['m2'] > 0:
            score -= abs(q['m2'] - t['m2']) / max(q['m2'], t['m2'])
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:k]]

def build_fewshot_prompt(q, ejemplos):
    ej_text = ""
    for i, e in enumerate(ejemplos, 1):
        attrs = []
        if e.get('m2') and e['m2'] > 1:
            attrs.append(f"{e['m2']:.0f}m²")
        if e.get('recamaras'):
            attrs.append(f"{e['recamaras']:.0f} rec")
        if e.get('amueblado'):
            attrs.append("amueblado")
        if e.get('lujo'):
            attrs.append("lujo")
        attr_str = ", ".join(attrs) if attrs else "sin datos"
        colonia = e.get('colonia', 'N/A')
        precio  = e.get('precio_real', 0)
        ej_text += f"Ejemplo {i}: {colonia} — {attr_str} → ${precio:,.0f} MXN/mes\n"

    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Propiedades similares en el mercado de Monterrey:\n{ej_text}\n"
        f"Basándote en estos ejemplos, responde:\n{q['pregunta']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

print("Cargando Qwen BASE con few-shot prompting...")
model, tokenizer = load(MODEL_ID)
print("✓ Listo\n")

test_qs  = [json.loads(l) for l in open(BASE / "data/benchmark_test.jsonl")]
train_qs = [json.loads(l) for l in open(BASE / "data/benchmark_train.jsonl")]

results = []
n = len(test_qs)
for i, q in enumerate(test_qs):
    if i % 10 == 0:
        print(f"  {i}/{n} ({i/n*100:.0f}%)")

    ejemplos = find_similar(q, train_qs, k=3)
    prompt   = build_fewshot_prompt(q, ejemplos)
    resp     = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
    precio_pred = extract_price(resp)
    precio_real = q.get('precio_real')

    err = abs(precio_pred - precio_real) / precio_real * 100 \
          if precio_pred and precio_real else 100.0

    results.append({
        "id": q.get("id"),
        "pregunta": q["pregunta"],
        "precio_real": precio_real,
        "precio_predicho": precio_pred,
        "respuesta_raw": resp.strip(),
        "error_pct": err,
        "modelo": "base_fewshot"
    })

errs  = [r["error_pct"] for r in results]
p_resp = [r for r in results if r["precio_predicho"] is not None]
mae_pct  = np.mean(errs)
p15 = sum(1 for e in errs if e <= 15) / len(errs) * 100
p20 = sum(1 for e in errs if e <= 20) / len(errs) * 100
non_resp = sum(1 for r in results if r["precio_predicho"] is None) / len(results) * 100

print(f"\n{'='*55}")
print("  BASE LLM — FEW-SHOT (3 ejemplos similares)")
print(f"{'='*55}")
print(f"  MAE% (penalizado)  : {mae_pct:.1f}%")
print(f"  Dentro de ±15%     : {p15:.1f}%")
print(f"  Dentro de ±20%     : {p20:.1f}%")
print(f"  Non-response rate  : {non_resp:.1f}%")
print(f"  Respondidas        : {len(p_resp)}/{len(results)}")

out = RES / "benchmark_base_fewshot_test.jsonl"
with open(out, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
print(f"\n✓ Guardado: {out}")
