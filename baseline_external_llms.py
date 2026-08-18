"""
baseline_external_llms.py — GPT-4o-mini y Gemini Flash sobre las 109 preguntas
Mismo prompt de tasador que los otros baselines para comparación justa.
"""
import os, sys, json, re, time
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


# ── GPT-4o-mini ───────────────────────────────────────────────────────────────

def run_gpt(test_qs):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    results = []
    n = len(test_qs)

    for i, q in enumerate(test_qs):
        if i % 10 == 0:
            print(f"  GPT-4o-mini: {i}/{n}")
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": q["pregunta"]}
                ],
                max_tokens=20,
                temperature=0,
            )
            raw  = resp.choices[0].message.content.strip()
            pred = extract_price(raw)
        except Exception as e:
            print(f"  Error en {q['id']}: {e}")
            raw, pred = "", None

        precio_real = q.get("precio_real")
        err = abs(pred - precio_real) / precio_real * 100 if pred and precio_real else 100.0
        results.append({
            "id": q["id"], "precio_real": precio_real,
            "precio_predicho": pred, "respuesta_raw": raw,
            "error_pct": err, "modelo": "gpt-4o-mini"
        })
        time.sleep(0.05)  # rate limit gentil

    return results


# ── Gemini Flash ──────────────────────────────────────────────────────────────

def run_gemini(test_qs):
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("  Instala: pip install google-genai")
        return []

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    results = []
    n = len(test_qs)

    for i, q in enumerate(test_qs):
        if i % 10 == 0:
            print(f"  Gemini Flash: {i}/{n}")
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{SYSTEM}\n\n{q['pregunta']}",
                config=types.GenerateContentConfig(
                    max_output_tokens=20,
                    temperature=0,
                ),
            )
            raw  = resp.text.strip()
            pred = extract_price(raw)
        except Exception as e:
            print(f"  Error en {q['id']}: {e}")
            raw, pred = "", None

        precio_real = q.get("precio_real")
        err = abs(pred - precio_real) / precio_real * 100 if pred and precio_real else 100.0
        results.append({
            "id": q["id"], "precio_real": precio_real,
            "precio_predicho": pred, "respuesta_raw": raw,
            "error_pct": err, "modelo": "gemini-flash"
        })
        time.sleep(0.05)

    return results


# ── Métricas ──────────────────────────────────────────────────────────────────

def metricas(results, nombre):
    errs = [r["error_pct"] for r in results]
    resp = [r for r in results if r["precio_predicho"] is not None]
    print(f"\n{'='*55}")
    print(f"  {nombre}")
    print(f"{'='*55}")
    print(f"  MAE% (penalizado)  : {np.mean(errs):.1f}%")
    print(f"  Dentro de ±15%     : {sum(1 for e in errs if e<=15)/len(errs)*100:.1f}%")
    print(f"  Dentro de ±20%     : {sum(1 for e in errs if e<=20)/len(errs)*100:.1f}%")
    print(f"  Non-response rate  : {sum(1 for r in results if r['precio_predicho'] is None)/len(results)*100:.1f}%")
    print(f"  Respondidas        : {len(resp)}/{len(results)}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_qs = [json.loads(l) for l in open(BASE / "data/benchmark_test.jsonl")]
    print(f"Evaluando {len(test_qs)} preguntas...\n")

    # GPT-4o-mini
    if os.environ.get("OPENAI_API_KEY"):
        print("Corriendo GPT-4o-mini...")
        res_gpt = run_gpt(test_qs)
        metricas(res_gpt, "GPT-4o-mini (zero-shot)")
        out = RES / "benchmark_gpt4omini_test.jsonl"
        with open(out, "w") as f:
            for r in res_gpt:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✓ Guardado: {out}")
    else:
        print("⚠ OPENAI_API_KEY no encontrada, saltando GPT-4o-mini")

    # Gemini Flash
    if os.environ.get("GEMINI_API_KEY"):
        print("\nCorriendo Gemini Flash...")
        res_gem = run_gemini(test_qs)
        metricas(res_gem, "Gemini 2.5 Flash (zero-shot)")
        out = RES / "benchmark_gemini_flash_test.jsonl"
        with open(out, "w") as f:
            for r in res_gem:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✓ Guardado: {out}")
    else:
        print("⚠ GEMINI_API_KEY no encontrada, saltando Gemini Flash")
