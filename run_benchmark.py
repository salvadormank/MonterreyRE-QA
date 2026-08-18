"""
run_benchmark.py
────────────────────────────────────────────────────────────────────────────
Evalúa el LLM fine-tuneado vs el modelo base en el benchmark de 909 preguntas.

Modo de uso:
  python3 run_benchmark.py --model finetuned   # Qwen + adaptador LoRA
  python3 run_benchmark.py --model base        # Qwen sin fine-tuning
  python3 run_benchmark.py --model both        # ambos (tarda más)
  python3 run_benchmark.py --model finetuned --subset test  # solo 109 no vistas

Salida:
  results/benchmark_finetuned.jsonl   → respuestas del modelo fine-tuneado
  results/benchmark_base.jsonl        → respuestas del modelo base
  results/benchmark_metrics.json      → métricas comparativas
────────────────────────────────────────────────────────────────────────────
"""

import sys
import re
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, "/opt/homebrew/Cellar/mlx/0.31.2/lib/python3.14/site-packages")

BASE    = Path(__file__).resolve().parent
DATA    = BASE / "data"
RES     = BASE / "results"
RES.mkdir(exist_ok=True)

MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
ADAPTER  = str(BASE / "adapters")

SYSTEM = (
    "Eres un experto en el mercado inmobiliario de renta en Monterrey, Nuevo León. "
    "Conoces a fondo las colonias, rangos de precios por zona, y el efecto de "
    "amenidades y características sobre el valor de renta. Siempre respondes en español, "
    "con argumentos concretos basados en el mercado local."
)

# ── Extractor de precio de la respuesta ───────────────────────────────────
def extract_price(text):
    """Extrae el precio estimado central de la respuesta.
    Prioriza el patrón 'precio estimado es $X' sobre rangos o menciones secundarias.
    """
    # Prioridad 1: patrón explícito "es **$X" o "es $X" (estimado central)
    patterns_priority = [
        r'es\s+\*{0,2}\$\s*([\d,]+(?:\.\d+)?)\s*(?:\([\w\s\-]+\))?\s*(?:MXN|pesos|/mes)?',
        r'estimado[^\$]*\$\s*([\d,]+(?:\.\d+)?)',
        r'estim[ao][^\$]*\$\s*([\d,]+(?:\.\d+)?)',
        r'precio[^$\n]*\$\s*([\d,]+(?:\.\d+)?)\s*(?:MXN|/mes)',
    ]
    for pat in patterns_priority:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = m.group(1).replace(",", "")
            try:
                v = float(val)
                if 1_000 < v < 500_000:
                    return v
            except:
                pass

    # Prioridad 2: primer precio en MXN mencionado
    m = re.search(r'\$\s*([\d,]+(?:\.\d+)?)\s*(?:MXN|pesos|/mes|mensual)', text, re.IGNORECASE)
    if m:
        val = m.group(1).replace(",", "")
        try:
            v = float(val)
            if 1_000 < v < 500_000:
                return v
        except:
            pass
    return None


def error_pct(pred, real):
    if pred is None or real is None or real == 0:
        return None
    return abs(pred - real) / real * 100


def within_range(pred, real, pct=15):
    e = error_pct(pred, real)
    return e is not None and e <= pct


# ── Evaluación de un modelo ───────────────────────────────────────────────
def run_evaluation(model_name, questions, adapter_path=None):
    from mlx_lm import load, generate

    print(f"\n{'='*60}")
    print(f"  Cargando modelo: {model_name}")
    print(f"  Adaptador: {'Sí' if adapter_path else 'No (modelo base)'}")
    print(f"  Preguntas: {len(questions)}")
    print(f"{'='*60}\n")

    model, tokenizer = load(MODEL_ID, adapter_path=adapter_path)
    print("✓ Modelo listo\n")

    results = []
    n = len(questions)

    for i, q in enumerate(questions):
        if i % 50 == 0:
            print(f"  Progreso: {i}/{n} ({i/n*100:.0f}%)")

        prompt = (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{q['pregunta']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        try:
            response = generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=400,
                verbose=False,
            )
        except Exception as e:
            response = f"ERROR: {e}"

        precio_pred = extract_price(response)
        precio_real = q.get("precio_real")

        result = {
            **q,
            "respuesta": response,
            "precio_predicho": precio_pred,
            "error_pct": error_pct(precio_pred, precio_real),
            "dentro_15pct": within_range(precio_pred, precio_real, 15),
            "dentro_20pct": within_range(precio_pred, precio_real, 20),
            "modelo": model_name,
        }
        results.append(result)

    return results


# ── Métricas ──────────────────────────────────────────────────────────────
def compute_metrics(results, label):
    # Solo preguntas con precio real conocido
    priced = [r for r in results if r.get("precio_real") and r.get("precio_predicho")]
    total  = len(results)
    n_priced = len(priced)

    if n_priced == 0:
        return {"label": label, "n_total": total, "n_con_precio": 0}

    errors = [r["error_pct"] for r in priced if r["error_pct"] is not None]
    mae_pct = np.mean(errors) if errors else None

    # MAE en MXN
    abs_errors = [abs(r["precio_predicho"] - r["precio_real"]) for r in priced]
    mae_mxn = np.mean(abs_errors) if abs_errors else None

    # % dentro del rango
    pct_15 = sum(1 for r in priced if r.get("dentro_15pct")) / n_priced * 100
    pct_20 = sum(1 for r in priced if r.get("dentro_20pct")) / n_priced * 100

    # Tasa de alucinación (no extrajo precio)
    n_no_precio = sum(1 for r in results
                      if r.get("precio_real") and not r.get("precio_predicho"))
    hal_rate = n_no_precio / n_priced * 100 if n_priced > 0 else 0

    # Por split
    by_split = {}
    for split in ["test", "train", "manual"]:
        sub = [r for r in priced if r.get("split") == split]
        if sub:
            errs = [r["error_pct"] for r in sub if r["error_pct"] is not None]
            by_split[split] = {
                "n": len(sub),
                "mae_pct": round(float(np.mean(errs)), 2) if errs else None,
                "pct_within_15": round(
                    sum(1 for r in sub if r.get("dentro_15pct")) / len(sub) * 100, 1)
            }

    return {
        "label": label,
        "n_total": total,
        "n_con_precio_real": n_priced,
        "mae_pct": round(float(mae_pct), 2) if mae_pct else None,
        "mae_mxn": round(float(mae_mxn), 2) if mae_mxn else None,
        "pct_within_15pct": round(pct_15, 1),
        "pct_within_20pct": round(pct_20, 1),
        "hallucination_rate_pct": round(hal_rate, 1),
        "by_split": by_split,
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["finetuned", "base", "both"],
                        default="finetuned")
    parser.add_argument("--subset", choices=["test", "train", "manual", "full"],
                        default="test", help="Qué subset evaluar (default: test)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limitar número de preguntas (para pruebas rápidas)")
    args = parser.parse_args()

    # Cargar preguntas
    bench_file = DATA / f"benchmark_{args.subset}.jsonl"
    questions  = [json.loads(l) for l in open(bench_file)]

    if args.limit:
        questions = questions[:args.limit]
        print(f"Modo limitado: {args.limit} preguntas")

    print(f"Benchmark: {bench_file.name} — {len(questions)} preguntas")

    all_metrics = {}

    if args.model in ("finetuned", "both"):
        results_ft = run_evaluation("finetuned", questions, adapter_path=ADAPTER)
        out = RES / f"benchmark_finetuned_{args.subset}.jsonl"
        with open(out, "w") as f:
            for r in results_ft:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"\n✓ Respuestas guardadas: {out}")
        all_metrics["finetuned"] = compute_metrics(results_ft, "Qwen fine-tuneado")

    if args.model in ("base", "both"):
        results_base = run_evaluation("base", questions, adapter_path=None)
        out = RES / f"benchmark_base_{args.subset}.jsonl"
        with open(out, "w") as f:
            for r in results_base:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"\n✓ Respuestas guardadas: {out}")
        all_metrics["base"] = compute_metrics(results_base, "Qwen base (sin fine-tuning)")

    # Guardar métricas
    metrics_path = RES / f"benchmark_metrics_{args.subset}.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    # Imprimir resumen
    print(f"\n{'='*60}")
    print("  RESULTADOS")
    print(f"{'='*60}")
    for key, m in all_metrics.items():
        print(f"\n  {m['label']}")
        print(f"  {'─'*40}")
        print(f"  Preguntas evaluadas  : {m['n_total']}")
        print(f"  Con precio real      : {m.get('n_con_precio_real', 0)}")
        if m.get('mae_pct'):
            print(f"  MAE (%)              : {m['mae_pct']:.1f}%")
        if m.get('mae_mxn'):
            print(f"  MAE (MXN)            : ${m['mae_mxn']:,.0f}")
        if m.get('pct_within_15pct') is not None:
            print(f"  Dentro de ±15%       : {m['pct_within_15pct']:.1f}%")
        if m.get('pct_within_20pct') is not None:
            print(f"  Dentro de ±20%       : {m['pct_within_20pct']:.1f}%")
        if m.get('hallucination_rate_pct') is not None:
            print(f"  Tasa alucinación     : {m['hallucination_rate_pct']:.1f}%")

    print(f"\n✓ Métricas: {metrics_path}")


if __name__ == "__main__":
    main()
