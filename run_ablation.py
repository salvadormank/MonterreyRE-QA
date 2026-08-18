"""
run_ablation.py — Ablation study sobre cantidad de datos de entrenamiento
Entrena 3 versiones de LoRA con 200, 500 y 872 ejemplos,
evalúa cada una en las 109 preguntas test y guarda métricas comparativas.

Uso:
  python3 run_ablation.py            # corre los 3 experimentos completos
  python3 run_ablation.py --sizes 200 500  # solo algunos tamaños
  python3 run_ablation.py --eval-only      # solo evalúa (ya entrenados)
"""

import sys, json, re, shutil, argparse, subprocess
import numpy as np
from pathlib import Path

sys.path.insert(0, "/opt/homebrew/Cellar/mlx/0.31.2/lib/python3.14/site-packages")

BASE     = Path(__file__).resolve().parent
DATA     = BASE / "data"
RES      = BASE / "results"
ABLATION = BASE / "ablation"
ABLATION.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"

SIZES = [200, 500, 872]   # 872 = entrenamiento completo

SYSTEM = (
    "Eres un experto en el mercado inmobiliario de renta en Monterrey, Nuevo León. "
    "Conoces a fondo las colonias, rangos de precios por zona, y el efecto de "
    "amenidades y características sobre el valor de renta. Siempre respondes en español, "
    "con argumentos concretos basados en el mercado local."
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_price(text):
    patterns = [
        r'es\s+\*{0,2}\$\s*([\d,]+(?:\.\d+)?)\s*(?:\([^)]+\))?\s*(?:MXN|pesos|/mes)?',
        r'estimado[^\$]*\$\s*([\d,]+(?:\.\d+)?)',
        r'precio[^$\n]*\$\s*([\d,]+(?:\.\d+)?)\s*(?:MXN|/mes)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                v = float(m.group(1).replace(",", ""))
                if 1_000 < v < 500_000:
                    return v
            except:
                pass
    m = re.search(r'\$\s*([\d,]+(?:\.\d+)?)\s*(?:MXN|pesos|/mes|mensual)', text, re.IGNORECASE)
    if m:
        try:
            v = float(m.group(1).replace(",", ""))
            if 1_000 < v < 500_000:
                return v
        except:
            pass
    return None


def compute_metrics(results, label):
    priced = [r for r in results if r.get("precio_real") and r.get("precio_predicho")]
    total  = len(results)
    if not priced:
        return {"label": label, "n_total": total, "n_con_precio": 0}
    errors   = [abs(r["precio_predicho"] - r["precio_real"]) / r["precio_real"] * 100 for r in priced]
    mae_mxn  = np.mean([abs(r["precio_predicho"] - r["precio_real"]) for r in priced])
    pct_15   = sum(1 for e in errors if e <= 15) / len(errors) * 100
    pct_20   = sum(1 for e in errors if e <= 20) / len(errors) * 100
    n_no_precio = sum(1 for r in results if r.get("precio_real") and not r.get("precio_predicho"))
    hal_rate = n_no_precio / total * 100
    return {
        "label":               label,
        "n_train":             int(label.split("_")[1]),
        "n_total":             total,
        "n_con_precio_real":   len(priced),
        "mae_pct":             round(float(np.mean(errors)), 2),
        "mae_mxn":             round(float(mae_mxn), 2),
        "pct_within_15pct":    round(pct_15, 1),
        "pct_within_20pct":    round(pct_20, 1),
        "hallucination_rate":  round(hal_rate, 1),
    }


# ── Paso 1: preparar subsets de datos ────────────────────────────────────────

def prepare_data_subset(size):
    """Crea un directorio data con train.jsonl recortado a `size` ejemplos."""
    subset_dir = ABLATION / f"data_{size}"
    subset_dir.mkdir(exist_ok=True)

    full_train = [json.loads(l) for l in open(DATA / "train.jsonl")]
    subset     = full_train[:size]   # primeros N (mismo orden, reproducible)

    with open(subset_dir / "train.jsonl", "w") as f:
        for ex in subset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # valid y test son siempre los mismos
    shutil.copy(DATA / "valid.jsonl", subset_dir / "valid.jsonl")
    shutil.copy(DATA / "test.jsonl",  subset_dir / "test.jsonl")

    print(f"  Subset {size}: {len(subset)} ejemplos en {subset_dir}")
    return subset_dir


# ── Paso 2: entrenar ─────────────────────────────────────────────────────────

def train(size, data_dir):
    adapter_dir = ABLATION / f"adapters_{size}"
    adapter_dir.mkdir(exist_ok=True)

    # Reutiliza si ya existe un adaptador entrenado
    if (adapter_dir / "adapters.safetensors").exists():
        print(f"  Adaptador {size} ya existe — saltando entrenamiento.")
        return adapter_dir

    log_path = ABLATION / f"train_{size}.log"
    cmd = [
        "python3", "-m", "mlx_lm", "lora",
        "--model",        MODEL_ID,
        "--train",
        "--data",         str(data_dir),
        "--adapter-path", str(adapter_dir),
        "--fine-tune-type", "lora",
        "--num-layers",   "8",
        "--batch-size",   "4",
        "--iters",        "600",
        "--val-batches",  "20",
        "--learning-rate","2e-5",
        "--steps-per-eval","50",
        "--save-every",   "100",
        "--max-seq-length","1024",
        "--mask-prompt",
        "--seed",         "42",
    ]

    print(f"\n  Entrenando con {size} ejemplos → log: {log_path}")
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                              cwd=str(BASE))
    if proc.returncode != 0:
        print(f"  ERROR en entrenamiento {size}. Revisa {log_path}")
    else:
        print(f"  Entrenamiento {size} completado.")
    return adapter_dir


# ── Paso 3: evaluar ──────────────────────────────────────────────────────────

def evaluate(size, adapter_dir):
    from mlx_lm import load, generate

    out_path = ABLATION / f"results_{size}.jsonl"
    if out_path.exists():
        print(f"  Resultados {size} ya existen — cargando.")
        return [json.loads(l) for l in open(out_path)]

    print(f"\n  Evaluando modelo con {size} ejemplos...")
    model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_dir))

    questions = [json.loads(l) for l in open(DATA / "benchmark_test.jsonl")]
    results   = []
    n = len(questions)

    for i, q in enumerate(questions):
        if i % 25 == 0:
            print(f"    {i}/{n} ({i/n*100:.0f}%)")

        prompt = (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{q['pregunta']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        try:
            resp = generate(model, tokenizer, prompt=prompt,
                            max_tokens=400, verbose=False)
        except Exception as e:
            resp = f"ERROR: {e}"

        precio_pred = extract_price(resp)
        precio_real = q.get("precio_real")
        err = abs(precio_pred - precio_real) / precio_real * 100 \
              if precio_pred and precio_real else None

        results.append({
            **q,
            "respuesta":       resp,
            "precio_predicho": precio_pred,
            "error_pct":       err,
            "n_train":         size,
        })

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    print(f"  Guardado: {out_path}")
    del model, tokenizer   # libera RAM antes del siguiente experimento
    return results


# ── Paso 4: reporte final ─────────────────────────────────────────────────────

def print_report(all_metrics):
    print(f"\n{'='*65}")
    print("  ABLATION STUDY — Efecto del tamaño de dataset en LoRA")
    print(f"{'='*65}")
    print(f"  {'Ejemplos':>10}  {'MAE%':>7}  {'MAE MXN':>10}  {'±15%':>7}  {'±20%':>7}  {'Aluc%':>7}")
    print(f"  {'─'*60}")
    for m in all_metrics:
        if m.get("n_con_precio_real", 0) == 0:
            continue
        print(f"  {m['n_train']:>10}  {m['mae_pct']:>6.1f}%  ${m['mae_mxn']:>9,.0f}  "
              f"{m['pct_within_15pct']:>6.1f}%  {m['pct_within_20pct']:>6.1f}%  "
              f"{m['hallucination_rate']:>6.1f}%")
    print(f"  {'─'*60}")

    out = RES / "ablation_metrics.json"
    json.dump(all_metrics, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\n✓ Métricas guardadas: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    parser.add_argument("--eval-only", action="store_true",
                        help="Solo evalúa adaptadores ya entrenados")
    args = parser.parse_args()

    print(f"Ablation study: tamaños {args.sizes}")
    print(f"Modo: {'solo evaluación' if args.eval_only else 'entrenar + evaluar'}\n")

    all_metrics = []

    for size in args.sizes:
        print(f"\n{'─'*50}")
        print(f"  EXPERIMENTO: {size} ejemplos de entrenamiento")
        print(f"{'─'*50}")

        # Para 872 reutiliza el adaptador ya entrenado
        if size == 872:
            adapter_dir = BASE / "adapters"
            data_dir    = DATA
        else:
            data_dir    = prepare_data_subset(size)
            if not args.eval_only:
                adapter_dir = train(size, data_dir)
            else:
                adapter_dir = ABLATION / f"adapters_{size}"

        results = evaluate(size, adapter_dir)
        m = compute_metrics(results, f"lora_{size}")
        all_metrics.append(m)
        print(f"  → MAE%={m.get('mae_pct','N/A')}  ±15%={m.get('pct_within_15pct','N/A')}%  "
              f"Alucinación={m.get('hallucination_rate','N/A')}%")

    print_report(all_metrics)


if __name__ == "__main__":
    main()
