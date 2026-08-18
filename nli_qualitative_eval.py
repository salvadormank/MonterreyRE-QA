"""
nli_qualitative_eval.py — Evaluación cualitativa con NLI
Paso 1: genera respuestas del modelo base y fine-tuned para las 100 preguntas cualitativas
Paso 2: scorea con xlm-roberta-large-xnli (español multilingüe)
Uso: python3 nli_qualitative_eval.py --model [base|finetuned|score]
"""
import sys, json, re, argparse
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA     = BASE_DIR / "data"
RES      = BASE_DIR / "results"

TIPOS_CUALI = ['comparacion_zona','efecto_amenidad','zona_no_vista',
               'tendencia','robustez','precio_m2']

# ── Hypotheses de referencia por tipo de pregunta ────────────────────────────
# Cada hypothesis define qué debe ser verdad en una buena respuesta
# Se aplica de forma genérica al tipo; la hipótesis es lo que esperamos que el
# modelo ENTAILE (confirme) con su respuesta.

HYPOTHESES = {
    "comparacion_zona": "Hay diferencias de precio de renta entre distintas zonas o colonias de Monterrey",
    "efecto_amenidad":  "Las amenidades como alberca, gimnasio o estacionamiento afectan el precio de renta",
    "zona_no_vista":    "El precio de renta varía según la colonia y las características del inmueble",
    "tendencia":        "Los precios de renta en Monterrey han cambiado en los últimos años",
    "robustez":         "El precio mencionado es inusual o sospechoso para el mercado de Monterrey",
    "precio_m2":        "El precio por metro cuadrado varía según la zona y el tipo de inmueble en Monterrey",
}

def load_qualitative():
    data = [json.loads(l) for l in open(DATA / "benchmark_full.jsonl")]
    return [r for r in data if r.get("tipo") in TIPOS_CUALI]

# ── Paso 1: generar respuestas ────────────────────────────────────────────────
def run_model(use_adapter: bool):
    sys.path.insert(0, "/opt/homebrew/Cellar/mlx/0.31.2/lib/python3.14/site-packages")
    from mlx_lm import load, generate

    MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    ADAPTER  = str(BASE_DIR / "adapters") if use_adapter else None
    label    = "finetuned" if use_adapter else "base"
    out_path = RES / f"qualitative_{label}.jsonl"

    SYSTEM = (
        "Eres un experto en el mercado inmobiliario de renta en Monterrey, Nuevo León. "
        "Respondes en español con argumentos concretos basados en el mercado local."
    )

    print(f"Cargando Qwen {'fine-tuned' if use_adapter else 'base'}...")
    model, tokenizer = load(MODEL_ID, **({"adapter_path": ADAPTER} if ADAPTER else {}))
    print("✓ Listo\n")

    questions = load_qualitative()
    results = []
    for i, q in enumerate(questions):
        if i % 10 == 0:
            print(f"  {i}/{len(questions)} ({i/len(questions)*100:.0f}%)")
        prompt = (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{q['pregunta']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        resp = generate(model, tokenizer, prompt=prompt, max_tokens=250, verbose=False)
        results.append({**q, "respuesta": resp, "modelo": label})

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✓ Guardado: {out_path}")

# ── Paso 2: NLI scoring ───────────────────────────────────────────────────────
def score_nli():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    MODEL_NLI = "joeddav/xlm-roberta-large-xnli"
    print(f"Cargando {MODEL_NLI}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NLI)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NLI)
    model.eval()
    # Labels del modelo: 0=contradiction, 1=neutral, 2=entailment
    print("✓ Listo\n")

    def entail_score(premise, hypothesis):
        """Retorna probabilidad de entailment (premise entaila hypothesis)."""
        inputs = tokenizer(premise[:512], hypothesis, return_tensors="pt",
                           truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = logits.softmax(-1)[0]
        return float(probs[2])  # índice 2 = entailment

    all_results = {}
    for label in ["base", "finetuned"]:
        path = RES / f"qualitative_{label}.jsonl"
        if not path.exists():
            print(f"  ⚠ No encontrado: {path} — corre --model {label} primero")
            continue

        records = [json.loads(l) for l in open(path)]
        scores_by_tipo = {}
        entail_total = []

        for i, r in enumerate(records):
            tipo = r["tipo"]
            hyp  = HYPOTHESES.get(tipo)
            if not hyp:
                continue
            resp = r.get("respuesta", "")
            if not resp.strip():
                entail_total.append(0)
                continue

            score = entail_score(resp, hyp)
            # threshold 0.33: en 3 clases, >0.33 indica que entailment domina
            is_correct = score > 0.33

            if tipo not in scores_by_tipo:
                scores_by_tipo[tipo] = []
            scores_by_tipo[tipo].append(int(is_correct))
            entail_total.append(int(is_correct))

            if (i+1) % 20 == 0:
                print(f"  {i+1}/{len(records)} procesadas...")

        print(f"\n{'='*55}")
        print(f"  NLI ACCURACY — Modelo: {label}")
        print(f"{'='*55}")
        print(f"  {'Tipo':<25} {'Accuracy':>10}  {'n':>4}")
        print(f"  {'─'*42}")
        for tipo, vals in sorted(scores_by_tipo.items()):
            print(f"  {tipo:<25} {np.mean(vals)*100:>9.1f}%  {len(vals):>4}")
        print(f"  {'─'*42}")
        print(f"  {'TOTAL':<25} {np.mean(entail_total)*100:>9.1f}%  {len(entail_total):>4}")

        all_results[label] = {
            "overall": round(float(np.mean(entail_total))*100, 1),
            "by_type": {t: round(float(np.mean(v))*100, 1)
                        for t, v in scores_by_tipo.items()},
            "n": len(entail_total)
        }

    if "base" in all_results and "finetuned" in all_results:
        delta = all_results["finetuned"]["overall"] - all_results["base"]["overall"]
        print(f"\n  Δ (fine-tuned − base): {delta:+.1f}%")

    json.dump(all_results, open(RES / "nli_qualitative_results.json", "w"), indent=2)
    print(f"\n✓ Guardado: results/nli_qualitative_results.json")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base","finetuned","score"], required=True)
    args = parser.parse_args()

    if args.model == "base":
        run_model(use_adapter=False)
    elif args.model == "finetuned":
        run_model(use_adapter=True)
    elif args.model == "score":
        score_nli()
