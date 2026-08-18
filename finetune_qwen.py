"""
finetune_qwen.py
────────────────────────────────────────────────────────────────────────────
Fine-tuning de Qwen2.5-7B-Instruct con LoRA via mlx-lm.
Optimizado para Apple Silicon (Metal GPU).

Uso:
  python3 finetune_qwen.py              # entrena
  python3 finetune_qwen.py --test       # prueba rápida (10 steps)
  python3 finetune_qwen.py --inference  # carga el adaptador y responde

Modelo base: mlx-community/Qwen2.5-7B-Instruct-4bit  (~4 GB)
Adaptador guardado en: adapters/

Requerimientos: mlx-lm >= 0.31
────────────────────────────────────────────────────────────────────────────
"""

import subprocess
import sys
import json
import argparse
from pathlib import Path

BASE     = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
ADAPTER  = BASE / "adapters"
ADAPTER.mkdir(exist_ok=True)

MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

# ── Parámetros LoRA ────────────────────────────────────────────────────────
LORA_CONFIG = {
    "model":           MODEL,
    "train":           True,
    "data":            str(DATA_DIR),
    "adapter_path":    str(ADAPTER),
    "batch_size":      4,
    "iters":           600,        # ~2 epochs sobre 872 ejemplos
    "val_batches":     20,
    "learning_rate":   2e-5,
    "lora_layers":     8,          # capas finales a adaptar
    "lora_rank":       16,
    "lora_scale":      20.0,
    "steps_per_eval":  50,
    "save_every":      100,
    "max_seq_length":  1024,
    "seed":            42,
}

LORA_CONFIG_TEST = {**LORA_CONFIG, "iters": 10, "val_batches": 2}


def run_training(test_mode=False):
    config_file = BASE / "lora_config.yaml"
    n_train = sum(1 for _ in open(DATA_DIR / "train.jsonl"))
    print(f"\n{'='*60}")
    print(f"  Fine-tuning: {MODEL}")
    print(f"  Config     : {config_file}")
    print(f"  Datos      : {DATA_DIR}/train.jsonl ({n_train} ejemplos)")
    if test_mode:
        print("  Modo TEST  : solo 10 pasos")
    print(f"{'='*60}\n")

    cmd = [sys.executable, "-m", "mlx_lm", "lora", "-c", str(config_file)]
    if test_mode:
        cmd += ["--iters", "10", "--val-batches", "2"]

    result = subprocess.run(cmd, cwd=BASE)
    if result.returncode == 0:
        print(f"\n✓ Adaptador guardado en: {ADAPTER}/")
        print("  Próximo paso: python3 finetune_qwen.py --inference")
    else:
        print(f"\n✗ Error durante entrenamiento (código {result.returncode})")


def run_inference():
    """Carga el adaptador y hace inferencia interactiva."""
    import sys
    sys.path.insert(0, "/opt/homebrew/Cellar/mlx/0.31.2/lib/python3.14/site-packages")
    from mlx_lm import load, generate

    adapter_path = str(ADAPTER) if (ADAPTER / "adapters.safetensors").exists() else None
    if adapter_path is None:
        print("✗ No se encontró el adaptador. Corre primero: python3 finetune_qwen.py")
        return

    print(f"Cargando {MODEL} + adaptador...")
    model, tokenizer = load(MODEL, adapter_path=adapter_path)
    print("✓ Modelo listo\n")

    SYSTEM = (
        "Eres un experto en el mercado inmobiliario de renta en Monterrey, Nuevo León. "
        "Conoces a fondo las colonias, rangos de precios por zona, y el efecto de "
        "amenidades y características sobre el valor de renta. Siempre respondes en español, "
        "con argumentos concretos basados en el mercado local."
    )

    print("Escribe las características de una propiedad (o 'salir' para terminar):\n")
    while True:
        user_input = input("Propiedad> ").strip()
        if user_input.lower() in ("salir", "exit", "q"):
            break
        if not user_input:
            continue

        prompt = (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        response = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=512,
            verbose=False,
        )
        print(f"\nQwen: {response}\n")


def fuse_and_export():
    """Fusiona el adaptador LoRA con el modelo base y guarda en modelo_finetuned/."""
    out = BASE / "modelo_finetuned"
    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model",        MODEL,
        "--adapter-path", str(ADAPTER),
        "--save-path",    str(out),
        "--de-quantize",
    ]
    print(f"Fusionando modelo → {out}/")
    subprocess.run(cmd, cwd=BASE)
    print(f"✓ Modelo fusionado guardado en: {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",      action="store_true", help="Solo 10 pasos (validación rápida)")
    parser.add_argument("--inference", action="store_true", help="Modo inferencia interactivo")
    parser.add_argument("--fuse",      action="store_true", help="Fusionar adaptador con modelo base")
    args = parser.parse_args()

    if args.inference:
        run_inference()
    elif args.fuse:
        fuse_and_export()
    else:
        run_training(test_mode=args.test)
