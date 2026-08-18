"""
constitutional_ai.py — SL-CAI pipeline para tasador inmobiliario
Fase 1 de Constitutional AI (Bai et al., 2022):
  1. Toma respuestas iniciales del modelo
  2. El modelo las critica contra la constitución
  3. El modelo las revisa según la crítica
  4. Guarda el dataset revisado para fine-tuning LoRA

Salida: data/train_cai.jsonl  (mismo formato que train.jsonl)
"""
import json, sys, re
from pathlib import Path
from mlx_lm import load, generate

BASE = Path(__file__).resolve().parent

# ── Constitución ────────────────────────────────────────────────────────────

CONSTITUTION = """
1. La respuesta DEBE incluir el precio estimado en MXN/mes de forma explícita.
2. Si los datos son insuficientes, admítelo — nunca inventes un número.
3. Ancla el precio en factores concretos del mercado local de Monterrey (zona, m², amenidades).
4. Sé conciso: máximo 3 oraciones en la respuesta final.
5. No generes código de programación bajo ninguna circunstancia.
6. No exageres ni subestimes el precio para complacer al usuario — sé honesto.
7. Usa datos reales de mercado como referencia, no suposiciones genéricas.
""".strip()

SYSTEM_CRITIC = (
    "Eres un auditor de respuestas sobre el mercado inmobiliario de Monterrey. "
    "Tu tarea es identificar violaciones a los principios constitucionales y sugerir mejoras específicas."
)

SYSTEM_REVISOR = (
    "Eres un experto en el mercado inmobiliario de renta en Monterrey, Nuevo León. "
    "Revisas y mejoras respuestas según principios de calidad y honestidad."
)

# ── Cargar modelo ────────────────────────────────────────────────────────────

print("Cargando modelo Qwen...", flush=True)
model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
print("Modelo cargado.\n", flush=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def chat(system, messages, max_tokens=300):
    msgs = [{"role": "system", "content": system}] + messages
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    return generate(model, tokenizer, prompt=prompt,
                    max_tokens=max_tokens, verbose=False)

def critique(pregunta, respuesta_inicial):
    user_msg = (
        f"Pregunta del usuario:\n{pregunta}\n\n"
        f"Respuesta a auditar:\n{respuesta_inicial}\n\n"
        f"Constitución de calidad:\n{CONSTITUTION}\n\n"
        "Identifica brevemente (1-3 puntos) qué principios viola esta respuesta y cómo mejorarla."
    )
    return chat(SYSTEM_CRITIC, [{"role": "user", "content": user_msg}], max_tokens=200)

def revise(pregunta, respuesta_inicial, critica):
    user_msg = (
        f"Pregunta del usuario:\n{pregunta}\n\n"
        f"Respuesta original:\n{respuesta_inicial}\n\n"
        f"Crítica recibida:\n{critica}\n\n"
        f"Constitución de calidad:\n{CONSTITUTION}\n\n"
        "Escribe una respuesta revisada que corrija los problemas señalados, "
        "siguiendo todos los principios. Máximo 3 oraciones."
    )
    return chat(SYSTEM_REVISOR, [{"role": "user", "content": user_msg}], max_tokens=250)

# ── Pipeline CAI ──────────────────────────────────────────────────────────────

train_data = [json.loads(l) for l in open(BASE / "data/train.jsonl")]
n = len(train_data)
print(f"Procesando {n} ejemplos con Constitutional AI...\n", flush=True)

output_path = BASE / "data/train_cai.jsonl"
log_path    = BASE / "data/train_cai_log.jsonl"

processed = 0
with open(output_path, "w") as f_out, open(log_path, "w") as f_log:
    for i, entry in enumerate(train_data):
        if i % 20 == 0:
            print(f"  {i}/{n}", flush=True)

        # Extraer contenido original
        messages  = entry["messages"]
        system    = messages[0]["content"]
        pregunta  = messages[1]["content"]
        resp_orig = messages[2]["content"]

        # Paso 1: Crítica
        critica = critique(pregunta, resp_orig)

        # Paso 2: Revisión
        resp_cai = revise(pregunta, resp_orig, critica)

        # Guardar ejemplo revisado (mismo formato que train.jsonl)
        new_entry = {
            "messages": [
                {"role": "system",    "content": system},
                {"role": "user",      "content": pregunta},
                {"role": "assistant", "content": resp_cai},
            ]
        }
        f_out.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

        # Log para inspección
        f_log.write(json.dumps({
            "id": i,
            "pregunta_short": pregunta[:80],
            "resp_original":  resp_orig[:200],
            "critica":        critica[:300],
            "resp_revisada":  resp_cai[:300],
        }, ensure_ascii=False) + "\n")

        processed += 1

print(f"\n✓ {processed}/{n} ejemplos procesados")
print(f"✓ Dataset CAI guardado: {output_path}")
print(f"✓ Log de auditoría:     {log_path}")
print("\nSiguiente paso:")
print("  mlx_lm.lora --model mlx-community/Qwen2.5-7B-Instruct-4bit \\")
print("    --train --data data/ --train-file train_cai.jsonl \\")
print("    --num-layers 8 --batch-size 4 --iters 600 --lora-rank 8")
