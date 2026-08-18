"""
prepare_finetune_data.py
────────────────────────────────────────────────────────────────────────────
Convierte propiedades_enriquecido.xlsx a pares instrucción/respuesta
en formato mlx-lm (JSONL) para fine-tuning de Qwen2.5-7B-Instruct.

Salida:
  data/train.jsonl   (~870 ejemplos, 80%)
  data/valid.jsonl   (~218 ejemplos, 20%)
  data/test.jsonl    (~109 ejemplos, 10% del valid)

Cada ejemplo enseña al modelo:
  - Dado características de una propiedad en Monterrey → estimar precio + razonar
────────────────────────────────────────────────────────────────────────────
"""

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

random.seed(42)
np.random.seed(42)

BASE  = Path(__file__).resolve().parent
INPUT = Path(Path(__file__).resolve().parent.parent / "propiedades_enriquecido.xlsx")
DATA  = BASE / "data"
DATA.mkdir(exist_ok=True)

# ── Carga y filtro básico ─────────────────────────────────────────────────
df = pd.read_excel(INPUT)
p99 = df["price"].quantile(0.99)
df  = df[df["price"].between(4_500, p99)].copy()
df["userViews"] = pd.to_numeric(df["userViews"], errors="coerce")

def parse_location(s):
    if not isinstance(s, str): return None, None
    parts = [p.strip() for p in s.split(",")]
    return (parts[0] if parts else None), (parts[1] if len(parts) > 1 else None)

df[["colonia", "municipio"]] = pd.DataFrame(
    df["location"].apply(parse_location).tolist(), index=df.index
)

# Amenidades de featuresPills
pill_cols = [c for c in df.columns if c.startswith("featuresPills")]
def get_amenidades(row):
    pills = [str(row[c]).strip() for c in pill_cols
             if pd.notna(row[c]) and str(row[c]).lower() not in ("nan","")]
    return pills

df["amenidades_lista"] = df.apply(get_amenidades, axis=1)

# ── Helpers para texto ────────────────────────────────────────────────────
def precio_rango(p):
    """Convierte precio a rango narrativo."""
    if p < 10_000:   return f"${p:,.0f}"
    if p < 20_000:   return f"${p:,.0f} (segmento económico-medio)"
    if p < 30_000:   return f"${p:,.0f} (segmento medio)"
    if p < 45_000:   return f"${p:,.0f} (segmento medio-alto)"
    return f"${p:,.0f} (segmento alto/lujo)"

def bool_txt(val, si="Sí", no="No"):
    if pd.isna(val): return "No especificado"
    return si if val == 1 else no

def num_txt(val, unit=""):
    if pd.isna(val): return "No especificado"
    return f"{int(val)}{unit}"

ZONAS_CARAS = {"valle oriente","santa bárbara","cumbres","haciendas de la sierra",
               "del valle","carrizalejo","cumbres elite","residencial santa bárbara"}
ZONAS_MEDIAS = {"contry","roma","mitras","obispado","san pedro garza garcía",
                "del paseo residencial","tecnológico","loma linda"}

def zona_contexto(colonia):
    if not isinstance(colonia, str): return ""
    c = colonia.lower()
    if any(z in c for z in ZONAS_CARAS):
        return f"{colonia} es una de las zonas premium de Monterrey, con alta demanda y precios elevados."
    if any(z in c for z in ZONAS_MEDIAS):
        return f"{colonia} es una zona consolidada de clase media-alta en Monterrey."
    return f"{colonia} es una colonia de Monterrey con mercado de renta activo."


# ── Generador de ejemplos ─────────────────────────────────────────────────
SYSTEM = (
    "Eres un experto en el mercado inmobiliario de renta en Monterrey, Nuevo León. "
    "Conoces a fondo las colonias, rangos de precios por zona, y el efecto de "
    "amenidades y características sobre el valor de renta. Siempre respondes en español, "
    "con argumentos concretos basados en el mercado local."
)

PREGUNTAS = [
    "¿Cuál es el precio de renta estimado para esta propiedad y por qué?",
    "¿Cuánto debería costar de renta mensual esta propiedad en el mercado actual?",
    "Analiza esta propiedad y estima su precio de renta.",
    "¿Es razonable el precio de esta propiedad? ¿Cuánto estimarías tú?",
    "Dame un estimado de renta para esta propiedad con tu justificación.",
]

def build_example(row):
    colonia   = row["colonia"] or "Monterrey"
    municipio = row["municipio"] or "Monterrey"
    price     = row["price"]
    m2        = row["m2"]
    rec       = row["recamaras"]
    ban       = row["banos"]
    est       = row["estacionamientos"]
    lujo      = bool_txt(row["Lujo"])
    amueblado = bool_txt(row["Amueblado"])
    nuevo     = bool_txt(row["Nuevo"])
    amenids   = row["amenidades_lista"]
    views     = row["userViews"]

    # Construir USER prompt
    lineas = [
        f"**Colonia:** {colonia}, {municipio}",
        f"**Recámaras:** {num_txt(rec)}",
        f"**Baños:** {num_txt(ban)}",
        f"**Estacionamientos:** {num_txt(est)}",
        f"**Superficie:** {num_txt(m2, ' m²') if pd.notna(m2) else 'No especificada'}",
        f"**Amueblado:** {amueblado}",
        f"**Lujo:** {lujo}",
        f"**Construcción nueva:** {nuevo}",
    ]
    if amenids:
        lineas.append(f"**Amenidades:** {', '.join(amenids)}")
    if pd.notna(views):
        lineas.append(f"**Vistas en portal:** {int(views)}")

    pregunta = random.choice(PREGUNTAS)
    user_msg = "Tengo una propiedad con estas características:\n\n" + \
               "\n".join(lineas) + f"\n\n{pregunta}"

    # Construir ASSISTANT respuesta
    zona_ctx = zona_contexto(colonia)
    precio_t = precio_rango(price)

    factores = []
    if pd.notna(m2) and m2 > 0:
        pm2 = price / m2
        factores.append(f"La superficie de {m2:.0f} m² implica un precio por m² de ${pm2:,.0f}/mes, "
                        f"{'alto' if pm2 > 300 else 'razonable' if pm2 > 150 else 'accesible'} para la zona.")
    if row["Lujo"] == 1:
        factores.append("El atributo de lujo incrementa el precio entre 20-35% respecto a propiedades estándar.")
    if row["Amueblado"] == 1:
        factores.append("Estar amueblado agrega aproximadamente 10-20% al precio de renta base.")
    if row["Nuevo"] == 1:
        factores.append("Al ser construcción nueva, el precio refleja un premium del 5-15% sobre propiedades similares más antiguas.")
    if amenids:
        top = amenids[:3]
        factores.append(f"Amenidades como {', '.join(top)} aumentan el atractivo y el precio de mercado.")
    if pd.notna(views) and views > 50:
        factores.append(f"Con {int(views)} vistas en portal, esta propiedad tiene alta demanda relativa.")

    factores_txt = ""
    if factores:
        factores_txt = "\n\n**Factores que justifican este precio:**\n" + \
                       "\n".join(f"- {f}" for f in factores)

    # Rango ±15%
    low  = price * 0.85
    high = price * 1.15
    rango_txt = f"${low:,.0f}–${high:,.0f} MXN/mes"

    assistant_msg = (
        f"Basándome en el mercado actual de Monterrey, el precio estimado de renta "
        f"para esta propiedad es **{precio_t} MXN/mes**, con un rango de mercado de "
        f"**{rango_txt}** dependiendo de la negociación y condiciones específicas.\n\n"
        f"{zona_ctx}"
        f"{factores_txt}\n\n"
        f"**Recomendación:** "
        f"{'Precio competitivo para el segmento. Publicar a ' + precio_t + ' es adecuado.' if row['Lujo'] != 1 else 'Para el segmento lujo, asegúrate de destacar los acabados y la exclusividad de la zona en el anuncio.'}"
    )

    # Formato messages — soporta mask_prompt en mlx-lm
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


# ── Generar todos los ejemplos ─────────────────────────────────────────────
print(f"Generando ejemplos de {len(df):,} propiedades...")
examples = []
for _, row in df.iterrows():
    try:
        ex = build_example(row)
        if len(json.dumps(ex)) > 200:   # filtrar vacíos
            examples.append(ex)
    except Exception as e:
        pass

random.shuffle(examples)
n = len(examples)
n_train = int(n * 0.80)
n_valid = int(n * 0.10)

train_data = examples[:n_train]
valid_data = examples[n_train:n_train + n_valid]
test_data  = examples[n_train + n_valid:]

print(f"  Train : {len(train_data):,} ejemplos")
print(f"  Valid : {len(valid_data):,} ejemplos")
print(f"  Test  : {len(test_data):,} ejemplos")

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Guardado: {path}")

save_jsonl(train_data, DATA / "train.jsonl")
save_jsonl(valid_data, DATA / "valid.jsonl")
save_jsonl(test_data,  DATA / "test.jsonl")

# Mostrar un ejemplo
print("\n── Ejemplo de entrenamiento ──────────────────────────────────────────")
sample = train_data[0]
for msg in sample["messages"]:
    print(f"[{msg['role']}]: {msg['content'][:300]}...")
    print()
print(f"✓ Datos listos en {DATA}/")
