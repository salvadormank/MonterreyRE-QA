"""
generate_benchmark.py
────────────────────────────────────────────────────────────────────────────
Genera benchmark de evaluación para el LLM fine-tuneado.
Equivalente a TruthfulQA pero para bienes raíces en Monterrey.

Salida:
  data/benchmark_test.jsonl    → 109 preguntas del test set (no vistas)
  data/benchmark_train.jsonl   → 700 preguntas del train set (vistas)
  data/benchmark_manual.jsonl  → 100 preguntas edge cases (plantilla)
  data/benchmark_full.jsonl    → todos juntos (~909 preguntas)

Formato de cada pregunta:
  {
    "id": "test_001",
    "tipo": "precio_directo",
    "pregunta": "...",
    "precio_real": 22000,
    "colonia": "Valle Oriente",
    "split": "test"   # test / train / manual
  }
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
DATA  = BASE / "data"

# ── Carga datos originales ─────────────────────────────────────────────────
INPUT = Path(Path(__file__).resolve().parent.parent / "propiedades_enriquecido.xlsx")
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

pill_cols = [c for c in df.columns if c.startswith("featuresPills")]
def get_amenidades(row):
    return [str(row[c]).strip() for c in pill_cols
            if pd.notna(row[c]) and str(row[c]).lower() not in ("nan","")]

df["amenidades_lista"] = df.apply(get_amenidades, axis=1)

# ── Templates de preguntas ─────────────────────────────────────────────────
TEMPLATES_PRECIO = [
    "Tengo un {tipo} en {colonia}, {municipio}. Tiene {rec} recámaras, {ban} baños y {m2} m². {extras}¿Cuánto cuesta de renta mensual?",
    "¿Cuánto debería costar de renta un {tipo} de {rec} recámaras en {colonia}? Superficie: {m2} m². {extras}",
    "Busco rentar un {tipo} en {colonia}, {municipio}. {rec} recámaras, {m2} m², {extras}¿Cuál es el precio de mercado?",
    "Dame un estimado de renta para: {tipo} en {colonia}, {rec} rec, {ban} baños, {m2} m². {extras}",
    "Un {tipo} en {colonia} con {rec} recámaras y {m2} m². {extras}¿Cuánto vale la renta?",
]

def num_txt(val, unit=""):
    if pd.isna(val): return "no especificado"
    return f"{int(val)}{unit}"

def build_extras(row):
    extras = []
    if row["Amueblado"] == 1: extras.append("Amueblado")
    if row["Lujo"] == 1: extras.append("Lujo")
    if row["Nuevo"] == 1: extras.append("Construcción nueva")
    if row["amenidades_lista"]: extras += row["amenidades_lista"][:2]
    if extras:
        return ", ".join(extras) + ". "
    return ""

TIPOS = {
    "departamento": "departamento",
    "casa": "casa",
    "oficina": "oficina",
    "local_comercial": "local comercial",
    "bodega_industrial": "bodega",
}

def build_question(row, split, idx):
    colonia   = row["colonia"] or "Monterrey"
    municipio = row["municipio"] or "Monterrey"
    rec  = num_txt(row["recamaras"])
    ban  = num_txt(row["banos"])
    m2   = num_txt(row["m2"], " m²") if pd.notna(row["m2"]) else "superficie no especificada"
    tipo = TIPOS.get(str(row.get("property_type", "departamento")).lower(), "propiedad")
    extras = build_extras(row)

    template = random.choice(TEMPLATES_PRECIO)
    pregunta = template.format(
        tipo=tipo, colonia=colonia, municipio=municipio,
        rec=rec, ban=ban, m2=m2, extras=extras
    )

    return {
        "id": f"{split}_{idx:04d}",
        "tipo": "precio_directo",
        "pregunta": pregunta,
        "precio_real": float(row["price"]),
        "colonia": colonia,
        "municipio": municipio,
        "recamaras": float(row["recamaras"]) if pd.notna(row["recamaras"]) else None,
        "m2": float(row["m2"]) if pd.notna(row["m2"]) else None,
        "amueblado": int(row["Amueblado"]) if pd.notna(row["Amueblado"]) else 0,
        "lujo": int(row["Lujo"]) if pd.notna(row["Lujo"]) else 0,
        "split": split,
    }


# ── Cargar splits originales para saber qué filas son test vs train ────────
train_data = [json.loads(l) for l in open(DATA / "train.jsonl")]
test_data  = [json.loads(l) for l in open(DATA / "test.jsonl")]

# Shuffle del dataframe para asignar splits
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(df_shuffled)
n_train = int(n * 0.80)
n_valid = int(n * 0.10)

df_test  = df_shuffled.iloc[n_train + n_valid:].reset_index(drop=True)
df_train = df_shuffled.iloc[:n_train].reset_index(drop=True)

# ── Generar preguntas test (109) ───────────────────────────────────────────
print(f"Generando preguntas test ({len(df_test)})...")
benchmark_test = []
for i, (_, row) in enumerate(df_test.iterrows()):
    benchmark_test.append(build_question(row, "test", i+1))

# ── Generar preguntas train (700) ──────────────────────────────────────────
print(f"Generando preguntas train (700 de {len(df_train)})...")
df_train_sample = df_train.sample(n=min(700, len(df_train)), random_state=42)
benchmark_train = []
for i, (_, row) in enumerate(df_train_sample.iterrows()):
    benchmark_train.append(build_question(row, "train", i+1))

# ── Preguntas manuales — edge cases (plantilla) ────────────────────────────
print("Generando plantilla de preguntas manuales (100)...")

ZONAS_CARAS = [
    ("Valle Oriente", "San Pedro Garza García"),
    ("Santa Bárbara", "San Pedro Garza García"),
    ("Cumbres Elite", "Monterrey"),
    ("Haciendas de la Sierra", "Monterrey"),
]
ZONAS_MEDIAS = [
    ("Contry", "Monterrey"),
    ("Roma", "Monterrey"),
    ("Tecnológico", "Monterrey"),
    ("Obispado", "Monterrey"),
]
ZONAS_BARATAS = [
    ("Centro", "Monterrey"),
    ("Ladrillera", "Monterrey"),
    ("Chepevera", "Monterrey"),
]

benchmark_manual = []
idx = 1

# Comparación de zonas (25)
comparaciones = [
    ("¿Es más caro rentar en Valle Oriente o en Contry, Monterrey?", "comparacion_zona"),
    ("¿Cuál zona es más barata para rentar, Centro o Tecnológico en Monterrey?", "comparacion_zona"),
    ("¿Vale la pena pagar más por un depa en Santa Bárbara vs Roma?", "comparacion_zona"),
    ("¿Cuánto más caro es Valle Oriente comparado con Chepevera?", "comparacion_zona"),
    ("¿Cuál es la zona más cara para rentar en Monterrey?", "comparacion_zona"),
    ("¿Cuál es la zona más accesible para rentar en Monterrey?", "comparacion_zona"),
    ("¿Es similar el precio en Obispado y Tecnológico?", "comparacion_zona"),
    ("¿Cumbres o Valle Oriente, cuál es más exclusivo?", "comparacion_zona"),
    ("¿Cuánto cuesta rentar en San Pedro Garza García vs Monterrey Centro?", "comparacion_zona"),
    ("¿Qué zona ha subido más de precio en Monterrey en los últimos meses?", "comparacion_zona"),
    ("¿Por qué Valle Oriente es más caro que otras zonas?", "comparacion_zona"),
    ("¿Hay diferencia de precio entre Cumbres y Cumbres Elite?", "comparacion_zona"),
    ("¿Qué municipio tiene las rentas más altas en el área metropolitana?", "comparacion_zona"),
    ("¿Es caro rentar cerca del Tecnológico de Monterrey?", "comparacion_zona"),
    ("¿Cuál zona tiene mejor relación precio-calidad para rentar?", "comparacion_zona"),
    ("¿Cuánto cuesta un depa de lujo en Valle Oriente vs uno estándar en Roma?", "comparacion_zona"),
    ("¿Por qué es tan cara la renta en Santa Bárbara?", "comparacion_zona"),
    ("¿Rentar en Guadalupe es más barato que en Monterrey?", "comparacion_zona"),
    ("¿Cuál zona tiene más oferta de departamentos amueblados?", "comparacion_zona"),
    ("¿Es buena inversión comprar para rentar en Obispado?", "comparacion_zona"),
    ("¿Cómo afecta estar cerca del Estadio BBVA al precio de renta?", "comparacion_zona"),
    ("¿El precio de renta en Monterrey es alto comparado con Guadalajara?", "comparacion_zona"),
    ("¿Cuáles colonias están subiendo de precio en Monterrey?", "comparacion_zona"),
    ("¿Vale más un depa en piso alto en Valle Oriente?", "comparacion_zona"),
    ("¿Cuánto cuesta rentar una casa vs departamento en Cumbres?", "comparacion_zona"),
]
for pregunta, tipo in comparaciones:
    benchmark_manual.append({
        "id": f"manual_{idx:04d}", "tipo": tipo,
        "pregunta": pregunta, "precio_real": None,
        "colonia": None, "split": "manual"
    })
    idx += 1

# Efecto de amenidades (25)
amenidades_preguntas = [
    ("¿Cuánto sube el precio de renta si el departamento tiene alberca?", "efecto_amenidad"),
    ("¿Vale la pena pagar más por un depa amueblado en Monterrey?", "efecto_amenidad"),
    ("¿Cuánto más caro es un departamento de lujo vs uno estándar?", "efecto_amenidad"),
    ("¿El gimnasio en el edificio sube mucho el precio de renta?", "efecto_amenidad"),
    ("¿Cuánto incrementa el precio tener estacionamiento incluido?", "efecto_amenidad"),
    ("¿Un departamento nuevo es mucho más caro que uno antiguo?", "efecto_amenidad"),
    ("¿Cuánto más cuesta un depa con rooftop en Monterrey?", "efecto_amenidad"),
    ("¿El elevador en el edificio afecta el precio de renta?", "efecto_amenidad"),
    ("¿Cuánto sube el precio si acepta mascotas?", "efecto_amenidad"),
    ("¿Un departamento con terraza cuánto más cuesta?", "efecto_amenidad"),
    ("¿Cuánto sube tener seguridad 24 horas?", "efecto_amenidad"),
    ("¿Vale más un departamento amueblado o sin amueblar en Valle Oriente?", "efecto_amenidad"),
    ("¿Cuántas amenidades necesita un depa para ser considerado de lujo?", "efecto_amenidad"),
    ("¿El cuarto de servicio sube el precio de renta?", "efecto_amenidad"),
    ("¿Cuánto más caro es un penthouse vs un departamento normal?", "efecto_amenidad"),
    ("¿La alberca es la amenidad que más sube el precio en Monterrey?", "efecto_amenidad"),
    ("¿Cuánto afecta el número de estacionamientos al precio?", "efecto_amenidad"),
    ("¿Un depa con jardín privado cuánto más vale?", "efecto_amenidad"),
    ("¿Cuánto sube el precio si hay área de coworking en el edificio?", "efecto_amenidad"),
    ("¿El lobby de lujo en el edificio afecta el precio de renta?", "efecto_amenidad"),
    ("¿Cuánto más caro es un depa con cisterna propia?", "efecto_amenidad"),
    ("¿El calentador solar sube el precio de renta?", "efecto_amenidad"),
    ("¿Cuánto más vale un depa en construcción nueva vs 10 años de antigüedad?", "efecto_amenidad"),
    ("¿Las vistas panorámicas afectan el precio en Valle Oriente?", "efecto_amenidad"),
    ("¿Cuánto más cuesta un depa con sala de juntas en el edificio?", "efecto_amenidad"),
]
for pregunta, tipo in amenidades_preguntas:
    benchmark_manual.append({
        "id": f"manual_{idx:04d}", "tipo": tipo,
        "pregunta": pregunta, "precio_real": None,
        "colonia": None, "split": "manual"
    })
    idx += 1

# Robustez — precios absurdos (10)
absurdos = [
    ("Me ofrecen un departamento de 3 recámaras en Valle Oriente por $3,000/mes. ¿Es normal?", "robustez"),
    ("¿Es posible rentar un depa de lujo en Santa Bárbara por $5,000/mes?", "robustez"),
    ("Me dicen que un depa en Contry cuesta $200,000/mes de renta. ¿Es real?", "robustez"),
    ("¿Puede costar $500/mes rentar en San Pedro Garza García?", "robustez"),
    ("Un depa de 150 m² en Valle Oriente por $8,000/mes, ¿es una ganga o algo raro?", "robustez"),
    ("¿Es normal pagar $100,000/mes por un departamento en Monterrey?", "robustez"),
    ("Me ofrecen una casa en Cumbres Elite por $2,000/mes. ¿Qué tan sospechoso es?", "robustez"),
    ("¿Cuánto debería desconfiar de un depa en Valle Oriente a $6,000/mes?", "robustez"),
    ("Un local comercial en Obispado por $500/mes, ¿es posible?", "robustez"),
    ("¿Es real una renta de $1,000/mes para una oficina en el Tecnológico?", "robustez"),
]
for pregunta, tipo in absurdos:
    benchmark_manual.append({
        "id": f"manual_{idx:04d}", "tipo": tipo,
        "pregunta": pregunta, "precio_real": None,
        "colonia": None, "split": "manual"
    })
    idx += 1

# Tendencia de mercado (11)
tendencias = [
    ("¿Cómo ha cambiado el precio de renta en Monterrey en 2026?", "tendencia"),
    ("¿El Mundial 2026 va a subir los precios de renta en Monterrey?", "tendencia"),
    ("¿Cuáles zonas de Monterrey van a subir de precio en los próximos meses?", "tendencia"),
    ("¿Es buen momento para rentar o comprar en Monterrey?", "tendencia"),
    ("¿Por qué han subido las rentas en Monterrey?", "tendencia"),
    ("¿Qué va a pasar con los precios después del Mundial 2026?", "tendencia"),
    ("¿Monterrey es más caro que Guadalajara para rentar?", "tendencia"),
    ("¿Cuánto ha subido el precio de renta en Valle Oriente en el último año?", "tendencia"),
    ("¿Hay sobreoferta de departamentos en Monterrey ahorita?", "tendencia"),
    ("¿Cuáles son los mejores meses para buscar depa en Monterrey?", "tendencia"),
    ("¿El tipo de cambio afecta los precios de renta en zonas de lujo?", "tendencia"),
]
for pregunta, tipo in tendencias:
    benchmark_manual.append({
        "id": f"manual_{idx:04d}", "tipo": tipo,
        "pregunta": pregunta, "precio_real": None,
        "colonia": None, "split": "manual"
    })
    idx += 1

# Zona no vista (20)
zonas_raras = [
    "Anáhuac", "Mitras Norte", "Fierro", "La Independencia",
    "Buenos Aires", "Garza Nieto", "Talleres", "La Purísima",
    "Industrial", "Las Puentes", "Villa de San Nicolás",
    "Cumbres 5to Sector", "Las Torres", "Jardines de San Nicolás",
    "El Mirador", "Colinas de San Jerónimo", "Las Cumbres",
    "Vistas del Valle", "Residencial Anáhuac", "Pinos"
]
for zona in zonas_raras:
    pregunta = f"¿Cuánto cuesta rentar un departamento de 2 recámaras en {zona}, Monterrey?"
    benchmark_manual.append({
        "id": f"manual_{idx:04d}", "tipo": "zona_no_vista",
        "pregunta": pregunta, "precio_real": None,
        "colonia": zona, "split": "manual"
    })
    idx += 1

# Precio por m² (9 para llegar a 100)
pm2_preguntas = [
    ("¿Cuánto cuesta el m² de renta en Valle Oriente?", "precio_m2"),
    ("¿Cuál es el precio por m² en Contry?", "precio_m2"),
    ("¿Cuánto es el precio por m² en San Pedro Garza García?", "precio_m2"),
    ("¿Qué precio por m² es razonable en Obispado?", "precio_m2"),
    ("¿El precio por m² en oficinas es mayor que en departamentos?", "precio_m2"),
    ("¿Cuánto vale el m² en una bodega industrial en Monterrey?", "precio_m2"),
    ("¿El precio por m² en locales comerciales es mayor que en departamentos?", "precio_m2"),
    ("¿Cuánto cuesta el m² de renta en una zona media de Monterrey?", "precio_m2"),
    ("¿Cuál es el precio por m² más alto registrado en Monterrey?", "precio_m2"),
]
for pregunta, tipo in pm2_preguntas:
    benchmark_manual.append({
        "id": f"manual_{idx:04d}", "tipo": tipo,
        "pregunta": pregunta, "precio_real": None,
        "colonia": None, "split": "manual"
    })
    idx += 1

print(f"  Preguntas manuales generadas: {len(benchmark_manual)}")

# ── Combinar todo ──────────────────────────────────────────────────────────
benchmark_full = benchmark_test + benchmark_train + benchmark_manual
random.shuffle(benchmark_full)

# Re-asignar IDs secuenciales
for i, item in enumerate(benchmark_full):
    item["id_seq"] = i + 1

# ── Guardar ────────────────────────────────────────────────────────────────
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Guardado: {path}  ({len(data)} preguntas)")

save_jsonl(benchmark_test,   DATA / "benchmark_test.jsonl")
save_jsonl(benchmark_train,  DATA / "benchmark_train.jsonl")
save_jsonl(benchmark_manual, DATA / "benchmark_manual.jsonl")
save_jsonl(benchmark_full,   DATA / "benchmark_full.jsonl")

print(f"\n{'='*55}")
print(f"  BENCHMARK TOTAL: {len(benchmark_full)} preguntas")
print(f"{'='*55}")
print(f"  Precio directo (test,  no vistas): {len(benchmark_test):>4}")
print(f"  Precio directo (train, vistas)   : {len(benchmark_train):>4}")
print(f"  Comparación de zonas             : {len([x for x in benchmark_manual if x['tipo']=='comparacion_zona']):>4}")
print(f"  Efecto de amenidades             : {len([x for x in benchmark_manual if x['tipo']=='efecto_amenidad']):>4}")
print(f"  Robustez (precios absurdos)      : {len([x for x in benchmark_manual if x['tipo']=='robustez']):>4}")
print(f"  Tendencia de mercado             : {len([x for x in benchmark_manual if x['tipo']=='tendencia']):>4}")
print(f"  Zona no vista                    : {len([x for x in benchmark_manual if x['tipo']=='zona_no_vista']):>4}")
print(f"  Precio por m²                    : {len([x for x in benchmark_manual if x['tipo']=='precio_m2']):>4}")
print(f"{'='*55}")
print(f"\n✓ Benchmark listo en {DATA}/")
