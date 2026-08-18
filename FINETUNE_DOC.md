# Fine-tuning de Qwen2.5 para Bienes Raíces en Monterrey

## ¿Qué estamos haciendo y por qué?

Tenemos 1,102 propiedades en renta de Monterrey con precios reales.
La idea es enseñarle a un modelo de lenguaje (Qwen2.5-7B) a razonar
sobre el mercado inmobiliario local — qué colonias son caras, qué
amenidades suben el precio, cómo se comporta cada zona.

La diferencia con ChatGPT o cualquier otro LLM genérico es que ese
modelo no sabe nada específico de Monterrey. El fine-tuning lo hace
experto en tu mercado con tus propios datos.

---

## ¿Qué es LoRA?

Qwen2.5-7B tiene 7,600 millones de parámetros. Reentrenarlos todos
requeriría semanas y decenas de GB de GPU.

**LoRA (Low-Rank Adaptation)** es un truco matemático:
en lugar de modificar todos los parámetros, agrega matrices pequeñas
("adaptadores") encima de las capas del modelo y solo entrena esas.

```
Modelo base (congelado, 7.6B params)
       +
Adaptador LoRA (5.7M params — 0.076% del total)
       ↓
Modelo que sabe de bienes raíces en Monterrey
```

En nuestro caso entrenamos **5.7 millones de parámetros** de los
7,600 millones totales. Por eso cabe en tu Mac y tarda 1-2 horas
en lugar de semanas.

---

## ¿Qué modelo usamos como base?

```
mlx-community/Qwen2.5-7B-Instruct-4bit
```

- **Qwen2.5-7B**: modelo de Alibaba, 7 mil millones de parámetros
- **Instruct**: ya fue entrenado para seguir instrucciones (chat)
- **4bit**: cuantizado a 4 bits — ocupa ~4 GB en vez de ~14 GB
- **mlx-community**: versión optimizada para Apple Silicon (Metal GPU)

---

## ¿Por qué mlx-lm y no HuggingFace?

Tu Mac tiene chip Apple Silicon (M-series). El framework **MLX**
de Apple usa el GPU Metal nativo, que es 3-4x más rápido que
PyTorch MPS para este tipo de tarea.

HuggingFace + bitsandbytes (la alternativa común) no soporta
cuantización de 4 bits en Mac. MLX sí.

---

## Los datos de entrenamiento

### Fuente
`propiedades_enriquecido.xlsx` — 1,102 propiedades de Websights
con precio real, m², recámaras, baños, zona, amenidades.

### Formato
Cada propiedad se convierte en un par pregunta/respuesta:

```
[SISTEMA]
Eres un experto en el mercado inmobiliario de Monterrey...

[USUARIO]
Tengo una propiedad con estas características:
- Colonia: Tecnológico, Monterrey
- Recámaras: 2
- Baños: 1
- Superficie: 80 m²
- Amueblado: Sí
- Lujo: Sí
¿Cuánto debería costar de renta mensual?

[ASISTENTE]
Basándome en el mercado actual de Monterrey, el precio estimado
es $18,800 MXN/mes (rango: $15,980–$21,620)...
[razonamiento basado en zona, amenidades, m², etc.]
```

### Split de datos

| Conjunto | Ejemplos | Uso |
|---|---|---|
| `train.jsonl` | 872 | Entrenamiento (80%) |
| `valid.jsonl` | 109 | Validación durante entrenamiento (10%) |
| `test.jsonl`  | 109 | Evaluación final (10%) |

---

## Configuración del entrenamiento

```yaml
# lora_config.yaml
model:          Qwen2.5-7B-Instruct-4bit
iters:          600        # pasos de entrenamiento (~2 épocas)
batch_size:     4          # propiedades por paso
learning_rate:  0.00002    # qué tan rápido aprende
num_layers:     8          # cuántas capas del modelo se adaptan
max_seq_length: 1024       # longitud máxima de cada ejemplo
mask_prompt:    true       # solo aprende de las RESPUESTAS, no las preguntas
```

**¿Por qué 600 iters?**
Con 872 ejemplos y batch_size=4 → cada época son ~218 pasos.
600 pasos ≈ 2.7 épocas: suficiente para aprender sin memorizar.

---

## Estructura de archivos

```
analisis_propiedades/
├── prepare_finetune_data.py   # Convierte xlsx → train/valid/test.jsonl
├── finetune_qwen.py           # Lanza el fine-tuning + inferencia
├── lora_config.yaml           # Hiperparámetros de entrenamiento
├── data/
│   ├── train.jsonl            # 872 pares pregunta/respuesta
│   ├── valid.jsonl            # 109 para validar durante entrenamiento
│   └── test.jsonl             # 109 para evaluación final
└── adapters/
    └── adapters.safetensors   # El adaptador LoRA entrenado
```

---

## Cómo correrlo

```bash
# 1. Entrenamiento completo (~1-2 horas)
cd ~/Desktop/analisis_propiedades
python3 finetune_qwen.py

# 2. Probar el modelo entrenado (chat interactivo)
python3 finetune_qwen.py --inference

# 3. Prueba rápida de 10 pasos (para verificar que todo funciona)
python3 finetune_qwen.py --test
```

---

## ¿Qué pasa durante el entrenamiento?

Verás algo así en pantalla:

```
Iter  10: Train loss 1.086  Val loss 2.265
Iter  50: Train loss 0.821  Val loss 1.840
Iter 100: Train loss 0.634  Val loss 1.423
...
Iter 600: Train loss 0.210  Val loss 0.380
```

- **Train loss**: qué tan mal predice en los datos de entrenamiento
- **Val loss**: qué tan mal predice en datos que NO vio

Si ambos bajan → el modelo está aprendiendo bien.
Si train baja pero val sube → overfitting (memorizó, no aprendió).

---

## Lo que ya construimos antes del fine-tuning

| Script | Qué hace | Resultado |
|---|---|---|
| `enrich_lamudi_openai.py` | GPT-4o-mini extrae amenidades de descripciones | `leads_lamudi_enriched_openai.csv` |
| `train_xgboost.py` | Modelo de precio con XGBoost | R²=0.70, MAE=±$3,610/mes |
| `pipeline_semanal.py` | Scraping + enrich + HTML + LaTeX automático | Actualización semanal |

---

## Siguiente paso: RAG

El fine-tuning enseña al modelo el mercado de Monterrey.
El siguiente paso es **RAG (Retrieval-Augmented Generation)**:
conectar el modelo a tus datos en tiempo real para que pueda
responder preguntas con datos actualizados y citar la fuente.

```
"¿Cuánto cuesta rentar en Valle Oriente?"
         ↓
    Busca en tu base de datos de propiedades
         ↓
    "Basado en 47 propiedades activas en Valle Oriente,
     el precio mediano es $28,500/mes. Fuente: Lamudi, mayo 2026."
```

Eso es lo que lo convierte en el "Perplexity de bienes raíces".
