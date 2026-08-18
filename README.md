# A Hybrid LLM–XGBoost System for Low-Resource Spanish Real Estate Valuation in Emerging Markets

Código y datos de soporte para el artículo enviado a **Knowledge-Based Systems / ESWA**.  
Mercado: propiedades en renta, Área Metropolitana de Monterrey, México.

---

## Qué hace este proyecto

Combina dos componentes para valuar propiedades en renta a partir de descripciones en español de baja calidad:

1. **XGBoost con Box-Cox** — modelo estructurado (m², recámaras, baños, zona, amenidades). R²=0.773, MAE=$3,459 MXN/mes en test set (n=218, bootstrap 10,000 resamples).
2. **Qwen2.5-7B-Instruct fine-tuned con LoRA** — extrae precio de texto libre en español. Fine-tuning vía MLX (rank=16, scale=20.0, 8 capas, 600 iters, lr=2e-5). Precisión ±15%: 67.0% → 73.4% tras fine-tuning.
3. **Sistema híbrido** — el LLM filtra propiedades fuera de rango; XGBoost calcula el precio final. Precisión ±15%: 75.2%, ±20%: 83.5% (n=109 preguntas benchmark).

El benchmark **MonterreyRE-QA** (909 preguntas generadas con CAI/Constitutional AI) evalúa la extracción de precio en español.

---

## Estructura del repositorio

```
analisis_propiedades/
│
├── README.md
│
├── # ── DATOS ────────────────────────────────────────────────
├── propiedades_enriquecido.xlsx        # Dataset principal (1,090 propiedades post-limpieza)
│                                       # ⚠️  NO incluir en repo público (datos scrapeados)
│
├── # ── PIPELINE PRINCIPAL ───────────────────────────────────
├── prepare_finetune_data.py            # Genera train/val/test splits para fine-tuning
├── train_xgboost.py                    # Entrena XGBoost con Box-Cox; guarda xgb_model.json
├── finetune_qwen.py                    # Fine-tuning LoRA sobre Qwen2.5-7B-Instruct-4bit (MLX)
├── generate_benchmark.py               # Genera MonterreyRE-QA (909 preguntas) con CAI
├── benchmark_hybrid_boxcox.py          # Evalúa sistema híbrido (LLM + XGBoost)
├── benchmark_multimodel.py             # Evalúa Llama 3.1-8B, Mistral-7B, Salamandra-7B
├── benchmark_commercial.py             # Evalúa GPT-4o-mini y Claude Haiku (APIs)
├── compare_models.py                   # Compara 7 modelos de regresión (Ridge→MLP)
├── compute_xgb_bootstrap.py            # Bootstrap CIs (10,000 resamples) para MAE y R²
├── statistical_tests.py                # McNemar y Wilcoxon entre configuraciones
├── generate_figures.py                 # Genera fig2–fig5 en PDF para el paper
│
├── # ── SCRIPTS AUXILIARES ───────────────────────────────────
├── ablation_base_hybrid.py             # Tabla de ablación (Base → Hybrid)
├── constitutional_ai.py                # Genera benchmark con principios CAI
├── benchmark_cai.py                    # Evalúa calidad CAI vs generación directa
├── check_leakage.py                    # Verifica que no hay data leakage en encodings
├── baseline_fewshot.py                 # Baseline few-shot sin XGBoost
├── baseline_rag.py                     # Baseline RAG
├── baseline_fewshot_rag_xgb.py         # Baseline few-shot + RAG + XGBoost
├── hybrid_system.py                    # Lógica del sistema híbrido (producción)
│
├── # ── RESULTADOS ───────────────────────────────────────────
└── results/
    ├── xgb_model.json                  # Modelo XGBoost serializado
    ├── xgb_metrics.json                # R², MAE en test set
    ├── xgb_bootstrap_ci.json           # Bootstrap CI (MAE y R²)
    ├── model_comparison.json           # Tabla comparativa de 7 modelos
    ├── ablation_metrics.json           # Tabla de ablación completa
    ├── statistical_tests.json          # p-values McNemar y Wilcoxon
    ├── benchmark_multimodel_summary.json  # Resultados Llama/Mistral/Salamandra
    ├── benchmark_commercial_summary.json  # Resultados GPT-4o-mini / Claude Haiku
    ├── benchmark_hybrid_boxcox_test.jsonl # Predicciones sistema híbrido (n=109)
    ├── benchmark_finetuned_test.jsonl  # Predicciones modelo fine-tuned
    ├── benchmark_base_test.jsonl       # Predicciones modelo base
    ├── benchmark_fewshot_xgb_test.jsonl # Predicciones few-shot + XGBoost
    └── figures/
        ├── fig2_training_curve.pdf     # Curva de training LoRA (Fig. 2 en paper)
        ├── fig3_model_comparison.pdf   # Comparación 7 modelos de regresión (Fig. 3)
        ├── fig4_bias_analysis.pdf      # Análisis de sesgo por rango de precio (Fig. 4)
        └── fig5_fewshot_xgb_scatter.pdf # XGBoost solo vs Few-shot+XGBoost (Fig. 5)
```

---

## Qué subir a GitHub

| Incluir | NO incluir |
|---|---|
| Todos los `.py` scripts | `propiedades_enriquecido.xlsx` (datos privados) |
| `results/*.json` (métricas) | Claves de API (OpenAI, Anthropic) |
| `results/*.jsonl` (predicciones) | Pesos del adapter LoRA (van en HuggingFace/Zenodo) |
| `figures/fig2–fig5.pdf` | Modelos base descargados (Qwen, Llama, etc.) |
| `README.md` | `.env`, archivos de configuración personal |

Los pesos LoRA y el dataset irán en un repositorio de datos separado (Zenodo o HuggingFace Hub) enlazado desde el paper.

---

## Reproducir los resultados

### Dependencias

```bash
pip install pandas numpy scipy scikit-learn xgboost lightgbm catboost \
            matplotlib openpyxl tqdm
# Para fine-tuning (Mac con Apple Silicon):
pip install mlx mlx-lm
```

### Orden de ejecución

```bash
# 1. Preparar datos para fine-tuning
python3 prepare_finetune_data.py

# 2. Entrenar XGBoost
python3 train_xgboost.py

# 3. Fine-tuning LoRA (~90 min en M2/M3)
python3 finetune_qwen.py

# 4. Generar benchmark MonterreyRE-QA
python3 generate_benchmark.py

# 5. Evaluar sistema híbrido
python3 benchmark_hybrid_boxcox.py

# 6. Evaluar modelos externos (requiere APIs)
python3 benchmark_multimodel.py
python3 benchmark_commercial.py

# 7. Comparar modelos de regresión
python3 compare_models.py

# 8. Bootstrap CIs
python3 compute_xgb_bootstrap.py

# 9. Tests estadísticos
python3 statistical_tests.py

# 10. Generar figuras
python3 generate_figures.py
```

> Los pasos 6 requieren variables de entorno `OPENAI_API_KEY` y `ANTHROPIC_API_KEY`.  
> Los pasos 3–6 requieren los modelos base descargados con `mlx_lm.convert` o `ollama pull`.

---

## Resultados principales

| Configuración | Acc ±15% | Acc ±20% |
|---|---|---|
| Base (Qwen2.5-7B, zero-shot) | 40.4% | 52.3% |
| Fine-tuned | 67.0% | 73.4% |
| Few-shot + XGBoost | 71.6% | 78.0% |
| **Híbrido (Fine-tuned + XGBoost)** | **75.2%** | **83.5%** |

XGBoost solo: R²=0.773, MAE=$3,459 MXN/mes [IC95%: $3,059–$3,889].

---

## Datos

El dataset contiene 1,090 anuncios de renta del AMM (Área Metropolitana de Monterrey) scrapeados en 2025. Variables principales: precio, m², recámaras, baños, estacionamientos, coordenadas, amenidades, descripción en texto libre.

Por restricciones de términos de servicio, el dataset completo se distribuye bajo solicitud. Una muestra anonimizada de 100 registros está disponible en `data/sample_100.csv`.
