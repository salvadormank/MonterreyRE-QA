# Features y Dataset para Modelo Mejorado (XGBoost v2)

## Estado actual del modelo
- R² = 0.719 | MAE = 3,548 MXN/mes | n = 1,090 registros
- Límite teórico estimado: ~0.85 R²

---

## Features nuevas a agregar

### Prioridad ALTA (mayor impacto en R²)

| Feature | Tipo | Descripción | Fuente | Impacto esperado |
|---|---|---|---|---|
| `piso` | Discreto | Número de piso de la unidad | Listing / scraping | +0.05–0.08 R² |
| `pisos_totales` | Discreto | Total de pisos del edificio | Listing / scraping | +0.02–0.03 R² |
| `piso_relativo` | Continuo | `piso / pisos_totales` (feature derivada) | Calculada | incluida arriba |
| `nombre_edificio` | Categórico | Nombre del edificio o torre | Listing / scraping | +0.04–0.06 R² |
| `edificio_enc` | Continuo | Target encoding del nombre_edificio | Calculada | incluida arriba |

**Por qué el piso importa:** mismo edificio en Valle Oriente puede ir de $28,000 (piso 2) a $52,000 (piso 20). El modelo actual no distingue.

---

### Prioridad MEDIA

| Feature | Tipo | Descripción | Fuente |
|---|---|---|---|
| `antiguedad_anios` | Continuo | Año actual − año de construcción | Listing / permiso de construcción |
| `medio_bano` | Discreto | Número de medios baños | Listing |
| `bodega` | Binario | 1 = tiene bodega/storage | Listing |
| `cuarto_servicio` | Binario | 1 = cuarto de servicio | Listing |
| `vista_ciudad` | Binario | 1 = vista panorámica / ciudad | Listing / descripción GPT |
| `precio_m2_colonia` | Continuo | Mediana precio/m² de la colonia (calculada del dataset) | Calculada |
| `distancia_centro` | Continuo | Distancia en km al centro de Monterrey | Calculada con lat/lon |
| `distancia_vialidad` | Continuo | Distancia a vialidades principales (Morones Prieto, Lázaro Cárdenas) | Calculada con lat/lon |

---

### Prioridad BAJA (rendimientos decrecientes)

| Feature | Tipo | Descripción | Fuente |
|---|---|---|---|
| `estrato_nse` | Ordinal | Nivel socioeconómico del AGEB (INEGI) | INEGI ENIGH |
| `densidad_oferta` | Continuo | # propiedades en renta en radio 500m | Calculada del dataset |
| `dias_mercado_colonia` | Continuo | Mediana días en mercado por colonia | Calculada del dataset |
| `variacion_precio_trim` | Continuo | Cambio % en precio promedio de colonia vs trimestre anterior | Requiere datos históricos |

---

## Dataset a conseguir

### 1. Datos de piso y nombre de edificio (CRÍTICO)

**Qué buscar en el listing:**
- Campo `"piso"` o `"nivel"` en el JSON de Lamudi / Inmuebles24
- Campo `"nombre_edificio"` o `"desarrollo"` o `"torre"`
- En descripciones libres: regex `r'piso\s*(\d+)'`, `r'nivel\s*(\d+)'`, `r'(\d+)o\s*piso'`

**Acción:** Actualizar el scraper para capturar estos campos. Ya están en el HTML de Lamudi — solo no se están guardando.

---

### 2. Registros adicionales (colonias sub-representadas)

**Colonias prioritarias para scraping focalizado** (pocas muestras en dataset actual):

| Colonia | Municipio | Por qué importa |
|---|---|---|
| Cumbres (todas las etapas) | Monterrey | Precio muy variable por etapa |
| Contry | Monterrey | Mercado de casas grande |
| Casco Urbano | San Pedro | Nuevo desarrollo, precios altos |
| Vía Cordillera | San Pedro | Pocos registros, precios >$40k |
| La Fe | San Nicolás | Mercado popular sub-representado |
| Paseo de los Leones | Monterrey | Zona en crecimiento |

**Meta:** 200–300 registros adicionales en estas colonias → +0.02 R² estimado.

---

### 3. INEGI — Valor catastral / AGEB (opcional, largo plazo)

**Dataset:** INEGI Cartografía Geoestadística Urbana
- URL: `https://www.inegi.org.mx/temas/ageb_manzana/`
- Contiene: estrato NSE por AGEB, densidad poblacional, uso de suelo
- Join: por coordenadas lat/lon → AGEB

**Impacto esperado:** +0.01–0.02 R² (ya capturado parcialmente por `colonia_enc`)

---

## Features derivadas a calcular (costo = cero, solo código)

```python
# Piso relativo (0 = planta baja, 1 = último piso)
df["piso_relativo"] = df["piso"] / df["pisos_totales"]

# Precio por m² de la colonia (leakage-safe: calcular en OOF)
df["precio_m2_colonia"] = df.groupby("colonia")["price"].transform("median") / df["m2"]

# Distancia al centro de Monterrey (25.6866° N, -100.3161° W)
from geopy.distance import geodesic
CENTRO_MTY = (25.6866, -100.3161)
df["dist_centro_km"] = df.apply(
    lambda r: geodesic((r["lat"], r["lon"]), CENTRO_MTY).km
    if pd.notna(r["lat"]) else np.nan, axis=1
)

# Distancia a Lázaro Cárdenas (vialidad principal)
# Aproximar como línea recta: lat≈25.65, lon variable
df["dist_lazaro_km"] = abs(df["lat"] - 25.65) * 111  # grados → km aprox

# Target encoding del edificio (mismo esquema que colonia)
counts = df.groupby("nombre_edificio")["bc_price"].agg(["mean", "count"])
smooth_k = 5  # k menor porque hay muchos edificios con pocos registros
smoothed = (counts["mean"] * counts["count"] + global_mean * smooth_k) / (counts["count"] + smooth_k)
df["edificio_enc"] = df["nombre_edificio"].map(smoothed).fillna(global_mean)
```

---

## Impacto estimado total

| Escenario | R² estimado | MAE estimado |
|---|---|---|
| Actual (v1) | 0.719 | 3,548 MXN/mes |
| + piso + edificio | ~0.77–0.80 | ~2,900 MXN/mes |
| + piso + edificio + colonias nuevas | ~0.80–0.83 | ~2,600 MXN/mes |
| Límite teórico (ruido irreducible) | ~0.85 | ~2,200 MXN/mes |

---

## Checklist de implementación

- [ ] Actualizar scraper WebSights para capturar `piso`, `pisos_totales`, `nombre_edificio`
- [ ] Re-enriquecer con GPT-4o-mini: extraer piso y edificio de descripciones existentes
- [ ] Calcular features derivadas (dist_centro, piso_relativo, precio_m2_colonia)
- [ ] Scraping focalizado en 6 colonias sub-representadas
- [ ] Re-entrenar XGBoost v2 con OOF encoding
- [ ] Comparar R² y MAE contra modelo actual
