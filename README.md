# 🏠 Inside Airbnb - Madrid: Análisis Exploratorio (EDA v2) & Modelo de Predicción de Precios 🔍📊

## 📌 Visión General del Proyecto
Este proyecto analiza datos de listados de Airbnb en Madrid para construir un modelo de predicción de precios. A través de ingeniería de características iterativa y optimización del modelo, redujimos el conjunto de datos de 105 columnas a 14 características clave mientras mejorábamos el rendimiento.

## 🔑 Pasos Clave

### 1. 🧹 Preparación de Datos
- 📂 Combinamos 4 conjuntos de datos trimestrales (Marzo 2024 - Marzo 2025) en un único DataFrame
- 🔍 Manejo de duplicados usando `id` y `scrape_id` como claves compuestas
- 💰 Limpieza de datos de precios (eliminamos símbolos $, convertimos a float)
- 🛠️ Tratamiento de valores faltantes en columnas clave (dormitorios, baños, camas)

### 2. ⚙️ Ingeniería de Características
**📉 Características Iniciales (105) → Reducidas a 70**
- ➕ Creación de términos de interacción (bed_bath_ratio, acc_to_beds)
- 📅 Características temporales (host_experience_years)
- 🛏️ Agregación de comodidades en un amenity_score
- 🏘️ Codificación target para datos de vecindarios

**📊 Reducción de Características (70 → 26)**
- 🗑️ Eliminación de características de baja importancia (< 0.01)
- 🔄 Eliminación de características altamente correlacionadas (r > 0.85)
- 🏷️ Consolidación de tipos de propiedad raros en categoría "Otros"

**🎯 Conjunto Final (26 → 14)**
- 🔝 Nos quedamos solo con las características más predictivas
- 🏡 Enfoque en características de propiedad, ubicación y factores del anfitrión

### 3. 🤖 Desarrollo del Modelo
- 🌳 Prueba de algoritmos (Random Forest y XGBoost)
- 📈 Transformación logarítmica del precio para mejor normalidad
- 🎛️ Optimización de hiperparámetros con GridSearchCV
- 📏 Evaluación con métricas R² y MAE

## 🏆 Rendimiento del Modelo

### 🥇 Mejor Modelo: Random Forest v5
- **R² Test**: 0.7726
- **MAE Test**: 20.11 €
- **🔝 Características Clave**:
  1. `room_type_Entire_home_apt` (20% importancia)
  2. `accommodates` (15% importancia)
  3. `neighbourhood_encoded` (8% importancia)
  4. `bedrooms` (7% importancia)
  5. `review_scores_rating` (4% importancia)

## 📂 Estructura del Repositorio
```
/data
  /raw        # Archivos CSV originales
  /processed  # Datasets limpios y transformados
/models      # Modelos guardados
/notebooks   # Cuadernos Jupyter con EDA y modelado
```

## 🛠️ Cómo Reproducir
1. Instalar requisitos: `pip install -r requirements.txt`
2. Ejecutar cuadernos en orden:
   - `1_data_cleaning.ipynb`
   - `2_feature_engineering.ipynb`
   - `3_model_training.ipynb`

## 💡 Hallazgos Clave
1. 🏠 Los alojamientos completos tienen precios 23% más altos en promedio
2. 🛌 Cada dormitorio adicional aumenta el precio ~15% (rendimientos decrecientes después de 3)
3. 🗺️ El vecindario es el determinante geográfico más fuerte del precio
4. 🏅 El estatus de Superhost añade ~5% al precio predicho

## 🚀 Mejoras Futuras
- 🌦️ Incorporar variaciones estacionales de precios
- 🗺️ Añadir proximidad a atracciones turísticas como característica
- 🔄 Implementar aprendizaje online para adaptarse a cambios del mercado

## 👥 Equipo
- 👩‍💻 Maryna Nalyvayko
- 👨‍💻 Max Beltrán
- 👨‍💻 Jorge Luis Mateos
- 👨‍💻 Juan Domingo

Abril 2025 📅
