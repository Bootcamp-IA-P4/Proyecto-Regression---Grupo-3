# Entrenamiento del Modelo - Random Forest v8

## Visión General

Este repositorio contiene el pipeline de entrenamiento para un Random Forest Regressor (v8) utilizado para predecir precios de propiedades en Airbnb. La implementación incluye dos enfoques principales de entrenamiento con características avanzadas como checkpointing, validación exhaustiva y gestión de metadatos.

## Características Principales

- **Sistema de Checkpoints**: Implementa un mecanismo robusto para guardar el progreso del entrenamiento
- **Búsqueda en Cuadrícula**: Ajuste de hiperparámetros con GridSearchCV de scikit-learn
- **Transformación Logarítmica**: La variable objetivo (precio) se transforma logarítmicamente para mejor rendimiento
- **Importancia de Features**: Análisis de contribuciones de variables a las predicciones
- **Suite de Validación**: Múltiples técnicas de validación para asegurar fiabilidad del modelo

## Versiones de Entrenamiento

### Entrenamiento del Modelo 2

Implementación básica con:
- GridSearchCV para ajuste de hiperparámetros
- Guardado de checkpoints durante el entrenamiento
- Análisis de importancia de variables
- Persistencia del modelo con nombres de features

### Entrenamiento del Modelo 3

Versión mejorada con:
- Gestor de checkpoints mejorado con seguimiento de metadatos
- Rastreo completo de la cuadrícula de parámetros
- Guardado de metadatos del modelo (JSON)
- Suite de validación exhaustiva que incluye:
  - Validación cruzada estricta
  - Comparación con línea base
  - Análisis de residuales
  - Importancia por permutación
  - Validación en subconjuntos aleatorios

## Resultados

### Métricas de Rendimiento

| Métrica                | Entrenamiento | Prueba    | Validación Cruzada (Media ± Desv) |
|-----------------------|--------------|-----------|-----------------------------------|
| R² Score              | 0.9713       | 0.8482    | 0.8464 ± 0.0200                   |
| MAE (EUR)            | -            | 19.83     | 20.12 ± 0.92                      |

### Hallazgos Clave

1. El modelo explica **84.82%** de la varianza en precios (R² = 0.8482)
2. Error de predicción promedio de **19.83 EUR**
3. Mejora significativa sobre la línea base (reducción de MAE en 31.36 EUR)
4. Rendimiento consistente en todos los métodos de validación

## Uso

### Requisitos

- Python 3.8+
- scikit-learn
- pandas
- numpy

### Ejecución del Entrenamiento

```bash
# Ejecutar Entrenamiento del Modelo 2
python model_training_2.py

# Ejecutar Entrenamiento del Modelo 3 con validación
python model_training_3.py
```

## Detalles de Implementación

### Gestor de Checkpoints

La clase personalizada `CheckpointManager` proporciona:

- Guardado automático del progreso del entrenamiento
- Recuperación de sesiones de entrenamiento interrumpidas
- Seguimiento de metadatos (parámetros, métricas, nombres de variables)
- Limpieza automática de checkpoints antiguos

```python
class CheckpointManager:
    def save_checkpoint(self, model, X, y, params, metrics=None, stage="training"):
        """Guarda el estado actual del entrenamiento"""
        # Detalles de implementación...
    
    def load_latest_checkpoint(self):
        """Carga el checkpoint más reciente"""
        # Detalles de implementación...
```

### Suite de Validación

La validación exhaustiva incluye:

1. **Validación Cruzada Estricta**: 5-fold CV con métricas R² y MAE
2. **Comparación con Línea Base**: Contra un predictor simple de mediana
3. **Análisis de Residuales**: Inspección visual de errores de predicción
4. **Importancia por Permutación**: Mide importancia de variables mediante mezcla aleatoria
5. **Validación en Subconjuntos**: Verificación en subconjuntos aleatorios de datos

## Estructura de Directorios

```
models/
  └── minimal_rf_model.pkl          # Modelo entrenado final
  └── model_metadata.json           # Metadatos del modelo

checkpoints/
  └── checkpoint_*.pkl              # Checkpoints de entrenamiento
  └── validation_*.pkl              # Resultados de validación

data/
  └── processed/
      └── df_minimal.csv            # Dataset procesado
```

## Recomendaciones para Producción

1. **Monitorizar Rendimiento**: Implementar detección de desviación en variables y predicciones
2. **Reentrenar Periódicamente**: Programar actualizaciones regulares del modelo con nuevos datos
3. **Manejar Outliers**: Añadir lógica de negocio para predicciones extremas
4. **Registrar Predicciones**: Trackear predicciones del modelo para análisis y depuración

## Conclusión

El modelo Random Forest v8 demuestra un excelente rendimiento predictivo con un R² de 0.8482 y MAE de 19.83 EUR. La suite de validación exhaustiva confirma la fiabilidad del modelo y su preparación para despliegue en producción.

Para preguntas o incidencias, por favor abre un issue en este repositorio.
