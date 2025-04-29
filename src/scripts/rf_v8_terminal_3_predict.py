# Predictor de Precios - Random Forest v8 Model Training 3 (Consola)

# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def load_model():
    """Carga el modelo entrenado y los nombres de las features"""
    model_path = Path("models/minimal_rf_model.pkl")
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        if 'model' not in model_data or 'feature_names' not in model_data:
            raise ValueError("El archivo del modelo no contiene la estructura esperada")
        
        return model_data['model'], model_data['feature_names']
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")
        return None, None

def get_input_value(feature_name):
    """Obtiene el valor de una feature desde la consola"""
    while True:
        try:
            value = input(f"Ingrese valor para '{feature_name}': ")
            # Intentar convertir a float primero, luego a int si es posible
            num_value = float(value)
            if num_value.is_integer():
                return int(num_value)
            return num_value
        except ValueError:
            # Si falla la conversión a número, mantener como string
            return value

def make_prediction(model, features, feature_names):
    """Realiza la predicción con el modelo cargado"""
    try:
        # Crear DataFrame manteniendo el orden correcto de las columnas
        input_data = pd.DataFrame([features], columns=feature_names)
        
        # Predecir (recordar que el modelo fue entrenado con log(price))
        log_prediction = model.predict(input_data)[0]
        
        # Convertir de vuelta a precio normal
        return np.expm1(log_prediction)
    except Exception as e:
        print(f"❌ Error en la predicción: {e}")
        return None

def main():
    print("\n🏠 Predictor de Precios de Propiedades - Modelo Random Forest v8\n")
    
    # Cargar modelo y nombres de features
    model, feature_names = load_model()
    if not model or not feature_names:
        return
    
    print(f"\n🔍 El modelo utiliza {len(feature_names)} características:")
    print(", ".join(feature_names))
    
    # Recoger datos de entrada
    print("\nPor favor ingrese los valores para cada característica:\n")
    features = {}
    for feature in feature_names:
        features[feature] = get_input_value(feature)
    
    # Realizar predicción
    prediction = make_prediction(model, features, feature_names)
    
    if prediction is not None:
        print(f"\n💶 Precio estimado: €{prediction:,.2f}")
    
    print("\n✅ Predicción completada\n")

if __name__ == "__main__":
    main()



## Características del Script

""" 1. **Carga del Modelo**:
   - Busca automáticamente el modelo en `../models/minimal_rf_model.pkl`
   - Verifica que contenga tanto el modelo como los nombres de las features

2. **Entrada de Datos**:
   - Solicita valores para cada feature una por una
   - Intenta convertir automáticamente a números (float o int)
   - Mantiene como string si la conversión falla

3. **Predicción**:
   - Crea un DataFrame con el orden correcto de columnas
   - Aplica la transformación inversa (expm1) para obtener el precio real
   - Muestra el resultado formateado con separadores de miles

4. **Manejo de Errores**:
   - Detecta problemas al cargar el modelo
   - Captura errores durante la predicción
   - Proporciona mensajes claros al usuario

## Cómo Usarlo

1. Guarda el código como `rf_v8_terminal_predict.py`
2. Ejecuta desde la terminal: `python rf_v8_terminal_predict.py`
3. Sigue las instrucciones para ingresar los valores de cada feature
4. Recibe la predicción de precio formateada

## Requisitos

- El script debe estar en la misma estructura de directorios que tu código de entrenamiento
- El archivo `minimal_rf_model.pkl` debe existir en `../models/`
- Python 3.x con las dependencias instaladas (pandas, numpy, scikit-learn)

Este script está diseñado específicamente para funcionar con el modelo entrenado 
por el código que proporcionaste, respetando la estructura de datos y las transformaciones (log1p/expm1) utilizadas durante el entrenamiento. """