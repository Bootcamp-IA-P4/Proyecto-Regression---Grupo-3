# Código completo para `rf_v8_terminal_predict.py` que permite hacer 
# predicciones interactivas desde la terminal:

# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Configuración de paths
model_dir = Path("models/")

def load_model_and_features():
    """Carga el modelo entrenado y los metadatos"""
    model_path = model_dir / "minimal_rf_model.pkl"
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Verificar que tenemos todos los componentes necesarios
        required_keys = {'model', 'feature_names', 'feature_dtypes'}
        if not all(key in model_data for key in required_keys):
            raise ValueError("El archivo del modelo no contiene toda la información requerida")
        
        return model_data
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return None

def get_feature_input(feature_name, feature_type):
    """Obtiene entrada del usuario para una característica específica"""
    while True:
        try:
            user_input = input(f"Ingrese el valor para '{feature_name}' ({feature_type}): ")
            
            # Convertir al tipo de dato correcto
            if 'int' in feature_type:
                return int(user_input)
            elif 'float' in feature_type:
                return float(user_input)
            else:
                return str(user_input)
        except ValueError:
            print(f"Error: Por favor ingrese un valor válido para {feature_type}")
            continue

def make_prediction(model, features):
    """Realiza una predicción con el modelo"""
    try:
        # Convertir a DataFrame manteniendo el orden de las columnas
        input_df = pd.DataFrame([features], columns=features.keys())
        
        # Hacer la predicción (el modelo fue entrenado con log(price))
        log_prediction = model.predict(input_df)[0]
        
        # Convertir de vuelta a precio original
        prediction = np.expm1(log_prediction)
        return prediction
    except Exception as e:
        print(f"Error haciendo la predicción: {e}")
        return None

def main():
    print("\n🔮 Predictor de Precios - Modelo Random Forest v8\n")
    
    # Cargar modelo y metadatos
    model_data = load_model_and_features()
    if not model_data:
        return
    
    model = model_data['model']
    feature_names = model_data['feature_names']
    feature_dtypes = model_data['feature_dtypes']
    
    print(f"\nEl modelo espera {len(feature_names)} características:")
    print(", ".join(feature_names))
    
    # Recopilar valores para cada característica
    features = {}
    print("\nPor favor ingrese los valores para cada característica:\n")
    
    for feature in feature_names:
        dtype = feature_dtypes.get(feature, 'str')  # Default a string si no se encuentra
        features[feature] = get_feature_input(feature, dtype)
    
    # Hacer la predicción
    prediction = make_prediction(model, features)
    
    if prediction is not None:
        print(f"\n🎯 Precio estimado: €{prediction:,.2f}")
    
    print("\n✅ Predicción completada\n")

if __name__ == "__main__":
    main()


### Características clave del script:

""" 1. **Carga inteligente del modelo**:
   - Verifica que el archivo del modelo contenga toda la información necesaria
   - Carga nombres de características y sus tipos de datos

2. **Interfaz interactiva**:
   - Solicita valores para cada característica una por una
   - Valida los tipos de datos ingresados por el usuario
   - Proporciona guías claras sobre qué se espera para cada entrada

3. **Conversión de datos**:
   - Convierte automáticamente las entradas a los tipos correctos
   - Maneja la transformación logarítmica/inversa para el precio

4. **Mensajes informativos**:
   - Muestra la lista de características esperadas
   - Proporciona feedback claro durante el proceso
   - Formatea el resultado final de manera legible

### Cómo usar el script:

1. Ejecútalo desde la terminal: `python rf_v8_terminal_predict.py`
2. Sigue las instrucciones para ingresar los valores de cada característica
3. Obtendrás una predicción de precio formateada al final

### Requisitos previos:

- Asegúrate que el archivo `minimal_rf_model.pkl` exista en el directorio `../models/`
- El script debe mantenerse en la misma estructura de directorios que tu código de entrenamiento

Este script está diseñado para funcionar perfectamente con los modelos entrenados por el código de entrenamiento
del Model Training Random Forest v8. """