# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import json

# --- Configuración de paths ---
PROJECT_ROOT = Path(__file__).parent.parent  # Ajusta según tu estructura
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_v5.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "models" / "feature_columns.pkl"

def load_artifacts():
    """Cargar modelo, scaler y columnas desde archivos .pkl"""
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(FEATURE_COLUMNS_PATH, 'rb') as f:
        feature_columns = pickle.load(f)
    return model, scaler, feature_columns

def get_user_input(feature_columns):
    """Recolecta datos de entrada del usuario vía terminal"""
    input_data = {}
    print("\n🛠️  Ingresa los valores para cada feature (presiona Enter para omitir y usar valor por defecto):")
    
    # Valores por defecto (basados en medianas o modas)
    default_values = {
        'accommodates': 2,
        'bathrooms': 1.0,
        'bedrooms': 1,
        'beds': 1,
        'minimum_nights': 2,
        'number_of_reviews': 10,
        'review_scores_rating': 90,
        'host_is_superhost': 0,
        'neighbourhood_density': 0.5,
        'has_wifi': 1,
        'has_air_conditioning': 1
    }
    
    for col in feature_columns:
        while True:
            try:
                user_input = input(f"{col} (default={default_values.get(col, 'N/A')}): ").strip()
                if not user_input:  # Si el usuario presiona Enter
                    input_data[col] = default_values[col]
                    break
                # Conversión de tipos
                if isinstance(default_values.get(col), float):
                    input_data[col] = float(user_input)
                elif isinstance(default_values.get(col), int):
                    input_data[col] = int(user_input)
                else:
                    input_data[col] = user_input
                break
            except ValueError:
                print(f"⚠️ Error: Ingresa un valor válido para {col} (ej: {default_values.get(col)})")
    
    return pd.DataFrame([input_data])

def predict_price(model, scaler, input_data):
    """Preprocesa y predice precios"""
    # Escalar features numéricas
    numeric_cols = input_data.select_dtypes(include=['int64', 'float64']).columns
    if scaler:
        input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
    # Predecir y convertir a EUR
    return np.expm1(model.predict(input_data))[0]

if __name__ == "__main__":
    # --- Cargar modelo y columnas ---
    model, scaler, feature_columns = load_artifacts()
    print(f"🔍 Columnas requeridas: {feature_columns}")
    
    # --- Obtener datos del usuario ---
    user_data_df = get_user_input(feature_columns)
    
    # --- Predecir y mostrar resultado ---
    predicted_price = predict_price(model, scaler, user_data_df)
    print(f"\n🎯 Precio predicho: ${predicted_price:.2f} EUR")
    
    # --- Salida en JSON (para integrar con React) ---
    output_json = {
        "input_data": user_data_df.iloc[0].to_dict(),
        "predicted_price": round(float(predicted_price), 2)
    }
    print("\n📤 JSON para React (copiar esto):")
    print(json.dumps(output_json, indent=2))