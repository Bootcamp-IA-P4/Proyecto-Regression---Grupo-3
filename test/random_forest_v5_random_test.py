# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import random

# --- Configuración de paths ---
PROJECT_ROOT = Path(__file__).parent.parent  # Ajusta según tu estructura
MODEL_PATH = PROJECT_ROOT / "artifacts" / "random_forest_v5.pkl"
SCALER_PATH = PROJECT_ROOT / "artifacts" / "scaler.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "artifacts" / "feature_columns.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "df_optimized.csv"  # CSV original

def load_artifacts():
    """Cargar modelo, scaler y columnas desde archivos .pkl"""
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(FEATURE_COLUMNS_PATH, 'rb') as f:
        feature_columns = pickle.load(f)
    return model, scaler, feature_columns

def generate_random_test_data(df_original, feature_columns, n_samples=5):
    """Genera datos aleatorios basados en estadísticas del dataset original"""
    random_data = {}
    for col in feature_columns:
        if df_original[col].dtype in ['int64', 'float64']:
            # Para numéricas: valor dentro del rango [Q1, Q3] (evita outliers)
            q1 = df_original[col].quantile(0.25)
            q3 = df_original[col].quantile(0.75)
            random_data[col] = [random.uniform(q1, q3) for _ in range(n_samples)]
        elif df_original[col].dtype == 'object':
            # Para categóricas: muestra aleatoria de valores únicos
            random_data[col] = random.choices(df_original[col].dropna().unique(), k=n_samples)
        else:
            # Para booleanas (ej: host_is_superhost): 0 o 1
            random_data[col] = [random.randint(0, 1) for _ in range(n_samples)]
    return pd.DataFrame(random_data)

def predict_price(model, scaler, X_test):
    """Preprocesa y predice precios"""
    # Escalar features numéricas (si el scaler existe)
    numeric_cols = X_test.select_dtypes(include=['int64', 'float64']).columns
    if scaler:
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    # Predecir y convertir a EUR
    return np.expm1(model.predict(X_test))

if __name__ == "__main__":
    # --- Cargar artefactos ---
    model, scaler, feature_columns = load_artifacts()
    print(f"🔄 Columnas esperadas: {feature_columns}")

    # --- Generar datos aleatorios consistentes ---
    df_original = pd.read_csv(DATA_PATH)
    X_test_random = generate_random_test_data(df_original, feature_columns, n_samples=3)
    
    print("\n🎲 Datos de prueba generados (aleatorios pero consistentes):")
    print(X_test_random)

    # --- Predecir ---
    prices = predict_price(model, scaler, X_test_random)
    for i, price in enumerate(prices):
        print(f"\n🏠 Predicción {i+1}: ${price:.2f} EUR")

    # --- Opcional: Exportar datos de prueba a CSV ---
    X_test_random.to_csv(PROJECT_ROOT / "data" / "processed" / "test_samples.csv", index=False)
    print("\n💾 Datos guardados en 'data/test_samples.csv'")