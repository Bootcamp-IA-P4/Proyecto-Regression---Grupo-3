import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """
    Загружает данные из CSV файла
    """
    return pd.read_csv(file_path)

def handle_missing_values(df, strategy='mean'):
    """
    Обрабатывает пропущенные значения
    """
    imputer = SimpleImputer(strategy=strategy)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df

def encode_categorical(df, categorical_columns):
    """
    Кодирует категориальные переменные
    """
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
    df = df.drop(categorical_columns, axis=1)
    return pd.concat([df, encoded_df], axis=1)

def scale_numeric(df, numeric_columns):
    """
    Масштабирует числовые переменные
    """
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

def prepare_features_target(df, target_column):
    """
    Разделяет данные на признаки и целевую переменную
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y

def preprocess_data(file_path, target_column, categorical_columns=None, numeric_columns=None):
    """
    Основная функция предобработки данных
    """
    # Загрузка данных
    df = load_data(file_path)
    
    # Обработка пропущенных значений
    df = handle_missing_values(df)
    
    # Если категориальные колонки не указаны, определяем их автоматически
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Если числовые колонки не указаны, определяем их автоматически
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = numeric_columns.drop(target_column)
    
    # Кодирование категориальных переменных
    if len(categorical_columns) > 0:
        df = encode_categorical(df, categorical_columns)
    
    # Масштабирование числовых переменных
    if len(numeric_columns) > 0:
        df = scale_numeric(df, numeric_columns)
    
    # Разделение на признаки и целевую переменную
    X, y = prepare_features_target(df, target_column)
    
    return X, y 