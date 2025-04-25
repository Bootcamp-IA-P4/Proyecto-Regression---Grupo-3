# Импорт необходимых библиотек / Import required libraries / Импорт необходимых библиотек
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_models():
    """
    Загружает все обученные модели
    Loads all trained models
    Carga todos los modelos entrenados
    
    Retorna / Returns / Возвращает:
    Словарь с загруженными моделями
    Dictionary with loaded models
    Diccionario con modelos cargados
    """
    models = {}
    try:
        models['linear'] = joblib.load('models/pkls/linear_model.pkl')
        models['tree'] = joblib.load('models/pkls/tree_model.pkl')
        models['forest'] = joblib.load('models/pkls/forest_model.pkl')
        models['xgboost'] = joblib.load('models/pkls/xgb_model.pkl')
        models['lightgbm'] = joblib.load('models/pkls/lgbm_model.pkl')
        models['catboost'] = joblib.load('models/pkls/catboost_model.pkl')
        models['stacking'] = joblib.load('models/pkls/stacking_model.pkl')
    except Exception as e:
        print(f"Error loading models: {e}")
        return None
    return models

def get_user_input():
    """
    Получает входные данные от пользователя
    Gets input data from user
    Recibe información del usuario
    
    Retorna / Returns / Возвращает:
    DataFrame с входными данными
    DataFrame with input data
    DataFrame con datos de entrada
    """
    print("\n=== Введите данные для предсказания цены / Enter data for price prediction / Introduzca datos para predecir el precio ===")
    
    # Получение числовых характеристик / Getting numeric features / Obtención de características numéricas
    accommodates = int(input("Введите количество гостей / Enter number of guests / Введите количество гостей: "))
    bathrooms = float(input("Введите количество ванных комнат / Enter number of bathrooms / Введите количество ванных комнат: "))
    beds = int(input("Введите количество кроватей / Enter number of beds / Введите количество кроватей: "))
    
    # Получение категориальных характеристик / Getting categorical features / Obtención de características categóricas
    print("\nВыберите тип жилья / Select room type / Seleccione el tipo de alojamiento:")
    print("1. Entire home/apt")
    print("2. Private room")
    print("3. Shared room")
    print("4. Hotel room")
    room_type_choice = int(input("Введите номер типа жилья / Enter room type number / Introduzca el número del tipo de vivienda: "))
    
    room_types = {
        1: 'Entire home/apt',
        2: 'Private room',
        3: 'Shared room',
        4: 'Hotel room'
    }
    room_type = room_types.get(room_type_choice, 'Entire home/apt')
    
    # Создание DataFrame с входными данными / Creating DataFrame with input data / Crear un DataFrame con datos de entrada
    input_data = pd.DataFrame({
        'accommodates': [accommodates],
        'bathrooms_numeric': [bathrooms],
        'beds': [beds],
        'room_type': [room_type],
        'neighbourhood_encoded': [0]  # Значение по умолчанию / Default value / Valor predeterminado
    })
    
    return input_data

def make_predictions(models, input_data):
    """
    Делает предсказания с использованием всех моделей
    Makes predictions using all models
    Hace predicciones utilizando todos los modelos
    
    Parámetros / Parameters / Параметры:
    models: словарь с моделями
            dictionary with models
            diccionario con modelos
    input_data: DataFrame с входными данными
                DataFrame with input data
                DataFrame con datos de entrada
    
    Retorna / Returns / Возвращает:
    Словарь с предсказаниями
    Dictionary with predictions
    Diccionario con predicciones
    """
    predictions = {}
    for model_name, model in models.items():
        try:
            pred = model.predict(input_data)[0]
            predictions[model_name] = max(0, round(pred, 2))  # Гарантируем неотрицательную цену
        except Exception as e:
            print(f"Error making prediction with {model_name}: {e}")
            predictions[model_name] = None
    
    return predictions

def display_predictions(predictions):
    """
    Отображает предсказания всех моделей
    Displays predictions from all models
    Muestra predicciones de todos los modelos.
    
    Parámetros / Parameters / Параметры:
    predictions: словарь с предсказаниями
                dictionary with predictions
                diccionario con predicciones
    """
    print("\n=== Результаты предсказаний / Prediction results / Resultados de la predicción ===")
    for model_name, pred in predictions.items():
        if pred is not None:
            print(f"{model_name.capitalize()}: ${pred:.2f}")
    
    # Вычисление среднего предсказания / Calculating average prediction / Calcular la predicción media
    valid_predictions = [p for p in predictions.values() if p is not None]
    if valid_predictions:
        avg_pred = sum(valid_predictions) / len(valid_predictions)
        print(f"\nСреднее предсказание / Average prediction / Predicción promedio: ${avg_pred:.2f}")

def main():
    """
    Основная функция для интерактивного предсказания цен
    Main function for interactive price prediction
    Función básica para la predicción interactiva de precios
    """
    # Загрузка моделей / Loading models / Cargando modelos
    print("Загрузка моделей... / Loading models... / Cargando modelos...")
    models = load_models()
    if models is None:
        print("Не удалось загрузить модели / Failed to load models / No se pudieron cargar los modelos")
        return
    
    while True:
        # Получение входных данных / Getting input data / Recibir datos de entrada
        input_data = get_user_input()
        
        # Создание предсказаний / Making predictions / Haciendo predicciones
        predictions = make_predictions(models, input_data)
        
        # Отображение результатов / Displaying results / Visualización de resultados
        display_predictions(predictions)
        
        # Проверка на продолжение / Check for continuation / Comprobar continuación
        cont = input("\nХотите сделать еще одно предсказание? (y/n) / Want to make another prediction? (y/n) / ¿Quieres hacer otra predicción? (y/n): ")
        if cont.lower() != 'y':
            break

if __name__ == "__main__":
    main() 