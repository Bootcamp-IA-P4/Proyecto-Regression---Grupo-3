# Импорт необходимых библиотек / Import required libraries / Импорт необходимых библиотек
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from rf_pipeline import FeatureEngineering  

def load_models():
    """
    Загружает все обученные модели
    Loads all trained models
    Загружает все обученные модели
    
    Retorna / Returns / Возвращает:
    Словарь с загруженными моделями
    Dictionary with loaded models
    Словарь с загруженными моделями
    """
    models = {}
    try:
        models['forest'] = joblib.load('pkls/rf_pipeline_model_finall.joblib')
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    return models

def get_user_input():
    """
    Получает входные данные от пользователя
    Gets input data from user
    Получает входные данные от пользователя
    
    Retorna / Returns / Возвращает:
    DataFrame с входными данными
    DataFrame with input data
    DataFrame с входными данными
    """
    print("\n=== Введите данные для предсказания цены / Enter data for price prediction / Введите данные для предсказания цены ===")
    
    # Получение числовых характеристик / Getting numeric features / Получение числовых характеристик
    accommodates = int(input("Введите количество гостей / Enter number of guests / Введите количество гостей: "))
    bathrooms = float(input("Введите количество ванных комнат / Enter number of bathrooms / Введите количество ванных комнат: "))
    beds = int(input("Введите количество кроватей / Enter number of beds / Введите количество кроватей: "))
    
    # Получение категориальных характеристик / Getting categorical features / Получение категориальных характеристик
    print("\nВыберите тип жилья / Select room type / Выберите тип жилья:")
    print("1. Entire home/apt")
    print("2. Private room")
    print("3. Shared room")
    print("4. Hotel room")
    room_type_choice = int(input("Введите номер типа жилья / Enter room type number / Введите номер типа жилья: "))
    
    room_types = {
        1: 'Entire home/apt',
        2: 'Private room',
        3: 'Shared room',
        4: 'Hotel room'
    }
    room_type = room_types.get(room_type_choice, 'Entire home/apt')
    
    # Запрос района / Getting neighbourhood / Запрос района
    print("\nВведите значение района (от 0 до 255) / Enter neighbourhood value (0-255) / Введите значение района (0-255):")
    neighbourhood_val = float(input("Значение района / Neighbourhood value / Значение района: "))
    
    # Создание DataFrame с входными данными / Creating DataFrame with input data / Создание DataFrame с входными данными
    input_data = pd.DataFrame({
        'accommodates': [accommodates],
        'bathrooms_numeric': [bathrooms],
        'beds': [beds],
        'room_type_Entire home/apt': [1 if room_type == 'Entire home/apt' else 0],
        'room_type_Hotel room': [1 if room_type == 'Hotel room' else 0],
        'room_type_Private room': [1 if room_type == 'Private room' else 0],
        'room_type_Shared room': [1 if room_type == 'Shared room' else 0],
        'neighbourhood_encoded': [neighbourhood_val]
    })
    
    return input_data

def make_predictions(models, input_data):
    """
    Делает предсказания с использованием всех моделей
    Makes predictions using all models
    Делает предсказания с использованием всех моделей
    
    Parámetros / Parameters / Параметры:
    models: словарь с моделями
            dictionary with models
            словарь с моделями
    input_data: DataFrame с входными данными
                DataFrame with input data
                DataFrame с входными данными
    
    Retorna / Returns / Возвращает:
    Словарь с предсказаниями
    Dictionary with predictions
    Словарь с предсказаниями
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
    Отображает предсказания всех моделей
    
    Parámetros / Parameters / Параметры:
    predictions: словарь с предсказаниями
                dictionary with predictions
                словарь с предсказаниями
    """
    print("\n=== Результаты предсказаний / Prediction results / Результаты предсказаний ===")
    for model_name, pred in predictions.items():
        if pred is not None:
            print(f"{model_name.capitalize()}: ${pred:.2f}")
    
    # Вычисление среднего предсказания / Calculating average prediction / Вычисление среднего предсказания
    valid_predictions = [p for p in predictions.values() if p is not None]
    if valid_predictions:
        avg_pred = sum(valid_predictions) / len(valid_predictions)
        print(f"\nСреднее предсказание / Average prediction / Среднее предсказание: ${avg_pred:.2f}")

def predict_price(accommodates, bathrooms, beds, room_type, neighbourhood_val):
    """Предсказание цены с использованием обеих моделей"""
    
    # Загрузка моделей из папки pkls
    rf_pipeline = joblib.load('models/pkls/rf_pipeline_model_finall.joblib')
    xgb_pipeline = joblib.load('models/pkls/xgb_pipeline_model_final.joblib')
    
    # Предсказание с использованием Random Forest
    rf_price = predict_price_rf(rf_pipeline, **{
        'accommodates': accommodates,
        'bathrooms': bathrooms,
        'beds': beds,
        'room_type': room_type,
        'neighbourhood_val': neighbourhood_val
    })
    
    # Предсказание с использованием XGBoost
    xgb_price = predict_price_xgb(xgb_pipeline, **{
        'accommodates': accommodates,
        'bathrooms': bathrooms,
        'beds': beds,
        'room_type': room_type,
        'neighbourhood_val': neighbourhood_val
    })
    
    # Вычисление среднего предсказания
    avg_price = (rf_price + xgb_price) / 2
    
    return {
        'random_forest': rf_price,
        'xgboost': xgb_price,
        'average': avg_price
    }

def main():
    """
    Основная функция для интерактивного предсказания цен
    Main function for interactive price prediction
    Основная функция для интерактивного предсказания цен
    """
    # Загрузка моделей / Loading models / Загрузка моделей
    print("Загрузка моделей... / Loading models... / Загрузка моделей...")
    models = load_models()
    if models is None:
        print("Не удалось загрузить модели / Failed to load models / Не удалось загрузить модели")
        return
    
    while True:
        # Получение входных данных / Getting input data / Получение входных данных
        input_data = get_user_input()
        
        # Создание предсказаний / Making predictions / Создание предсказаний
        predictions = make_predictions(models, input_data)
        
        # Отображение результатов / Displaying results / Отображение результатов
        display_predictions(predictions)
        
        # Проверка на продолжение / Check for continuation / Проверка на продолжение
        cont = input("\nХотите сделать еще одно предсказание? (y/n) / Want to make another prediction? (y/n) / Хотите сделать еще одно предсказание? (y/n): ")
        if cont.lower() != 'y':
            break

if __name__ == "__main__":
    main() 