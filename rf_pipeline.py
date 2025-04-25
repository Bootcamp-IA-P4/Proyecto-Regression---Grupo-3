import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import optuna
import joblib
import os

class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Класс для создания новых признаков"""
    
    def __init__(self):
        self.original_feature_names = None
        self.feature_names = None
        
    def fit(self, X, y=None):
        # Сохраняем исходные имена признаков, если X - это DataFrame
        if hasattr(X, 'columns'):
            self.original_feature_names = X.columns.tolist()
        return self
    
    def transform(self, X):
        # Преобразуем в DataFrame, если это numpy массив
        if not hasattr(X, 'columns'):
            if self.original_feature_names is None:
                # Если исходные имена признаков не сохранены, создаем стандартные
                self.original_feature_names = [f'f{i}' for i in range(X.shape[1])]
            X_fe = pd.DataFrame(X, columns=self.original_feature_names)
        else:
            X_fe = X.copy()
        
        # Сохраняем текущие имена признаков
        self.feature_names = X_fe.columns.tolist()
        
              
        # Создаем новые признаки на основе доступных
        # Например, если у нас есть числовые признаки, мы можем создавать их квадраты
        numeric_features = X_fe.select_dtypes(include=['int64', 'float64']).columns
        for feature in numeric_features:
            X_fe[f'{feature}_squared'] = X_fe[feature] ** 2
        
        # Замена бесконечностей нулями
        X_fe.replace([np.inf, -np.inf], 0, inplace=True)
        
        # Заполнение пропущенных значений
        X_fe.fillna(0, inplace=True)
        
        return X_fe

def objective_rf_pipeline(trial, X_train, y_train):
    # Определение пространства поиска гиперпараметров
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 25),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
    }
    
    if params['bootstrap']:
        params['oob_score'] = trial.suggest_categorical('oob_score', [True, False])
    else:
        params['oob_score'] = False
    
    # Создание пайплайна
    pipeline = Pipeline([
        ('feature_engineering', FeatureEngineering()),
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(**params, random_state=42, n_jobs=-1))
    ])
    
    # Кросс-валидация
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = -cross_val_score(
        pipeline, X_train, y_train,
        cv=cv, 
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    return np.mean(cv_scores)

def calculate_overfitting(train_metrics, test_metrics):
    """Расчет степени переобучения модели"""
    overfitting = {}
    for metric in train_metrics.keys():
        if metric in test_metrics:
            # Для RMSE и MAE - чем меньше, тем лучше
            if metric in ['rmse', 'mae']:
                overfitting[metric] = (test_metrics[metric] - train_metrics[metric]) / train_metrics[metric]
            # Для R² - чем больше, тем лучше
            elif metric == 'r2':
                overfitting[metric] = (train_metrics[metric] - test_metrics[metric]) / test_metrics[metric]
    return overfitting

def print_metrics_with_overfitting(model_name, train_metrics, test_metrics):
    """Вывод метрик с информацией о переобучении"""
    overfitting = calculate_overfitting(train_metrics, test_metrics)
    
    print(f"\n{model_name} Metrics:")
    print("Training Set:")
    print(f"RMSE: ${train_metrics['rmse']:.2f}")
    print(f"MAE: ${train_metrics['mae']:.2f}")
    print(f"R²: {train_metrics['r2']:.4f}")
    
    print("\nTest Set:")
    print(f"RMSE: ${test_metrics['rmse']:.2f}")
    print(f"MAE: ${test_metrics['mae']:.2f}")
    print(f"R²: {test_metrics['r2']:.4f}")
    
    print("\nOverfitting Analysis:")
    for metric, value in overfitting.items():
        if metric in ['rmse', 'mae']:
            print(f"{metric.upper()} overfitting: {value:.2%}")
        elif metric == 'r2':
            print(f"R² overfitting: {value:.2%}")
    
    # Анализ степени переобучения
    print("\nOverfitting Assessment:")
    if any(abs(value) > 0.2 for value in overfitting.values()):
        print("⚠️ Warning: Significant overfitting detected!")
    elif any(abs(value) > 0.1 for value in overfitting.values()):
        print("ℹ️ Note: Moderate overfitting detected")
    else:
        print("✅ Good: No significant overfitting detected")

def train_and_evaluate_rf():
    # Определение путей
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'final_data2.csv')
    
    # Загрузка данных
    print(f"Looking for data at: {data_path}")
    df = pd.read_csv(data_path)
    
    # Вывод информации о признаках
    print("\nДоступные признаки в данных:")
    print(df.columns.tolist())
    
    # Подготовка данных
    X = df.drop(['id', 'last_scraped', 'price'], axis=1)
    y = df['price']
    
    # Вывод информации о признаках после подготовки
    print("\nПризнаки после подготовки:")
    print(X.columns.tolist())
    
    # Разделение на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nNumber of features: {X_train.shape[1]}")
    print(f"Train set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Создание директории для моделей
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Оптимизация гиперпараметров для Random Forest
    print("\nStarting Optuna hyperparameter optimization for Random Forest...")
    rf_study = optuna.create_study(direction='minimize')
    rf_study.optimize(lambda trial: objective_rf_pipeline(trial, X_train, y_train), n_trials=20)
    
    # Лучшие параметры для Random Forest
    rf_best_params = rf_study.best_params
    print("\nBest hyperparameters for Random Forest:")
    for key, value in rf_best_params.items():
        print(f"{key}: {value}")
    
    # Создание итогового пайплайна для Random Forest
    rf_pipeline = Pipeline([
        ('feature_engineering', FeatureEngineering()),
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(**rf_best_params, random_state=42, n_jobs=-1))
    ])
    
    # Обучение итогового RF пайплайна
    print("\nTraining final Random Forest model with optimized hyperparameters...")
    rf_pipeline.fit(X_train, y_train)
    
    # Оценка на обучающей выборке
    y_train_pred = rf_pipeline.predict(X_train)
    train_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'mae': mean_absolute_error(y_train, y_train_pred),
        'r2': r2_score(y_train, y_train_pred)
    }
    
    # Оценка на тестовой выборке
    y_test_pred = rf_pipeline.predict(X_test)
    test_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'r2': r2_score(y_test, y_test_pred)
    }
    
    print_metrics_with_overfitting("Random Forest", train_metrics, test_metrics)
    
    # Сохранение RF пайплайна
    joblib.dump(rf_pipeline, 'models/rf_pipeline_model_finall.joblib')
    print("Random Forest Pipeline saved to: models/rf_pipeline_model_final.joblib")
    
    return rf_pipeline

def predict_price_rf(pipeline, **kwargs):
    """Предсказание цены с использованием Random Forest пайплайна"""
    # Создаем DataFrame с точными именами признаков из данных
    input_data = pd.DataFrame({
        'accommodates': [kwargs.get('accommodates', 0)],
        'bathrooms_numeric': [kwargs.get('bathrooms', 0)],
        'beds': [kwargs.get('beds', 0)],
        'room_type_Entire home/apt': [1 if kwargs.get('room_type') == 'Entire home/apt' else 0],
        'room_type_Hotel room': [1 if kwargs.get('room_type') == 'Hotel room' else 0],
        'room_type_Private room': [1 if kwargs.get('room_type') == 'Private room' else 0],
        'room_type_Shared room': [1 if kwargs.get('room_type') == 'Shared room' else 0],
        'neighbourhood_encoded': [kwargs.get('neighbourhood_val', 0)]
    })
    
    predicted_price = pipeline.predict(input_data)[0]
    return max(0, round(predicted_price, 2))

if __name__ == "__main__":
    rf_pipeline = train_and_evaluate_rf()
    
    # Пример предсказания с правильными именами признаков
    example = {
        'accommodates': 4,
        'bathrooms': 2,
        'beds': 3,
        'room_type': 'Entire home/apt',
        'neighbourhood_val': 130.0
    }
    
    rf_price = predict_price_rf(rf_pipeline, **example)
    print(f"\nRandom Forest predicted price: ${rf_price:.2f}") 