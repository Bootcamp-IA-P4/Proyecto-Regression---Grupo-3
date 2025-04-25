# Importar las librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Cargar los datos
df = pd.read_csv('data/final_data.csv')

# Preparar los datos
X = df.drop(['id', 'last_scraped', 'price'], axis=1)
y = df['price']

# Identificar columnas numéricas y categóricas
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Crear preprocesadores para cada tipo de columna
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un pipeline con XGBoost (Basico)
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ))
])

# Entrenar el modelo
print("Entrenando modelo XGBoost...")
xgb_pipeline.fit(X_train, y_train)

# Realizar predicciones
y_pred = xgb_pipeline.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nMétricas de evaluación del modelo XGBoost:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

# Optimización de hiperparámetros
param_grid = {
    'regressor__n_estimators': [150, 200, 250],
    'regressor__learning_rate': [0.15, 0.2, 0.25],
    'regressor__max_depth': [6, 7, 8],
    'regressor__min_child_weight': [1, 2],
    'regressor__gamma': [0, 0.1],
    'regressor__subsample': [0.9, 1.0],
    'regressor__colsample_bytree': [0.9, 1.0]
}

grid_search = GridSearchCV(
    xgb_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

print("Iniciando búsqueda de hiperparámetros...")
grid_search.fit(X_train, y_train)

print("Mejores hiperparámetros:", grid_search.best_params_)
best_xgb_model = grid_search.best_estimator_

# Evaluar con el mejor modelo
y_pred_best = best_xgb_model.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)
r2_best = r2_score(y_test, y_pred_best)
mae_best = mean_absolute_error(y_test, y_pred_best)

print(f"\nMétricas con el modelo optimizado:") #Tras la inclusion de los hiperparametros
print(f"MSE: {mse_best:.2f}")
print(f"RMSE: {rmse_best:.2f}")
print(f"MAE: {mae_best:.2f}")
print(f"R²: {r2_best:.2f}")



# Aplicar transformación logarítmica a los precios
print("Aplicando transformación logarítmica...")
y_log = np.log1p(y)  # log(1+y) para manejar valores cero

# Dividir los datos transformados
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Crear un pipeline con XGBoost
xgb_pipeline_log = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ))
])

# Entrenar el modelo con datos transformados
print("Entrenando modelo XGBoost con transformación logarítmica...")
xgb_pipeline_log.fit(X_train, y_train_log)

# Predecir y transformar de vuelta
y_pred_log = xgb_pipeline_log.predict(X_test)
y_pred_original = np.expm1(y_pred_log)  # Transformar de vuelta a la escala original
y_test_original = np.expm1(y_test_log)  # Transformar de vuelta para calcular métricas

# Evaluar el modelo
mse_log = mean_squared_error(y_test_original, y_pred_original)
rmse_log = np.sqrt(mse_log)
r2_log = r2_score(y_test_original, y_pred_original)
mae_log = mean_absolute_error(y_test_original, y_pred_original)

print(f"\nMétricas con transformación logarítmica:")
print(f"MSE: {mse_log:.2f}")
print(f"RMSE: {rmse_log:.2f}")
print(f"MAE: {mae_log:.2f}")
print(f"R²: {r2_log:.2f}")

# Visualizar importancia de características
xgb_model = xgb_pipeline.named_steps['regressor']

# Validación cruzada para evaluar la robustez del modelo
cv_scores = cross_val_score(
    xgb_pipeline, 
    X, y, 
    cv=5, 
    scoring='neg_mean_squared_error'
)

# Convertir a RMSE
rmse_scores = np.sqrt(-cv_scores)
print(f"\nRMSE en validación cruzada: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")

# Ingeniería de características
print("Aplicando ingeniería de características...")
df_engineered = df.copy()

#Nuevas características (ajustamos según las columnas disponibles en el dataset)
# Ratio de baños por habitación
if 'bedrooms' in df.columns and 'bathrooms' in df.columns:
    df_engineered['bath_per_bed'] = df['bathrooms'] / df['bedrooms'].replace(0, 1)

# Ratio de personas por habitación
if 'accommodates' in df.columns and 'bedrooms' in df.columns:
    df_engineered['people_per_bed'] = df['accommodates'] / df['bedrooms'].replace(0, 1)

# Características de densidad de reviews
if 'number_of_reviews' in df.columns and 'review_scores_rating' in df.columns:
    df_engineered['review_density'] = df['number_of_reviews'] * df['review_scores_rating']

# Características binarias
if 'instant_bookable' in df.columns:
    df_engineered['is_instant_bookable'] = df['instant_bookable'].map({'t': 1, 'f': 0})

# Preparar los datos con nuevas características
X_engineered = df_engineered.drop(['id', 'last_scraped', 'price'], axis=1)
y_engineered = df_engineered['price']

# Identificar columnas numéricas y categóricas
numeric_features_eng = X_engineered.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features_eng = X_engineered.select_dtypes(include=['object']).columns.tolist()

# Crear preprocesadores para cada tipo de columna
preprocessor_eng = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features_eng),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_eng)
    ])

# Dividir los datos
X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(
    X_engineered, y_engineered, test_size=0.2, random_state=42
)

# Crear pipeline
xgb_pipeline_eng = Pipeline(steps=[
    ('preprocessor', preprocessor_eng),
    ('regressor', xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ))
])

# Entrenar modelo
print("Entrenando modelo con características ingenierizadas...")
xgb_pipeline_eng.fit(X_train_eng, y_train_eng)

# Evaluar modelo
y_pred_eng = xgb_pipeline_eng.predict(X_test_eng)
mse_eng = mean_squared_error(y_test_eng, y_pred_eng)
rmse_eng = np.sqrt(mse_eng)
r2_eng = r2_score(y_test_eng, y_pred_eng)
mae_eng = mean_absolute_error(y_test_eng, y_pred_eng)

print(f"\nMétricas con ingeniería de características:")
print(f"MSE: {mse_eng:.2f}")
print(f"RMSE: {rmse_eng:.2f}")
print(f"MAE: {mae_eng:.2f}")
print(f"R²: {r2_eng:.2f}")

# Ahora puedes continuar con la combinación de técnicas


# Combinación de técnicas: ingeniería de características + transformación logarítmica + filtrado de outliers
print("\nAplicando combinación de técnicas...")

# 1. Ingeniería de características
df_combined = df_engineered.copy()

# 2. Filtrar outliers
Q1 = y_engineered.quantile(0.25)
Q3 = y_engineered.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outlier_mask = (y_engineered >= lower_bound) & (y_engineered <= upper_bound)

X_combined = X_engineered[outlier_mask]
y_combined = y_engineered[outlier_mask]

# 3. Transformación logarítmica
y_combined_log = np.log1p(y_combined)

# Dividir datos
X_train_comb, X_test_comb, y_train_comb_log, y_test_comb_log = train_test_split(
    X_combined, y_combined_log, test_size=0.2, random_state=42
)

# Crear pipeline
xgb_pipeline_comb = Pipeline(steps=[
    ('preprocessor', preprocessor_eng),
    ('regressor', xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ))
])

# Entrenar modelo
print("Entrenando modelo combinado...")
xgb_pipeline_comb.fit(X_train_comb, y_train_comb_log)

# Predecir y transformar de vuelta
y_pred_comb_log = xgb_pipeline_comb.predict(X_test_comb)
y_pred_comb = np.expm1(y_pred_comb_log)
y_test_comb = np.expm1(y_test_comb_log)

# Evaluar modelo
mse_comb = mean_squared_error(y_test_comb, y_pred_comb)
rmse_comb = np.sqrt(mse_comb)
r2_comb = r2_score(y_test_comb, y_pred_comb)
mae_comb = mean_absolute_error(y_test_comb, y_pred_comb)

print(f"\nMétricas con combinación de técnicas:")
print(f"MSE: {mse_comb:.2f}")
print(f"RMSE: {rmse_comb:.2f}")
print(f"MAE: {mae_comb:.2f}")
print(f"R²: {r2_comb:.2f}")

# ... existing code ...

# Después de entrenar el modelo combinado (xgb_pipeline_comb)
print("\nEvaluando overfitting del modelo combinado...")

# Métricas en datos de entrenamiento
y_train_pred_log = xgb_pipeline_comb.predict(X_train_comb)
y_train_pred = np.expm1(y_train_pred_log)
y_train_original = np.expm1(y_train_comb_log)

train_mse = mean_squared_error(y_train_original, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train_original, y_train_pred)
train_mae = mean_absolute_error(y_train_original, y_train_pred)

print("\nMétricas en datos de entrenamiento:")
print(f"RMSE: {train_rmse:.2f}")
print(f"MAE: {train_mae:.2f}")
print(f"R²: {train_r2:.2f}")

print("\nMétricas en datos de prueba:")
print(f"RMSE: {rmse_comb:.2f}")
print(f"MAE: {mae_comb:.2f}")
print(f"R²: {r2_comb:.2f}")

# Calcular diferencias porcentuales
rmse_diff = ((rmse_comb - train_rmse) / train_rmse) * 100
mae_diff = ((mae_comb - train_mae) / train_mae) * 100
r2_diff = ((train_r2 - r2_comb) / train_r2) * 100

print("\nDiferencias porcentuales (prueba vs entrenamiento):")
print(f"Diferencia RMSE: {rmse_diff:.2f}%")
print(f"Diferencia MAE: {mae_diff:.2f}%")
print(f"Diferencia R²: {r2_diff:.2f}%")

# Validación cruzada para una evaluación más robusta
cv_scores = cross_val_score(
    xgb_pipeline_comb,
    X_combined,
    y_combined_log,
    cv=5,
    scoring='neg_mean_squared_error'
)

cv_rmse_scores = np.sqrt(-cv_scores)
print(f"\nRMSE en validación cruzada: {cv_rmse_scores.mean():.2f} ± {cv_rmse_scores.std():.2f}")

# ... existing code ...

'''# Guardar el modelo combinado (el mejor) como archivo .pkl
import pickle

# Definir el nombre del archivo
model_filename = 'modelo_xgboost_combinado.pkl'

# Guardar el modelo
print(f"\nGuardando el modelo en {model_filename}...")
with open(model_filename, 'wb') as file:
    pickle.dump(xgb_pipeline_comb, file)

print(f"Modelo guardado exitosamente en {model_filename}")

# Ejemplo de cómo cargar el modelo para hacer predicciones
print("\nEjemplo de cómo cargar y usar el modelo guardado:")
print("with open('modelo_xgboost_combinado.pkl', 'rb') as file:")
print("    modelo_cargado = pickle.load(file)")
print("# Hacer predicciones")
print("predicciones = modelo_cargado.predict(X_test_comb)")'''