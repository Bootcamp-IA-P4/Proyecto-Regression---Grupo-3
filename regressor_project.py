#  Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, 'data', 'final_data2.csv')
print(f"Looking for data at: {data_path}")
        
df = pd.read_csv(data_path)

#  Setup for Advanced Regression Models

# Set random seed for reproducibility
np.random.seed(42)

# Feature Engineering
# Create engineered features to improve model performance

def engineer_features(df):
    """Create engineered features for improved model performance"""
    df_fe = df.copy()

    # Interaction terms
    df_fe['accommodates_beds'] = df_fe['accommodates'] * df_fe['beds']
    df_fe['accommodates_bathrooms'] = df_fe['accommodates'] * df_fe['bathrooms_numeric']
    df_fe['beds_bathrooms'] = df_fe['beds'] * df_fe['bathrooms_numeric']

    # Polynomial features
    df_fe['accommodates_squared'] = df_fe['accommodates'] ** 2
    df_fe['bathrooms_squared'] = df_fe['bathrooms_numeric'] ** 2
    df_fe['beds_squared'] = df_fe['beds'] ** 2

    # Ratio features
    df_fe['beds_per_person'] = df_fe['beds'] / df_fe['accommodates']
    df_fe['baths_per_person'] = df_fe['bathrooms_numeric'] / df_fe['accommodates']

    # Replace infinities with 0 (in case of division by zero)
    df_fe.replace([np.inf, -np.inf], 0, inplace=True)

    # Fill NaN values with 0
    df_fe.fillna(0, inplace=True)

    return df_fe

# Apply feature engineering to our dataset
df_fe = engineer_features(df)

df_fe.head()

# Prepare data for modeling
X_fe = df_fe.drop(['id', 'last_scraped', 'price'], axis=1)
y_fe = df_fe['price']

# Train-test split with the enhanced feature set
X_train, X_test, y_train, y_test = train_test_split(X_fe, y_fe, test_size=0.2, random_state=42)

# Standardize the features for better optimization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Number of features after engineering: {X_train.shape[1]}")
print(f"Train set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Optuna Hyperparameter Optimization for XGBoost
def objective(trial):
    """Optuna objective function for XGBoost hyperparameter optimization"""
    # Hyperparameter search space
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
    }

    # Add parameters based on booster type
    if param['booster'] == 'gbtree' or param['booster'] == 'dart':
        param['max_depth'] = trial.suggest_int('max_depth', 3, 12)
        param['eta'] = trial.suggest_float('eta', 0.01, 0.3, log=True)
        param['gamma'] = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
        param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        param['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
        param['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        param['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 10)

    # K-fold cross-validation for more robust evaluation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, valid_idx in kf.split(X_train_scaled):
        X_train_fold, X_valid_fold = X_train_scaled[train_idx], X_train_scaled[valid_idx]
        y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dvalid = xgb.DMatrix(X_valid_fold, label=y_valid_fold)

        # Train the model
        model = xgb.train(param, dtrain, num_boost_round=1000,
                          evals=[(dvalid, 'validation')],
                          early_stopping_rounds=50,
                          verbose_eval=False)

        # Predict and evaluate
        preds = model.predict(dvalid)
        rmse = np.sqrt(mean_squared_error(y_valid_fold, preds))
        cv_scores.append(rmse)

    # Return the mean RMSE across all folds
    return np.mean(cv_scores)

# Setup and run the optimization study
print("Starting Optuna hyperparameter optimization for XGBoost...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)  # Use more trials for better results

print("\nBest trial:")
trial = study.best_trial
print(f"  Value (RMSE): {trial.value:.4f}")
print("  Best hyperparameters:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

#  Train the optimized XGBoost model
# Get the best parameters from the study


best_params =  {
    'booster': 'gbtree',
    'lambda': 0.9104814210454353,
    'alpha': 1.213886019978349e-05,
    'max_depth': 6,
    'eta': 0.07708970936388457,
    'gamma': 0.0004693060323028388,
    'grow_policy': 'lossguide',
    'subsample': 0.8912801035803928,
    'colsample_bytree': 0.6544912688955683,
    'min_child_weight': 10,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

# Create a validation set from the training data
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_scaled, y_train, test_size=0.25, random_state=42)

# Convert data to DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train_final, label=y_train_final)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# Train the final model with the best parameters
print("\nTraining final XGBoost model with optimized hyperparameters...")
final_model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=3000,
    evals=[(dtrain, 'train'), (dval, 'validation')],
    early_stopping_rounds=50,
    verbose_eval=100
)

# Make predictions on the test set (only after training is complete)
y_pred_xgb = final_model.predict(dtest)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae = mean_absolute_error(y_test, y_pred_xgb)
r2 = r2_score(y_test, y_pred_xgb)

print("\nXGBoost Model Evaluation:")
print(f"RMSE: ${rmse:.2f}")
print(f"MAE: ${mae:.2f}")
print(f"R²: {r2:.4f}")

#  Train additional models for comparison
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet

print("\n=== TRAINING ADDITIONAL MODELS FOR COMPARISON ===")

# Function to evaluate and compare different models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print results
    print(f"\n{model_name} Model Performance:")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE: ${mae:.2f}")
    print(f"R²: {r2:.4f}")

    return rmse, mae, r2, y_pred

# Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_metrics = evaluate_model(rf_model, X_train_scaled, X_test_scaled, y_train, y_test, "Random Forest")

# Ridge Regression
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_metrics = evaluate_model(ridge_model, X_train_scaled, X_test_scaled, y_train, y_test, "Ridge")

# Lasso Regression
lasso_model = Lasso(alpha=0.1, random_state=42)
lasso_metrics = evaluate_model(lasso_model, X_train_scaled, X_test_scaled, y_train, y_test, "Lasso")

# ElasticNet
elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_metrics = evaluate_model(elastic_model, X_train_scaled, X_test_scaled, y_train, y_test, "ElasticNet")

# Compare model performances
model_names = ['XGBoost', 'Random Forest', 'Ridge', 'Lasso', 'ElasticNet']
rmse_values = [rmse, rf_metrics[0], ridge_metrics[0], lasso_metrics[0], elastic_metrics[0]]
r2_values = [r2, rf_metrics[2], ridge_metrics[2], lasso_metrics[2], elastic_metrics[2]]

# Create a comparison dataframe
comparison_df = pd.DataFrame({
    'Model': model_names,
    'RMSE': rmse_values,
    'R²': r2_values
})

# Sort by RMSE (lower is better)
comparison_df = comparison_df.sort_values('RMSE')

print("\nModel Comparison:")
print(comparison_df)

# Visualize model comparison
plt.figure(figsize=(14, 10))

# RMSE comparison
plt.subplot(2, 1, 1)
sns.barplot(x='Model', y='RMSE', data=comparison_df)
plt.title('Model Comparison: RMSE (lower is better)')
plt.ylabel('RMSE ($)')
plt.xticks(rotation=45)

for i, v in enumerate(comparison_df['RMSE']):
    plt.text(i, v + 0.5, f'${v:.2f}', ha='center')

# R² comparison
plt.subplot(2, 1, 2)
sns.barplot(x='Model', y='R²', data=comparison_df)
plt.title('Model Comparison: R² (higher is better)')
plt.ylabel('R² Score')
plt.xticks(rotation=45)

for i, v in enumerate(comparison_df['R²']):
    plt.text(i, v - 0.05, f'{v:.4f}', ha='center')

plt.tight_layout()
plt.show()

"""# Random Forest"""

# Create proper train-test split
# We'll keep the test set completely separate from the optimization process
X_train_full, X_test, y_train_full, y_test = train_test_split(X_fe, y_fe, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

print(f"Number of features after engineering: {X_train_full.shape[1]}")
print(f"Training set shape: {X_train_full.shape}")
print(f"Test set shape: {X_test.shape}")

from sklearn.model_selection import train_test_split, KFold, cross_val_score

def objective(trial):
    """Optuna objective function for Random Forest hyperparameter optimization"""
    # Define the hyperparameter search space
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

    # Use K-fold cross-validation to get a robust estimate of performance
    # This way we never expose the test set during optimization
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize the Random Forest model
    model = RandomForestRegressor(
        **params,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    # Perform cross-validation and return the mean RMSE
    cv_scores = -cross_val_score(model, X_train_full_scaled, y_train_full,
                                cv=cv, scoring='neg_root_mean_squared_error')

    return np.mean(cv_scores)  # Return mean RMSE (lower is better)

# Run Optuna Study for Random Forest
print("Starting Optuna hyperparameter optimization for Random Forest...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Get the best parameters from the study
best_params = study.best_params

# Initialize the model with the best parameters
rf_best = RandomForestRegressor(
    **best_params,
    random_state=42,
    n_jobs=-1
)

# Fit the model on the full training set
print("\nTraining final Random Forest model with optimized hyperparameters...")
rf_best.fit(X_train_full_scaled, y_train_full)

# Make predictions on the test set
y_pred_rf = rf_best.predict(X_test_scaled)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae = mean_absolute_error(y_test, y_pred_rf)
r2 = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Model Evaluation:")
print(f"RMSE: ${rmse:.2f}")
print(f"MAE: ${mae:.2f}")
print(f"R²: {r2:.4f}")

# Check for OVERFITTING in the Random Forest model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Compute predictions for both training and test sets
y_pred_train = rf_best.predict(X_train_full_scaled)
y_pred_test = rf_best.predict(X_test_scaled)

# Calculate metrics for both sets
train_rmse = np.sqrt(mean_squared_error(y_train_full, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train_full, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Print performance comparison
print("Random Forest Performance Comparison:")
print(f"Training RMSE: ${train_rmse:.2f}")
print(f"Test RMSE: ${test_rmse:.2f}")
print(f"RMSE difference: ${test_rmse - train_rmse:.2f}")
print(f"RMSE ratio (test/train): {test_rmse / train_rmse:.2f}x")
print("\n")
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"R² difference: {test_r2 - train_r2:.4f}")

# Calculate overfitting percentage
overfitting_percentage = ((test_rmse - train_rmse) / train_rmse) * 100
print(f"\nOverfitting percentage: {overfitting_percentage:.2f}%")

# Get feature importance
feature_importance = rf_best.feature_importances_
feature_names = X_train_full.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Calculate percentage of total importance
total_importance = importance_df['Importance'].sum()
importance_df['Importance_Percentage'] = (importance_df['Importance'] / total_importance) * 100

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
plt.title('Random Forest Feature Importance (Top 15)')
plt.tight_layout()
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(12, 8))

# Actual vs Predicted
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred_rf, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Random Forest: Actual vs Predicted Prices')

# Residual plot
residuals = y_test - y_pred_rf
plt.subplot(2, 2, 2)
plt.scatter(y_pred_rf, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Random Forest: Residuals Plot')

#Price Prediction Function
def predict_price_rf(accommodates, bathrooms, beds, room_type, neighbourhood_val):
    """
    Predict the price for a given accommodation using the optimized Random Forest model.

    Parameters:
    - accommodates: Number of people it accommodates
    - bathrooms: Number of bathrooms
    - beds: Number of beds
    - room_type: 'Entire home/apt', 'Hotel room', 'Private room', or 'Shared room'
    - neighbourhood_val: Neighbourhood encoded value

    Returns:
    - Predicted price
    """
    # Create a DataFrame with the input values
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

    # Apply feature engineering
    input_engineered = engineer_features(input_data)

    # Drop id and last_scraped columns if they exist
    if 'id' in input_engineered.columns:
        input_engineered = input_engineered.drop('id', axis=1)
    if 'last_scraped' in input_engineered.columns:
        input_engineered = input_engineered.drop('last_scraped', axis=1)
    if 'price' in input_engineered.columns:
        input_engineered = input_engineered.drop('price', axis=1)

    # Scale the features
    input_scaled = scaler.transform(input_engineered)

    # Make prediction
    predicted_price = rf_best.predict(input_scaled)[0]

    return max(0, round(predicted_price, 2))  # Ensure non-negative price

example_price = predict_price_rf(
    accommodates=4,
    bathrooms=2,
    beds=3,
    room_type='Entire home/apt',
    neighbourhood_val=130.0
)

print(f"\nExample prediction using Random Forest model:")
print(f"For a 4-person, 2-bathroom, 3-bed entire home in neighbourhood value 130:")
print(f"Predicted price: ${example_price:.2f}")

import joblib
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Save the Random Forest model
joblib.dump(rf_best, 'models/random_forest_model.joblib')
print("Random Forest model saved to: models/random_forest_model.joblib")

# Save the scaler
joblib.dump(scaler, 'models/scaler.joblib')
print("Scaler saved to: models/scaler.joblib")

# Load the model and scaler
loaded_model = joblib.load('models/random_forest_model2.joblib')
loaded_scaler = joblib.load('models/scaler2.joblib')

print("\nModel and scaler loaded successfully")

# Make a test prediction with the loaded model
import pandas as pd
import numpy as np

def predict_price(model, scaler, accommodates, bathrooms, beds, room_type, neighbourhood_val):
    """Simple function to make a price prediction"""
    # Create input data
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

    # Apply feature engineering (simplified version with just the necessary features)
    input_fe = input_data.copy()
    input_fe['accommodates_beds'] = input_fe['accommodates'] * input_fe['beds']
    input_fe['accommodates_bathrooms'] = input_fe['accommodates'] * input_fe['bathrooms_numeric']
    input_fe['beds_bathrooms'] = input_fe['beds'] * input_fe['bathrooms_numeric']
    input_fe['accommodates_squared'] = input_fe['accommodates'] ** 2
    input_fe['bathrooms_squared'] = input_fe['bathrooms_numeric'] ** 2
    input_fe['beds_squared'] = input_fe['beds'] ** 2
    input_fe['beds_per_person'] = input_fe['beds'] / input_fe['accommodates']
    input_fe['baths_per_person'] = input_fe['bathrooms_numeric'] / input_fe['accommodates']
    input_fe.replace([np.inf, -np.inf], 0, inplace=True)
    input_fe.fillna(0, inplace=True)

    # Scale features
    input_scaled = scaler.transform(input_fe)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    return round(prediction, 2)

# Test prediction
price = predict_price(
    model=loaded_model,
    scaler=loaded_scaler,
    accommodates=4,
    bathrooms=2,
    beds=3,
    room_type='Entire home/apt',
    neighbourhood_val=130.0
)

print(f"Test prediction for a 4-person, 2-bathroom, 3-bed entire home: ${price}")