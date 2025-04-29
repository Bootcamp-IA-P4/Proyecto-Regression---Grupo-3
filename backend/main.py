# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from pathlib import Path
import logging
from fastapi.middleware.cors import CORSMiddleware

# Configuración inicial
app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# CORS (para conectar con React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, reemplazar con tu dominio de React
    allow_methods=["POST"],
)

# Modelo Pydantic para validación de entrada
class PropertyFeatures(BaseModel):
    accommodates: float
    bathrooms: float
    bedrooms: float
    beds: float
    minimum_nights: float
    number_of_reviews: int
    review_scores_rating: float
    instant_bookable: int  # 0 o 1
    neighbourhood_density: float
    host_experience: float
    room_type_Entire_home_apt: int  # 0 o 1
    neighbourhood_encoded: int
    amenity_score: float

# Cargar modelo al iniciar
try:
    model_path = Path("models/minimal_rf_model.pkl")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]
    feature_names = model_data["feature_names"]
    logger.info("✅ Modelo cargado correctamente")
except Exception as e:
    logger.error(f"❌ Error cargando el modelo: {e}")
    raise

@app.post("/predict")
async def predict(features: PropertyFeatures):
    try:
        # Convertir entrada a lista en el orden correcto
        input_data = [getattr(features, feature) for feature in feature_names]
        
        # Predecir y convertir a precio real
        log_prediction = model.predict([input_data])[0]
        prediction = np.expm1(log_prediction)
        
        return {
            "predicted_price": round(prediction, 2),
            "currency": "EUR",
            "features_used": feature_names
        }
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/features")
async def get_expected_features():
    """Endpoint para que el frontend sepa qué campos necesita"""
    return {"features": feature_names}