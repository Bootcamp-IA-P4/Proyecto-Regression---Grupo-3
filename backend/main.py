# backend/main.py
from wsgiref.validate import validator
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
import pickle
import numpy as np
from pathlib import Path
import logging
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Configuración inicial
app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# CORS Config (mejorada)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Error-Type"]
)

# Modelo Pydantic con validación estricta
class PropertyFeatures(BaseModel):
    accommodates: float = Field(..., gt=0, description="Número de huéspedes")
    bathrooms: float = Field(..., gt=0)
    bedrooms: float = Field(..., gt=0)
    beds: float = Field(..., gt=0)
    minimum_nights: float = Field(..., gt=0)
    number_of_reviews: int = Field(..., ge=0)
    review_scores_rating: float = Field(..., ge=0, le=100)
    instant_bookable: int = Field(..., ge=0, le=1)
    neighbourhood_density: float = Field(..., ge=0)
    host_experience: float = Field(..., ge=0)
    room_type_Entire_home_apt: int = Field(..., ge=0, le=1)
    neighbourhood_encoded: int = Field(..., ge=0)
    amenity_score: float = Field(..., ge=0)

    @field_validator('*', mode='before')
    def check_nan_values(cls, v):
        if isinstance(v, float) and np.isnan(v):
            raise ValueError("No se permiten valores NaN")
        return v

# Cargar modelo con verificación de features
try:
    model_path = Path("models/minimal_rf_model.pkl")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    model = model_data["model"]
    feature_names: List[str] = model_data["feature_names"]
    
    # Verificar integridad del modelo
    if not all(isinstance(f, str) for f in feature_names):
        raise ValueError("Nombres de features inválidos en el modelo")
    
    logger.info(f"✅ Modelo cargado. Features esperadas: {feature_names}")
except Exception as e:
    logger.critical(f"❌ Error cargando modelo: {str(e)}")
    raise

@app.post("/predict", response_model=dict)
async def predict(features: PropertyFeatures):
    try:
        # Convertir a diccionario y mantener orden
        input_dict = features.model_dump()
        input_data = [input_dict[feature] for feature in feature_names]
        
        # Validación adicional del input
        if len(input_data) != len(feature_names):
            raise ValueError("Número de features no coincide con el modelo")
        
        # Predecir
        log_pred = model.predict([input_data])[0]
        prediction = float(np.expm1(log_pred))  # Convertir explícitamente
        
        return {
            "predicted_price": round(prediction, 2),
            "currency": "EUR",
            "model_version": "1.0"
        }
        
    except ValueError as ve:
        logger.warning(f"Error de validación: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"type": "validation_error", "msg": str(ve)}
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"type": "server_error", "msg": "Error interno al procesar la predicción"}
        )

@app.get("/features", response_model=dict)
async def get_features():
    """Endpoint para documentación"""
    return {
        "required_features": feature_names,
        "feature_descriptions": {
            "instant_bookable": "0: No, 1: Sí",
            "room_type_Entire_home_apt": "0: Habitación, 1: Casa completa"
        }
    }