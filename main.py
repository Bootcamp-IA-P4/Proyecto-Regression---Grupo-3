from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, validator
from typing import List, Optional, Dict, Any
from database import (
    get_one_user, create_user, update_user, delete_user,
    save_search, get_user_searches
)

app = FastAPI(
    title="Análisis Inteligente de Datos de Airbnb",
    description="API para consultar y analizar datos de Airbnb en Madrid",
    version="1.0.0"
)

# Configuración de CORS para permitir solicitudes desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],  # Orígenes del frontend
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP
    allow_headers=["*"],  # Permitir todos los headers
)

# Modelos de datos
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    
    @validator('password')
    def password_must_be_strong(cls, v):
        if len(v) < 8:
            raise ValueError('La contraseña debe tener al menos 8 caracteres')
        if not any(char.isdigit() for char in v):
            raise ValueError('La contraseña debe contener al menos un número')
        if not any(char.isupper() for char in v):
            raise ValueError('La contraseña debe contener al menos una letra mayúscula')
        return v

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    full_name: Optional[str] = None

class SearchParams(BaseModel):
    neighborhood: Optional[str] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    room_type: Optional[str] = None
    min_nights: Optional[int] = None
    # Otros parámetros de búsqueda relevantes

# Rutas principales
@app.get("/") 
def welcome():
    return {'message': "Bienvenido a la API de Análisis de Datos de Airbnb Madrid"}

# Rutas de usuarios
@app.get('/api/users/{username}')
async def get_user_endpoint(username: str):
    user = await get_one_user(username)
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return user

@app.post('/api/users/register')
async def create_user_endpoint(user_data: UserCreate):
    existing_user = await get_one_user(user_data.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="El nombre de usuario ya existe")
    new_user = await create_user(user_data.username, user_data.email, user_data.password)
    
    # Generar un token simple para el usuario (en producción usar JWT)
    import hashlib
    import time
    import os
    token = hashlib.sha256(f"{user_data.username}{time.time()}{os.urandom(8)}".encode()).hexdigest()
    
    # Devolver el usuario con el token
    return {"username": new_user["username"], "email": new_user["email"], "token": token}

@app.post('/api/users/login')
async def login_user(username: str = Body(...), password: str = Body(...)):
    # Verificar si el usuario existe
    user = await get_one_user(username)
    if not user:
        raise HTTPException(status_code=400, detail="Nombre de usuario o contraseña incorrectos")
    
    # Verificar la contraseña
    import hashlib
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    if user["password"] != hashed_password:
        raise HTTPException(status_code=400, detail="Nombre de usuario o contraseña incorrectos")
    
    # Generar un token simple para el usuario (en producción usar JWT)
    import time
    import os
    token = hashlib.sha256(f"{username}{time.time()}{os.urandom(8)}".encode()).hexdigest()
    
    # Devolver el usuario con el token
    return {"username": user["username"], "email": user["email"], "token": token}

@app.put('/api/users/{username}')
async def update_user_endpoint(username: str, user_data: UserUpdate):
    existing_user = await get_one_user(username)
    if not existing_user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    updated_user = await update_user(username, user_data.dict(exclude_unset=True))
    return updated_user

@app.delete('/api/users/{username}')
async def delete_user_endpoint(username: str):
    existing_user = await get_one_user(username)
    if not existing_user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    success = await delete_user(username)
    if success:
        return {"message": f"Usuario {username} eliminado correctamente"}
    raise HTTPException(status_code=500, detail="Error al eliminar el usuario")

# Rutas de búsquedas guardadas
@app.post('/api/users/{username}/searches')
async def save_search_endpoint(username: str, search_params: SearchParams):
    user = await get_one_user(username)
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    search_id = await save_search(username, search_params.dict(exclude_unset=True))
    return {"message": "Búsqueda guardada correctamente", "search_id": str(search_id)}

@app.get('/api/users/{username}/searches')
async def get_searches_endpoint(username: str):
    user = await get_one_user(username)
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    searches = await get_user_searches(username)
    return searches
    