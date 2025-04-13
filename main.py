from email import message
from fastapi import FastAPI 

app = FastAPI()

@app.get("/") # es la ruta inicial del app 
def welcome():
    return {'message': "Bienvenido a mi API"}

@app.get('api/users')
async def get_users():
    return {"Lista de usuarios"}

@app.post('api/users')
async def create_user():
    return {"Crear usuario"}

@app.get('api/users/{user_id}')
async def get_user(user_id: int):
    return {"Usuario": user_id}

@app.put('api/users/{user_id}')
async def update_user(user_id: int):
    return {"Actualizar usuario": user_id}

@app.delete('api/users/{user_id}')
async def delete_user(user_id: int):
    return {"Eliminar usuario": user_id}
    