import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración de la conexión a MongoDB
DATABASE_URL = os.getenv('URL_DB_MONGO').strip('" ')
DB_NAME = os.getenv('DB_NAME').strip()

print(f"Conectando a: {DATABASE_URL}")
print(f"Base de datos: {DB_NAME}")

client = AsyncIOMotorClient(DATABASE_URL)
db = client[DB_NAME]

# Colecciones
users_collection = db.users
search_save_collection = db.search_save

# Operaciones de usuarios
async def get_one_user(username: str):
    user = await users_collection.find_one({"username": username})
    return user
    
async def create_user(username: str, email: str, password: str):
    # En un entorno de producción, deberías usar una biblioteca como passlib para el hash
    # Por ejemplo: hashed_password = pwd_context.hash(password)
    # Por simplicidad, usamos un enfoque básico aquí
    import hashlib
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    new_user = await users_collection.insert_one({
        "username": username,
        "email": email,
        "password": hashed_password
    })
    created_user = await get_one_user(username)
    return created_user

async def update_user(username: str, data: dict):
    await users_collection.update_one({"username": username}, {"$set": data})
    updated_user = await get_one_user(username)
    return updated_user

async def delete_user(username: str):
    result = await users_collection.delete_one({"username": username})
    return result.deleted_count > 0

# Operaciones de búsquedas guardadas
async def save_search(username: str, search_params: dict):
    search_data = {"username": username, "search_params": search_params}
    result = await search_save_collection.insert_one(search_data)
    return result.inserted_id

async def get_user_searches(username: str):
    cursor = search_save_collection.find({"username": username})
    searches = await cursor.to_list(length=100)
    return searches