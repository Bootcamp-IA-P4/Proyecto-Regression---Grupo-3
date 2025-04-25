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

#ñistado de barrios 
async def get_all_neighborhoods():
    neighborhoods = []
    cursor = db.neighbourhood.find({}, {"_id": 0})  # Excluimos el _id para simplificar la respuesta
    async for document in cursor:
        neighborhoods.append(document)
    return neighborhoods

async def get_beds():
    try:
        unique_beds = []
        # Verificamos si la colección existe
        if "listings_pre" not in await db.list_collection_names():
            print("La colección 'listings_pre' no existe en la base de datos")
            return [1, 2, 3, 4, 5]  # Valores predeterminados
            
        # Usamos distinct para obtener valores únicos
        cursor = await db.listings_pre.distinct("beds")
        
        # Filtramos valores None y manejamos los valores Double
        for bed in cursor:
            if bed is not None:
                # Como los valores son Double, los mantenemos como float
                unique_beds.append(float(bed))
        
        # Ordenamos los valores para mejor presentación
        unique_beds = sorted(unique_beds)
        return unique_beds
    except Exception as e:
        print(f"Error en get_beds: {e}")
        # Devolvemos valores predeterminados en caso de error
        return [1, 2, 3, 4, 5]

async def get_accommodates():
    unique_accommodates = []
    cursor = await db.listings_pre.distinct("accommodates")
    unique_accommodates = sorted(cursor)
    return unique_accommodates

async def get_bathrooms():
    unique_bathrooms = []
    cursor = await db.listings_pre.distinct("bathrooms_numeric")
    unique_bathrooms = sorted(cursor)
    return unique_bathrooms


async def get_neighbourhood_value(neighborhood):
    """
    Convierte el nombre del barrio a un valor numérico para el modelo.
    
    Args:
        neighborhood: Nombre del barrio
        
    Returns:
        int: Valor numérico del barrio
    """
    # Implementar la lógica para convertir el nombre del barrio a un valor numérico
    # Esto podría ser una consulta a una base de datos o un diccionario predefinido
    neighborhood_mapping = {
        # Mapeo de barrios a valores numéricos
        "Centro": 128,
        "Salamanca": 200,
        # Añadir más barrios según sea necesario
    }
    return neighborhood_mapping.get(neighborhood, 128)  # Valor predeterminado

async def get_average_price(neighborhood):
    """Obtiene el precio medio actual para un barrio"""
    # Implementar lógica para obtener el precio medio actual
    return 85.0  # Valor de ejemplo
