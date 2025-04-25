# Importar las librerías necesarias
import pickle
import pandas as pd
import numpy as np

# Cargar el modelo
print("Cargando el modelo...")
try:
    with open('modelo_xgboost_combinado.pkl', 'rb') as file:
        modelo = pickle.load(file)
    print("Modelo cargado exitosamente!")
except FileNotFoundError:
    print("Error: No se encontró el archivo del modelo.")
    exit(1)
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit(1)

# Función para solicitar datos al usuario
def solicitar_datos():
    print("\n--- Ingrese los datos para la predicción ---")
    datos = {}
    
    # Solicitar datos numéricos
    print("\nDatos numéricos:")
    datos['accommodates'] = int(input("Capacidad (número de personas): "))
    datos['beds'] = float(input("Número de camas: "))
    
    # Convertir baños a formato numérico
    baños_texto = input("Número de baños (ejemplo: 1, 1.5, 2): ")
    try:
        datos['bathrooms_numeric'] = float(baños_texto)
    except ValueError:
        print("Valor no válido para baños, se usará 1.0 por defecto")
        datos['bathrooms_numeric'] = 1.0
    
    # Solicitar tipo de habitación
    print("\nTipo de habitación:")
    tipo_habitacion = input("Tipo de habitación (1: Habitación privada, 2: Casa/Apto completo, 3: Habitación compartida, 4: Habitación de hotel): ")
    
    # Inicializar todas las columnas de tipo de habitación en 0
    datos['room_type_Private room'] = 0
    datos['room_type_Entire home/apt'] = 0
    datos['room_type_Shared room'] = 0
    datos['room_type_Hotel room'] = 0
    
    # Establecer la columna correspondiente en 1 según la selección
    if tipo_habitacion == '1':
        datos['room_type_Private room'] = 1
    elif tipo_habitacion == '2':
        datos['room_type_Entire home/apt'] = 1
    elif tipo_habitacion == '3':
        datos['room_type_Shared room'] = 1
    elif tipo_habitacion == '4':
        datos['room_type_Hotel room'] = 1
    else:
        print("Opción no válida, se usará 'Habitación privada' por defecto")
        datos['room_type_Private room'] = 1
    
    # Solicitar barrio y codificarlo
    print("\nBarrio:")
    print("Seleccione un barrio (1-10):")
    print("1: Centro, 2: Salamanca, 3: Chamberí, 4: Retiro, 5: Chamartín")
    print("6: Tetuán, 7: Arganzuela, 8: Moncloa, 9: Ciudad Lineal, 10: Otros")
    barrio = input("Selección: ")
    
    # Codificar el barrio (simplificado para este ejemplo)
    try:
        barrio_num = int(barrio)
        if 1 <= barrio_num <= 10:
            datos['neighbourhood_encoded'] = barrio_num * 10  # Multiplicamos por 10 como ejemplo de codificación
        else:
            datos['neighbourhood_encoded'] = 100  # Valor por defecto
    except ValueError:
        datos['neighbourhood_encoded'] = 100  # Valor por defecto
    
    # Crear un DataFrame con los datos ingresados
    df = pd.DataFrame([datos])
    
    return df

# Función para realizar la predicción
def predecir_precio(datos):
    try:
        # Realizar la predicción
        prediccion = modelo.predict(datos)
        
        # Si el modelo usa transformación logarítmica, deshacer la transformación
        if hasattr(modelo, 'named_steps') and 'regressor' in modelo.named_steps:
            # Asumimos que se aplicó transformación logarítmica
            prediccion = np.expm1(prediccion)
        
        return prediccion[0]
    except Exception as e:
        print(f"Error al realizar la predicción: {e}")
        return None

# Función principal
def main():
    print("=== SISTEMA DE PREDICCIÓN DE PRECIOS DE AIRBNB ===")
    
    while True:
        # Solicitar datos
        datos = solicitar_datos()
        
        # Realizar predicción
        precio_predicho = predecir_precio(datos)
        
        if precio_predicho is not None:
            print(f"\nPrecio predicho: ${precio_predicho:.2f} por noche")
        
        # Preguntar si desea hacer otra predicción
        continuar = input("\n¿Desea realizar otra predicción? (s/n): ")
        if continuar.lower() != 's':
            break
    
    print("\n¡Gracias por usar el sistema de predicción!")

if __name__ == "__main__":
    main()