# Airbnb Price Predictor - Backend/Frontend System
## Random Forest Regressor v8

![Project Structure](https://i.imgur.com/example-image.png)

## 📌 Descripción del Proyecto

Sistema completo para predecir precios de propiedades en Airbnb (Madrid) dirigido a inversores inmobiliarios. Combina:

- **Backend**: API REST con FastAPI y modelo Random Forest
- **Frontend**: Interfaz React con validación avanzada
- **Modelo ML**: Entrenado con 13 features clave

## 🏗️ Estructura del Proyecto

### Backend (`/backend`)
```
backend/
├── models/
│   ├── minimal_rf_model.pkl        # Modelo entrenado
│   └── model_metadata.json         # Metadatos del modelo
├── main.py                         # API FastAPI
├── requirements.txt                # Dependencias
└── test/                           # Pruebas unitarias
```

### Frontend (`/frontend`)
```
frontend/
├── public/
├── src/
│   ├── components/
│   │   └── PredictForm.jsx         # Componente principal
│   ├── App.js
│   └── index.js
├── package.json
└── README.md
```

## 🚀 Instalación y Ejecución

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
**API Docs**: http://localhost:8000/docs

## 📷 Documentación Visual

![FastAPI Swagger UI](https://raw.githubusercontent.com/Bootcamp-IA-P4/Proyecto-Regression---Grupo-3/feature/backend-rf-v8-deploy/imgs/fastapi.png)

---
### Frontend
```bash
cd frontend
npm install
npm start
```
**Aplicación**: http://localhost:3000

## 🔍 Características Clave

### Backend (FastAPI)
✔ Validación estricta con Pydantic v2  
✔ Endpoints:  
- `POST /predict` - Recibe 13 features, devuelve precio predicho  
- `GET /features` - Documentación de features esperadas  
✔ CORS configurado  
✔ Manejo de errores estructurado  

### Frontend (React)
✔ Formulario con validación en tiempo real  
✔ Manejo de decimales (hasta 0.0001)  
✔ Estados de carga/error  
✔ Display de resultados formateados  

## 📊 Modelo de Machine Learning
- **Algoritmo**: Random Forest Regressor
- **Features**:
  ```python
  ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'minimum_nights',
   'number_of_reviews', 'review_scores_rating', 'instant_bookable',
   'neighbourhood_density', 'host_experience', 'room_type_Entire_home_apt',
   'neighbourhood_encoded', 'amenity_score']
  ```
- **Preprocesamiento**: Transformación logarítmica del target

## 🌟 Ejemplo de Uso
1. Introducir datos de propiedad en el formulario
2. Enviar para predicción
3. Recibir precio estimado en euros

![Demo](https://i.imgur.com/example-demo.gif)

## 🤝 Contribución
1. Haz fork del proyecto
2. Crea tu rama (`git checkout -b feature/AmazingFeature`)
3. Haz commit de tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Haz push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia
Distribuido bajo licencia MIT. Ver `LICENSE` para más información.

## ✉️ Contacto
Equipo de Desarrollo - [email@example.com](mailto:email@example.com)

---


## 🛠️ Dependencias Principales

### Backend
```python
fastapi==0.109.1
uvicorn==0.27.0
pydantic==2.5.3
scikit-learn==1.3.2
numpy==1.26.4
```

### Frontend
```json
"react": "^18.2.0",
"axios": "^1.6.2",
"react-dom": "^18.2.0"
```

## 🔗 Endpoints API

| Método | Endpoint    | Descripción                     |
|--------|-------------|---------------------------------|
| POST   | /predict    | Obtener predicción de precio    |
| GET    | /features   | Lista de features requeridas    |

## 🎨 Interfaz de Usuario
- Validación en tiempo real
- Mensajes de error descriptivos
- Diseño responsive
- Animaciones de carga
```
