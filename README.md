# 📊 Análisis Inteligente de Datos de Airbnb

Este proyecto tiene como objetivo principal analizar datos abiertos de Airbnb mediante técnicas de regresión y análisis de datos, para extraer conclusiones valiosas, generar visualizaciones interactivas y presentar un informe ejecutivo claro. Se desarrollará una solución completa compuesta por un backend en **FastAPI**, una base de datos **MongoDB**, y un frontend interactivo para visualización de datos.

---

## 📁 Dataset Principal

🔗 [Airbnb Open Data - Kaggle](https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata)

> Se podrán incorporar datasets adicionales relacionados con el mercado de alquiler, turismo, estacionalidad, precios o cualquier otro factor que ayude a enriquecer el análisis.

---

## 🎯 Objetivos del Proyecto

- Analizar datos de Airbnb con algoritmos de regresión para entender patrones de precios y otros factores clave.
- Construir un informe ejecutivo con las conclusiones del análisis.
- Desarrollar una aplicación web que permita visualizar gráficamente la información obtenida.
- Fomentar el trabajo colaborativo usando herramientas modernas de desarrollo.

---

## 🚀 Entregables

| Entregable              | Descripción                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| Google Colab           | Notebook con limpieza, análisis exploratorio, regresiones y visualizaciones. |
| Informe Ejecutivo      | PDF/Markdown con conclusiones, gráficas clave y recomendaciones.            |
| Aplicación Web         | Dashboard con gráficos interactivos, tablas, y exploración de los datos.    |
| Repositorio GitHub     | Código fuente bien documentado y organizado.                                |

---

## 🧰 Stack Tecnológico

| Componente     | Tecnología                                         |
|----------------|---------------------------------------------------|
| Backend        | [FastAPI](https://fastapi.tiangolo.com/)          |
| Base de datos  | [MongoDB](https://www.mongodb.com/)               |
| Frontend       | [React](https://reactjs.org/) + [Vite.js]  |
| Ciencia de Datos | [Python](https://www.python.org/), [Pandas](https://pandas.pydata.org/), [Scikit-learn](https://scikit-learn.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [Google Colab](https://colab.research.google.com/) | [XGBoost](https://xgboost.ai) | 
| Colaboración   | [GitHub](https://github.com/), [GitHub Projects](https://github.com/features/project-management) |

---

## 🧪 Módulos y Funcionalidades Esperadas

### 🔍 Análisis de Datos
- Limpieza y preprocesamiento.
- Análisis exploratorio (EDA).
- Algoritmos de regresión (lineal, polinómica, random forest, etc.).
- Evaluación de modelos y métricas.

### 🧩 Backend (API)
- Endpoints RESTful para consultar datos analizados.
- Integración con MongoDB.
- Seguridad básica y documentación automática (Swagger UI).

### 🌐 Frontend
- Visualización de datos:
  - Gráficos de precios por zonas, fechas, tipo de propiedad, etc.
  - Mapas interactivos (opcional).
  - Filtros por ciudad, rango de precios, fechas.
- Tablas de datos exportables.
- Panel de comparación de predicciones vs datos reales.

---

## 🧑💻 Organización del Equipo

- **Control de versiones**: Git + GitHub.
- **Planificación y tareas**: GitHub Projects.
- **Entorno de trabajo colaborativo**: Google Colab, branches por funcionalidad, pull requests y code reviews.

---

### **🌿 Nomenclatura de Ramas (Git Branching)**  
Se sigue un flujo basado en **Git Flow modificado** para garantizar un desarrollo organizado. Las ramas deben nombrarse así:  

#### **Ramas Principales**  
| Rama       | Descripción                                                                 | Origen       | Destino de Merge |  
|------------|-----------------------------------------------------------------------------|--------------|-------------------|  
| `main`     | Versión estable en producción (solo releases validados).                    | -            | -                |  
| `dev`      | Integración de features en desarrollo.                                      | `main`       | `test`           |  
| `test`     | Entorno de pruebas pre-producción (QA).                                     | `main`       | `main`           |  

#### **Ramas de Soporte**  
| Tipo de Rama  | Convención               | Ejemplo                | Origen       | Destino de Merge |  
|---------------|--------------------------|------------------------|--------------|-------------------|  
| **Feature**   | `feature/<nombre>`       | `feature/eda`          | `dev`        | `dev`            |  
| **Hotfix**    | `hotfix/<descripción>`   | `hotfix/login-error`   | `main`       | `main` + `dev`   |  


### **📌 Reglas Clave**  
1. **Prefixes obligatorios**: Usar siempre `feature/`, `hotfix/`, etc.  
2. **Nombres descriptivos**: En inglés y en minúsculas, separados por guiones (`feature/user-authentication`).  
3. **Protección de ramas**:  
   - `main` y `test` están protegidas (requieren **PR** y aprobación).  
   - `dev` acepta merges directos desde features.  


### **💡 Buenas Prácticas**  
- **Sincronizar antes de crear ramas**:  
  ```bash
  git fetch --all && git pull origin dev
  ```  

---

## 📂 Estructura del Proyecto (sugerida)
```markdown
airbnb-data-analysis/
│
├── data/                   # Datos (estructura tipo data science)
│   ├── raw/                # Datos crudos (sin procesar)
│   ├── processed/          # Datos procesados/transformados
│   └── external/           # Datos de terceros (APIs, descargas)
│
├── notebooks/              # Análisis exploratorios y experimentos
│   └── airbnb_analysis.ipynb  # Jupyter Notebook principal
│
├── backend/                # Backend (API/ETL)
│   ├── app/                # Módulos de la aplicación
│   ├── main.py             # Punto de entrada
│   └── requirements.txt    # Dependencias de Python
│
├── frontend/               # Frontend (visualización/interfaz)
│   ├── public/             # Assets estáticos (HTML, imágenes)
│   ├── src/                # Código fuente (JS/React/Vue)
│   └── package.json        # Dependencias de Node.js
│
├── reports/                # Reportes/documentación
│   └── informe_ejecutivo.md  # Conclusiones en markdown
│
├── README.md               # Documentación principal del proyecto
└── .gitignore              # Archivos excluidos de Git
```
