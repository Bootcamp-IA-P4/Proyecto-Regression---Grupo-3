import os
import jupytext
from pathlib import Path

# Configuración
SRC_DIR = "src/"  # Ruta base
OUTPUT_DIR = "notebooks/"  # Carpeta destino
TARGET_PY = "scripts/inside_airbnb_eda_3_last_edited.py"  # Ruta relativa al archivo a convertir

# Crear directorio de salida si no existe
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def convert_py_to_ipynb(py_relative_path):
    """Convierte un archivo .py a .ipynb manteniendo estructura de carpetas"""
    py_path = os.path.join(SRC_DIR, py_relative_path)
    ipynb_path = os.path.join(OUTPUT_DIR, py_relative_path).replace(".py", ".ipynb")
    
    # Crear estructura equivalente en notebooks
    os.makedirs(os.path.dirname(ipynb_path), exist_ok=True)
    
    # Leer el archivo .py y escribirlo como .ipynb
    with open(py_path, "r") as py_file:
        notebook = jupytext.read(py_file, fmt="py")
    with open(ipynb_path, "w") as ipynb_file:
        jupytext.write(notebook, ipynb_file, fmt="ipynb")
    
    print(f"✅ Convertido: {py_path} -> {ipynb_path}")

# Ejecutar la conversión solo para el archivo especificado
convert_py_to_ipynb(TARGET_PY)
