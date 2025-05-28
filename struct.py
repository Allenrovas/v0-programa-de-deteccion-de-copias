import os

# Estructura de carpetas y archivos
structure = {
    "backend": {
        "app": {
            "__init__.py": "",
            "main.py": "# Punto de entrada de la API\n",
            "routers": {
                "__init__.py": "",
                "upload.py": "# Endpoints para subir archivos\n"
            },
            "core": {
                "__init__.py": "",
                "config.py": "# Configuraciones\n",
                "security.py": "# Autenticación (si es necesario)\n"
            },
            "services": {
                "__init__.py": "",
                "extractor.py": "# Extracción de archivos ZIP\n",
                "tokenizer": {
                    "__init__.py": "",
                    "python_tokenizer.py": "",
                    "java_tokenizer.py": "",
                    "cpp_tokenizer.py": "",
                    "js_tokenizer.py": ""
                },
                "similarity": {
                    "__init__.py": "",
                    "token_based.py": "",
                    "ml_based.py": ""
                },
                "report.py": "# Generación de reportes\n"
            },
            "models": {
                "__init__.py": "",
                "ml_models": {}
            }
        },
        "tests": {},
        "requirements.txt": "# Dependencias\n"
    }
}

def create_structure(base_path, struct):
    for name, content in struct.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):  # Carpeta
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:  # Archivo
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

# Ejecutar la creación
create_structure(".", structure)
print("Estructura creada con éxito.")
