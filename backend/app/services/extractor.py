# Extracción de archivos ZIP
import os
import zipfile
import shutil

def extract_zip_files(zip_paths, output_dir):
    """
    Extrae múltiples archivos ZIP en directorios separados
    """
    extracted_dirs = []
    
    for zip_path in zip_paths:
        # Crear directorio para este ZIP basado en su nombre
        zip_name = os.path.basename(zip_path).replace('.zip', '')
        extract_dir = os.path.join(output_dir, zip_name)
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extraer contenido
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        extracted_dirs.append(extract_dir)
    
    return extracted_dirs