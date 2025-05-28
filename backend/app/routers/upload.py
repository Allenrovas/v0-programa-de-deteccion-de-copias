# Endpoints para subir archivos
import os
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
import shutil
from app.services.extractor import extract_zip_files
from app.services.similarity import process_submissions

router = APIRouter(prefix="/upload", tags=["upload"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/")
async def upload_files(
    files: List[UploadFile] = File(...),
    language: str = Form(...)
):
    if not files or len(files) < 2:
        raise HTTPException(status_code=400, detail="Se requieren al menos dos archivos ZIP")
    
    if language not in ["python", "java", "cpp", "javascript"]:
        raise HTTPException(status_code=400, detail="Lenguaje no soportado")
    
    # Crear directorio único para esta sesión
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    # Guardar archivos ZIP
    zip_paths = []
    for file in files:
        if not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail=f"El archivo {file.filename} no es un ZIP")
        
        file_path = os.path.join(session_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        zip_paths.append(file_path)
    
    # Extraer archivos
    try:
        extracted_dirs = extract_zip_files(zip_paths, session_dir)
        
        # Iniciar procesamiento asíncrono
        # Esto debería ser una tarea en segundo plano en producción
        result = await process_submissions(extracted_dirs, language, session_id)
        
        return {
            "session_id": session_id,
            "message": "Archivos procesados correctamente",
            "num_files": len(files),
            "language": language,
            "result_summary": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando archivos: {str(e)}")