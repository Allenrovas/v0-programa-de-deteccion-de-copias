import os
import uuid
import stat
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
import shutil
from services.extractor import extract_zip_files
from services.similarity import process_submissions

router = APIRouter(prefix="/api/upload", tags=["upload"])

# Use system temp directory instead of local uploads
UPLOAD_DIR = tempfile.gettempdir() #backend\uploads

def set_permissions_recursive(path):
    """Set read/write permissions recursively for all files and directories"""
    try:
        for root, dirs, files in os.walk(path):
            # Set directory permissions
            for d in dirs:
                dir_path = os.path.join(root, d)
                try:
                    os.chmod(dir_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                except PermissionError:
                    pass
            
            # Set file permissions
            for f in files:
                file_path = os.path.join(root, f)
                try:
                    os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)
                except PermissionError:
                    pass
    except Exception as e:
        print(f"Warning: Could not set permissions: {e}")

def safe_remove_readonly(func, path, exc_info):
    """Error handler for removing read-only files"""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

@router.post("/")
async def upload_files(
    files: List[UploadFile] = File(...),
    language: str = Form(...)
):
    if not files or len(files) < 2:
        raise HTTPException(status_code=400, detail="Se requieren al menos dos archivos ZIP")
    
    if language not in ["python", "java", "cpp", "javascript"]:
        raise HTTPException(status_code=400, detail="Lenguaje no soportado")
    
    print(f"Recibidos {len(files)} archivos para el lenguaje: {language}")

    # Create unique directory for this session using pathlib
    session_id = str(uuid.uuid4())
    session_dir = Path(UPLOAD_DIR) / "plagiarism_checker" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Save ZIP files
    zip_paths = []
    try:
        for file in files:
            if not file.filename.endswith('.zip'):
                raise HTTPException(status_code=400, detail=f"El archivo {file.filename} no es un ZIP")
            
            file_path = session_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Set permissions immediately after creating file
            try:
                os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)
            except PermissionError:
                pass
            
            zip_paths.append(str(file_path))
        
        # Extract files with proper error handling
        extracted_dirs = extract_zip_files(zip_paths, str(session_dir))
        
        # Set permissions on extracted files
        for extracted_dir in extracted_dirs:
            set_permissions_recursive(extracted_dir)
        
        # Process submissions
        result = await process_submissions(extracted_dirs, language, session_id)
        
        return {
            "session_id": session_id,
            "message": "Archivos procesados correctamente",
            "num_files": len(files),
            "language": language,
            "result_summary": result
        }
        
    except Exception as e:
        # Clean up on error
        try:
            if session_dir.exists():
                shutil.rmtree(str(session_dir), onerror=safe_remove_readonly)
        except Exception:
            pass
        
        raise HTTPException(status_code=500, detail=f"Error procesando archivos: {str(e)}")