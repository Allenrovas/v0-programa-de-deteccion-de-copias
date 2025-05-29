# services/extractor.py
import zipfile
import os
import stat
from pathlib import Path
import tempfile
import subprocess

def extract_zip_files(zip_paths, base_dir):
    """Extract ZIP files with proper permission handling"""
    extracted_dirs = []
    
    for zip_path in zip_paths:
        try:
            # Create extraction directory
            zip_name = Path(zip_path).stem
            extract_dir = Path(base_dir) / f"extracted_{zip_name}"
            extract_dir.mkdir(exist_ok=True)
            
            # Extract with proper handling
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract all files
                zip_ref.extractall(str(extract_dir))
                
                # Fix permissions for extracted files
                for root, dirs, files in os.walk(extract_dir):
                    # Fix directory permissions
                    for d in dirs:
                        dir_path = os.path.join(root, d)
                        try:
                            os.chmod(dir_path, 0o755)
                        except (PermissionError, OSError):
                            pass
                    
                    # Fix file permissions
                    for f in files:
                        file_path = os.path.join(root, f)
                        try:
                            os.chmod(file_path, 0o644)
                        except (PermissionError, OSError):
                            pass

                # Call fix_windows_permissions for the extracted directory
                fix_windows_permissions(str(extract_dir))
            
            extracted_dirs.append(str(extract_dir))
            
        except zipfile.BadZipFile:
            raise Exception(f"Archivo ZIP corrupto: {zip_path}")
        except PermissionError as e:
            raise Exception(f"Error de permisos al extraer {zip_path}: {str(e)}")
        except Exception as e:
            raise Exception(f"Error extrayendo {zip_path}: {str(e)}")
    
    return extracted_dirs

def fix_windows_permissions(path):
    """Fix Windows file permissions"""
    try:
        # Use icacls to fix permissions on Windows
        subprocess.run([
            'icacls', path, '/grant', f'{os.getlogin()}:(OI)(CI)F', '/T'
        ], check=False, capture_output=True)
    except Exception:
        pass

def fix_unix_permissions(path):
    """Fix Unix-like system permissions"""
    try:
        os.system(f'chmod -R 755 "{path}"')
    except Exception:
        pass