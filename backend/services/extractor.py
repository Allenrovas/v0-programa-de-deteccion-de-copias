# services/extractor.py
import zipfile
import os
import stat
import platform
from pathlib import Path
import tempfile
import subprocess
import time

def extract_zip_files(zip_paths, base_dir):
    """Extract ZIP files with comprehensive permission handling"""
    extracted_dirs = []

    print(f"Extracting ZIP files to: {base_dir}")  
    for zip_path in zip_paths:
        try:
            # Create extraction directory
            zip_name = Path(zip_path).stem
            extract_dir = Path(base_dir) / f"extracted_{zip_name}"
            
            # Remove existing directory if it exists
            if extract_dir.exists():
                force_remove_directory(str(extract_dir))
            
            extract_dir.mkdir(exist_ok=True)
            
            # Extract with proper handling
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files to extract
                file_list = zip_ref.namelist()
                
                for file_info in zip_ref.infolist():
                    try:
                        # Extract individual file
                        zip_ref.extract(file_info, str(extract_dir))
                        
                        # Get the extracted file path
                        extracted_file_path = extract_dir / file_info.filename
                        
                        # Fix permissions immediately after extraction
                        if extracted_file_path.exists():
                            fix_file_permissions(str(extracted_file_path))
                            
                    except Exception as e:
                        print(f"Warning: Could not extract {file_info.filename}: {e}")
                        continue
                
                # Apply comprehensive permission fix
                fix_directory_permissions_comprehensive(str(extract_dir))
            
            extracted_dirs.append(str(extract_dir))
            
        except zipfile.BadZipFile:
            raise Exception(f"Archivo ZIP corrupto: {zip_path}")
        except PermissionError as e:
            raise Exception(f"Error de permisos al extraer {zip_path}: {str(e)}")
        except Exception as e:
            raise Exception(f"Error extrayendo {zip_path}: {str(e)}")
    
    return extracted_dirs

def fix_file_permissions(file_path):
    """Fix permissions for a single file"""
    try:
        path_obj = Path(file_path)
        
        if path_obj.is_file():
            # Remove read-only attribute
            if platform.system() == "Windows":
                try:
                    # Remove read-only attribute on Windows
                    os.chmod(file_path, stat.S_IWRITE | stat.S_IREAD)
                except:
                    pass
                
                # Use attrib command as backup
                try:
                    subprocess.run(['attrib', '-R', file_path], 
                                 check=False, capture_output=True, timeout=5)
                except:
                    pass
            else:
                os.chmod(file_path, 0o644)
        
        elif path_obj.is_dir():
            if platform.system() == "Windows":
                try:
                    os.chmod(file_path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
                except:
                    pass
            else:
                os.chmod(file_path, 0o755)
                
    except Exception as e:
        print(f"Warning: Could not fix permissions for {file_path}: {e}")

def fix_directory_permissions_comprehensive(directory_path):
    """Comprehensive permission fixing for directory and all contents"""
    try:
        # Method 1: Python os.chmod
        for root, dirs, files in os.walk(directory_path):
            # Fix directory permissions
            try:
                if platform.system() == "Windows":
                    os.chmod(root, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
                else:
                    os.chmod(root, 0o755)
            except:
                pass
            
            # Fix file permissions
            for file in files:
                file_path = os.path.join(root, file)
                fix_file_permissions(file_path)
        
        # Method 2: System commands as backup
        if platform.system() == "Windows":
            fix_windows_permissions_advanced(directory_path)
        else:
            fix_unix_permissions(directory_path)
            
    except Exception as e:
        print(f"Warning: Could not fix directory permissions: {e}")

def fix_windows_permissions_advanced(path):
    """Advanced Windows permission fixing"""
    try:
        # Method 1: icacls
        username = os.getlogin()
        subprocess.run([
            'icacls', path, '/grant', f'{username}:(OI)(CI)F', '/T', '/Q'
        ], check=False, capture_output=True, timeout=30)
        
        # Method 2: attrib to remove read-only
        subprocess.run([
            'attrib', '-R', f'{path}\\*.*', '/S', '/D'
        ], check=False, capture_output=True, timeout=30)
        
        # Method 3: takeown (as last resort)
        subprocess.run([
            'takeown', '/F', path, '/R', '/D', 'Y'
        ], check=False, capture_output=True, timeout=30)
        
    except Exception as e:
        print(f"Warning: Advanced Windows permission fix failed: {e}")

def fix_unix_permissions(path):
    """Fix Unix-like system permissions"""
    try:
        subprocess.run(['chmod', '-R', '755', path], 
                      check=False, capture_output=True, timeout=30)
    except Exception as e:
        print(f"Warning: Unix permission fix failed: {e}")

def force_remove_directory(directory_path):
    """Force remove directory with all contents"""
    try:
        if os.path.exists(directory_path):
            # First try to fix permissions
            fix_directory_permissions_comprehensive(directory_path)
            
            # Wait a moment for file system to update
            time.sleep(0.1)
            
            # Try to remove
            import shutil
            shutil.rmtree(directory_path, onerror=handle_remove_readonly)
            
    except Exception as e:
        print(f"Warning: Could not remove directory {directory_path}: {e}")

def handle_remove_readonly(func, path, exc_info):
    """Error handler for removing read-only files"""
    try:
        # Fix permissions and try again
        fix_file_permissions(path)
        time.sleep(0.1)
        func(path)
    except Exception:
        try:
            # Force delete on Windows
            if platform.system() == "Windows":
                subprocess.run(['del', '/F', '/Q', path], 
                             shell=True, check=False, capture_output=True)
            else:
                subprocess.run(['rm', '-f', path], 
                             check=False, capture_output=True)
        except:
            pass