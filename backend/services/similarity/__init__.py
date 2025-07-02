from typing import List, Dict

from .enhanced_processor import UltraFastPlagiarismProcessor as OptimizedPlagiarismProcessor


async def process_submissions(submission_dirs: List[str], language: str, session_id: str) -> Dict:
    """
    Procesa las entregas usando el procesador ultra-optimizado
    """
    processor = OptimizedPlagiarismProcessor()
    return await processor.process_submissions(submission_dirs, language, session_id)

# Exportar función para añadir al ground truth
def add_to_ground_truth(file1_path, file2_path, is_plagiarism, plagiarism_type="unknown", 
                       confidence=1.0, fragments=None, notes=""):
    """
    Añade un par de archivos al conjunto de ground truth
    """
    return plagiarism_processor.add_to_ground_truth(
        file1_path, file2_path, is_plagiarism, plagiarism_type, confidence, fragments, notes
    )

# Exportar función para optimizar umbrales
def optimize_thresholds(results):
    """
    Optimiza los umbrales basándose en el ground truth
    """
    return plagiarism_processor.optimize_thresholds(results)
