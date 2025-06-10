from services.similarity.processor import PlagiarismProcessor

# Crear una instancia global del procesador
plagiarism_processor = PlagiarismProcessor()

# Exportar la función de procesamiento
async def process_submissions(submission_dirs, language, session_id):
    """
    Procesa las entregas para detectar similitudes
    """
    return await plagiarism_processor.process_submissions(submission_dirs, language, session_id)

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
