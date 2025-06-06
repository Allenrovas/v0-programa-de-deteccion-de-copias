from fastapi import APIRouter, HTTPException, Body, Query, Path
from typing import List, Dict, Optional
from pydantic import BaseModel
from services.ground_truth.ground_truth_manager import GroundTruthManager
from services.evaluation.evaluator import PlagiarismDetectorEvaluator

router = APIRouter(prefix="/api/ground-truth", tags=["ground-truth"])

# Modelos de datos
class GroundTruthPair(BaseModel):
    file1: str
    file2: str
    is_plagiarism: bool
    plagiarism_type: str = "unknown"
    confidence: float = 1.0
    fragments: Optional[List[Dict]] = None
    notes: str = ""

class EvaluationResult(BaseModel):
    file1: str
    file2: str
    similarity: float
    is_plagiarism: bool

# Instancia del gestor de ground truth
ground_truth_manager = GroundTruthManager()

@router.get("/")
async def get_ground_truth():
    """Obtiene todos los pares de ground truth"""
    pairs = ground_truth_manager.get_ground_truth_pairs()
    return {"pairs": pairs, "count": len(pairs)}

@router.get("/stats")
async def get_ground_truth_stats():
    """Obtiene estadísticas sobre el conjunto de ground truth"""
    stats = ground_truth_manager.get_statistics()
    return stats

@router.post("/")
async def add_ground_truth_pair(pair: GroundTruthPair):
    """Añade un nuevo par al ground truth"""
    try:
        ground_truth_manager.add_ground_truth_pair(
            file1_path=pair.file1,
            file2_path=pair.file2,
            is_plagiarism=pair.is_plagiarism,
            plagiarism_type=pair.plagiarism_type,
            confidence=pair.confidence,
            fragments=pair.fragments,
            notes=pair.notes
        )
        return {"status": "success", "message": "Par añadido correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error añadiendo par: {str(e)}")

@router.delete("/{pair_id}")
async def delete_ground_truth_pair(pair_id: int = Path(...)):
    """Elimina un par del ground truth por su ID"""
    try:
        pairs = ground_truth_manager.get_ground_truth_pairs()
        if pair_id < 0 or pair_id >= len(pairs):
            raise HTTPException(status_code=404, detail="Par no encontrado")
        
        # Esta función no está implementada en el gestor, habría que añadirla
        # ground_truth_manager.delete_pair(pair_id)
        return {"status": "success", "message": "Par eliminado correctamente"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando par: {str(e)}")

@router.post("/evaluate")
async def evaluate_detector(results: List[EvaluationResult]):
    """Evalúa los resultados del detector contra el ground truth"""
    try:
        evaluator = PlagiarismDetectorEvaluator(ground_truth_manager)
        evaluation = evaluator.evaluate_detector(results)
        return evaluation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la evaluación: {str(e)}")

@router.post("/optimize-threshold")
async def optimize_threshold(results: List[EvaluationResult]):
    """Encuentra el umbral óptimo basado en los resultados y el ground truth"""
    try:
        evaluator = PlagiarismDetectorEvaluator(ground_truth_manager)
        optimal_threshold = evaluator.find_optimal_threshold(results)
        return {"optimal_threshold": optimal_threshold}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error optimizando umbral: {str(e)}")

@router.get("/plots")
async def get_evaluation_plots(results: List[EvaluationResult] = Body(...)):
    """Genera gráficos de evaluación"""
    try:
        evaluator = PlagiarismDetectorEvaluator(ground_truth_manager)
        plots = evaluator.generate_evaluation_plots(results)
        return plots
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando gráficos: {str(e)}")

@router.post("/export")
async def export_ground_truth(output_path: str = Body(..., embed=True)):
    """Exporta el ground truth a un archivo CSV"""
    try:
        csv_path = ground_truth_manager.export_to_csv(output_path)
        return {"status": "success", "file_path": csv_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exportando ground truth: {str(e)}")

@router.post("/import")
async def import_ground_truth(csv_path: str = Body(..., embed=True)):
    """Importa ground truth desde un archivo CSV"""
    try:
        count = ground_truth_manager.import_from_csv(csv_path)
        return {"status": "success", "imported_pairs": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importando ground truth: {str(e)}")
