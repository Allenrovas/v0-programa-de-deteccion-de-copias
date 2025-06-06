from fastapi import APIRouter, HTTPException, Body
from typing import List, Dict, Optional
from pydantic import BaseModel
from services.similarity.processor import PlagiarismProcessor

router = APIRouter(prefix="/api/adaptive-learning", tags=["adaptive-learning"])

# Instancia del procesador
processor = PlagiarismProcessor()

class TrainingExample(BaseModel):
    features: Dict[str, float]
    is_plagiarism: bool

class WeightUpdate(BaseModel):
    token_weights: Optional[Dict[str, float]] = None
    ml_weights: Optional[Dict[str, float]] = None
    combined_weights: Optional[Dict[str, float]] = None

@router.post("/add-training-example")
async def add_training_example(example: TrainingExample):
    """Añade un ejemplo de entrenamiento al sistema adaptativo"""
    try:
        processor.adaptive_learner.add_training_example(
            example.features,
            example.is_plagiarism
        )
        return {"status": "success", "message": "Ejemplo añadido correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error añadiendo ejemplo: {str(e)}")

@router.post("/train-model")
async def train_model():
    """Entrena el modelo adaptativo con los datos disponibles"""
    try:
        result = processor.adaptive_learner.train_model()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error entrenando modelo: {str(e)}")

@router.get("/optimal-weights")
async def get_optimal_weights():
    """Obtiene los pesos óptimos basados en el aprendizaje adaptativo"""
    try:
        weights = processor.adaptive_learner.get_optimal_weights()
        return weights
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo pesos: {str(e)}")

@router.post("/use-optimal-weights")
async def use_optimal_weights():
    """Aplica los pesos óptimos al procesador"""
    try:
        weights = processor.use_optimal_weights()
        return {"status": "success", "weights": weights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error aplicando pesos: {str(e)}")

@router.post("/update-weights")
async def update_weights(weights: WeightUpdate):
    """Actualiza manualmente los pesos del procesador"""
    try:
        processor.set_weights(
            token_weights=weights.token_weights,
            combined_weights=weights.combined_weights
        )
        return {"status": "success", "message": "Pesos actualizados correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error actualizando pesos: {str(e)}")

@router.get("/stats")
async def get_adaptive_learning_stats():
    """Obtiene estadísticas del aprendizaje adaptativo"""
    try:
        stats = processor.get_adaptive_learning_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")

@router.post("/predict")
async def predict_plagiarism(features: Dict[str, float] = Body(...)):
    """Predice si un par de archivos es plagio basado en sus características"""
    try:
        prediction = processor.adaptive_learner.predict_plagiarism(features)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@router.get("/weight-history")
async def get_weight_history():
    """Obtiene el historial de evolución de pesos"""
    try:
        plot = processor.adaptive_learner.get_weight_history_plot()
        return {"plot": plot}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo historial: {str(e)}")
