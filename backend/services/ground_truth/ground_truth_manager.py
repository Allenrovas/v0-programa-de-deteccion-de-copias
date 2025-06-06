import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class GroundTruthManager:
    """
    Gestiona el conjunto de datos de ground truth para el detector de copias
    """
    def __init__(self, storage_path: str = "data/ground_truth"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.ground_truth_file = os.path.join(storage_path, "ground_truth.json")
        self.load_ground_truth()
    
    def load_ground_truth(self):
        """Carga el conjunto de datos de ground truth"""
        if os.path.exists(self.ground_truth_file):
            with open(self.ground_truth_file, 'r') as f:
                self.ground_truth = json.load(f)
        else:
            self.ground_truth = {
                "pairs": [],
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }
            self.save_ground_truth()
    
    def save_ground_truth(self):
        """Guarda el conjunto de datos de ground truth"""
        self.ground_truth["metadata"]["last_updated"] = datetime.now().isoformat()
        with open(self.ground_truth_file, 'w') as f:
            json.dump(self.ground_truth, f, indent=2)
    
    def add_ground_truth_pair(self, 
                             file1_path: str, 
                             file2_path: str, 
                             is_plagiarism: bool, 
                             plagiarism_type: str = "unknown",
                             confidence: float = 1.0,
                             fragments: List[Dict] = None,
                             notes: str = ""):
        """
        Añade un par de archivos al conjunto de ground truth
        
        Args:
            file1_path: Ruta al primer archivo
            file2_path: Ruta al segundo archivo
            is_plagiarism: True si es plagio, False si no
            plagiarism_type: Tipo de plagio (exact, renamed, restructured, logic_changed)
            confidence: Nivel de confianza en la clasificación (0-1)
            fragments: Lista de fragmentos específicos que son plagio
            notes: Notas adicionales sobre este par
        """
        # Normalizar rutas
        file1_path = os.path.normpath(file1_path)
        file2_path = os.path.normpath(file2_path)
        
        # Verificar si ya existe
        for pair in self.ground_truth["pairs"]:
            if (pair["file1"] == file1_path and pair["file2"] == file2_path) or \
               (pair["file1"] == file2_path and pair["file2"] == file1_path):
                # Actualizar par existente
                pair["is_plagiarism"] = is_plagiarism
                pair["plagiarism_type"] = plagiarism_type
                pair["confidence"] = confidence
                pair["fragments"] = fragments if fragments else pair.get("fragments", [])
                pair["notes"] = notes
                pair["updated_at"] = datetime.now().isoformat()
                self.save_ground_truth()
                return
        
        # Añadir nuevo par
        self.ground_truth["pairs"].append({
            "file1": file1_path,
            "file2": file2_path,
            "is_plagiarism": is_plagiarism,
            "plagiarism_type": plagiarism_type,
            "confidence": confidence,
            "fragments": fragments if fragments else [],
            "notes": notes,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        })
        self.save_ground_truth()
    
    def get_ground_truth_pairs(self, filter_plagiarism: Optional[bool] = None) -> List[Dict]:
        """
        Obtiene los pares de ground truth, opcionalmente filtrados por estado de plagio
        """
        if filter_plagiarism is None:
            return self.ground_truth["pairs"]
        
        return [pair for pair in self.ground_truth["pairs"] if pair["is_plagiarism"] == filter_plagiarism]
    
    def get_statistics(self) -> Dict:
        """
        Obtiene estadísticas sobre el conjunto de ground truth
        """
        total_pairs = len(self.ground_truth["pairs"])
        plagiarism_pairs = len([p for p in self.ground_truth["pairs"] if p["is_plagiarism"]])
        non_plagiarism_pairs = total_pairs - plagiarism_pairs
        
        plagiarism_types = {}
        for pair in self.ground_truth["pairs"]:
            if pair["is_plagiarism"]:
                ptype = pair["plagiarism_type"]
                plagiarism_types[ptype] = plagiarism_types.get(ptype, 0) + 1
        
        return {
            "total_pairs": total_pairs,
            "plagiarism_pairs": plagiarism_pairs,
            "non_plagiarism_pairs": non_plagiarism_pairs,
            "plagiarism_types": plagiarism_types,
            "last_updated": self.ground_truth["metadata"]["last_updated"]
        }
    
    def export_to_csv(self, output_path: str) -> str:
        """
        Exporta el conjunto de ground truth a un archivo CSV
        """
        df = pd.DataFrame(self.ground_truth["pairs"])
        csv_path = os.path.join(output_path, f"ground_truth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(csv_path, index=False)
        return csv_path
    
    def import_from_csv(self, csv_path: str) -> int:
        """
        Importa datos de ground truth desde un archivo CSV
        Retorna el número de pares importados
        """
        df = pd.read_csv(csv_path)
        count = 0
        
        for _, row in df.iterrows():
            try:
                self.add_ground_truth_pair(
                    file1_path=row["file1"],
                    file2_path=row["file2"],
                    is_plagiarism=bool(row["is_plagiarism"]),
                    plagiarism_type=row["plagiarism_type"],
                    confidence=float(row["confidence"]),
                    fragments=json.loads(row["fragments"]) if isinstance(row["fragments"], str) else [],
                    notes=row["notes"] if "notes" in row else ""
                )
                count += 1
            except Exception as e:
                print(f"Error importando fila: {e}")
        
        return count
