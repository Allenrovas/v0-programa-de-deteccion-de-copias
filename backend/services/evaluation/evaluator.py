import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import io
import base64

class PlagiarismDetectorEvaluator:
    """
    Evalúa el rendimiento del detector de plagio usando ground truth
    """
    def __init__(self, ground_truth_manager):
        self.ground_truth_manager = ground_truth_manager
    
    def evaluate_detector(self, detector_results: List[Dict]) -> Dict:
        """
        Evalúa los resultados del detector contra el ground truth
        
        Args:
            detector_results: Lista de resultados del detector, cada uno con:
                - file1: ruta al primer archivo
                - file2: ruta al segundo archivo
                - similarity: puntuación de similitud (0-1)
                - is_plagiarism: predicción de plagio (True/False)
        
        Returns:
            Diccionario con métricas de evaluación
        """
        # Obtener pares de ground truth
        ground_truth_pairs = self.ground_truth_manager.get_ground_truth_pairs()
        
        # Mapear resultados del detector a pares de ground truth
        y_true = []
        y_pred = []
        y_scores = []
        
        for result in detector_results:
            file1 = result["file1"]
            file2 = result["file2"]
            
            # Buscar en ground truth
            gt_match = None
            for gt_pair in ground_truth_pairs:
                if (gt_pair["file1"] == file1 and gt_pair["file2"] == file2) or \
                   (gt_pair["file1"] == file2 and gt_pair["file2"] == file1):
                    gt_match = gt_pair
                    break
            
            if gt_match:
                y_true.append(1 if gt_match["is_plagiarism"] else 0)
                y_pred.append(1 if result["is_plagiarism"] else 0)
                y_scores.append(result["similarity"])
        
        if not y_true:
            return {"error": "No matching ground truth pairs found"}
        
        # Calcular métricas
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calcular curva ROC si hay suficientes datos
        roc_auc = 0
        roc_curve_data = None
        if len(set(y_true)) > 1:  # Necesitamos tanto positivos como negativos
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            roc_curve_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
            "roc_auc": float(roc_auc),
            "roc_curve": roc_curve_data,
            "evaluated_pairs": len(y_true)
        }
    
    def find_optimal_threshold(self, detector_results: List[Dict]) -> float:
        """
        Encuentra el umbral óptimo para clasificar plagio basado en F1-score
        """
        ground_truth_pairs = self.ground_truth_manager.get_ground_truth_pairs()
        
        # Recopilar pares con ground truth
        pairs_with_gt = []
        for result in detector_results:
            file1 = result["file1"]
            file2 = result["file2"]
            
            # Buscar en ground truth
            for gt_pair in ground_truth_pairs:
                if (gt_pair["file1"] == file1 and gt_pair["file2"] == file2) or \
                   (gt_pair["file1"] == file2 and gt_pair["file2"] == file1):
                    pairs_with_gt.append({
                        "similarity": result["similarity"],
                        "is_plagiarism": gt_pair["is_plagiarism"]
                    })
                    break
        
        if not pairs_with_gt:
            return 0.7  # Valor por defecto si no hay datos
        
        # Probar diferentes umbrales
        thresholds = np.arange(0.1, 1.0, 0.05)
        best_f1 = 0
        best_threshold = 0.7
        
        for threshold in thresholds:
            y_true = [1 if p["is_plagiarism"] else 0 for p in pairs_with_gt]
            y_pred = [1 if p["similarity"] >= threshold else 0 for p in pairs_with_gt]
            
            if len(set(y_pred)) < 2:  # Todos los resultados son iguales
                continue
                
            _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    
    def generate_evaluation_plots(self, detector_results: List[Dict]) -> Dict[str, str]:
        """
        Genera gráficos de evaluación y los devuelve como imágenes base64
        """
        ground_truth_pairs = self.ground_truth_manager.get_ground_truth_pairs()
        
        # Recopilar datos
        similarities = []
        is_plagiarism = []
        
        for result in detector_results:
            file1 = result["file1"]
            file2 = result["file2"]
            
            # Buscar en ground truth
            for gt_pair in ground_truth_pairs:
                if (gt_pair["file1"] == file1 and gt_pair["file2"] == file2) or \
                   (gt_pair["file1"] == file2 and gt_pair["file2"] == file1):
                    similarities.append(result["similarity"])
                    is_plagiarism.append(gt_pair["is_plagiarism"])
                    break
        
        if not similarities:
            return {"error": "No matching ground truth pairs found"}
        
        # Crear gráficos
        plots = {}
        
        # Histograma de similitudes
        plt.figure(figsize=(10, 6))
        plt.hist([
            [s for s, p in zip(similarities, is_plagiarism) if p],
            [s for s, p in zip(similarities, is_plagiarism) if not p]
        ], bins=20, label=['Plagio', 'No Plagio'], alpha=0.7)
        plt.xlabel('Puntuación de Similitud')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Similitudes por Clase')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convertir a base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots["histogram"] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Curva ROC
        if len(set(is_plagiarism)) > 1:  # Necesitamos tanto positivos como negativos
            fpr, tpr, _ = roc_curve(is_plagiarism, similarities)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Curva ROC')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plots["roc_curve"] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        
        return plots
