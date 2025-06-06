import os
import glob
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from services.tokenizer import get_tokenizer_for_language
from services.similarity.enhanced_token_based import EnhancedTokenBasedSimilarity
from services.similarity.ml_based import MLSimilarityDetector
from services.similarity.adaptive_learning import AdaptivePlagiarismLearner
from services.ground_truth.ground_truth_manager import GroundTruthManager
from config.detector_config import get_config

class PlagiarismProcessor:
    """
    Procesa las entregas para detectar similitudes y posibles plagios
    """
    def __init__(self, ground_truth_path: str = "data/ground_truth", model_path: str = "data/models"):
        self.token_detector = EnhancedTokenBasedSimilarity()
        self.ml_detector = MLSimilarityDetector()
        self.ground_truth_manager = GroundTruthManager(ground_truth_path)
        self.adaptive_learner = AdaptivePlagiarismLearner(model_path)
        
        # Cargar configuración
        config = get_config()
        self.similarity_threshold = config["similarity_threshold"]
        self.fragment_threshold = config["fragment_threshold"]
        self.token_weights = config["similarity_weights"]
        self.token_ml_weights = config["token_ml_weights"]
    
    def set_thresholds(self, similarity_threshold: float = None, fragment_threshold: float = None):
        """
        Establece los umbrales de similitud
        """
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
        
        if fragment_threshold is not None:
            self.fragment_threshold = fragment_threshold
    
    def set_weights(self, token_weights: Dict = None, ml_weights: Dict = None, combined_weights: Dict = None):
        """
        Establece los pesos para diferentes métricas
        """
        if token_weights is not None:
            self.token_weights = token_weights
        
        if combined_weights is not None:
            self.token_ml_weights = combined_weights
    
    def use_optimal_weights(self):
        """
        Usa los pesos óptimos basados en el aprendizaje adaptativo
        """
        optimal_weights = self.adaptive_learner.get_optimal_weights()
        self.token_weights = optimal_weights["token_based"]
        self.token_ml_weights = optimal_weights["combined"]
        
        return optimal_weights
    
    async def process_submissions(self, submission_dirs: List[str], language: str, session_id: str) -> Dict:
        """
        Procesa las entregas para detectar similitudes
        
        Args:
            submission_dirs: Lista de directorios de entregas
            language: Lenguaje de programación
            session_id: ID de la sesión
        
        Returns:
            Resultados del análisis de similitud
        """
        # Obtener el tokenizador adecuado para el lenguaje
        tokenizer = get_tokenizer_for_language(language)
        
        # Recopilar todos los archivos de código
        file_extension = {
            "python": "*.py",
            "java": "*.java",
            "cpp": "*.cpp",
            "javascript": "*.js"
        }.get(language)
        
        all_files = []
        for submission_dir in submission_dirs:
            submission_name = os.path.basename(submission_dir)
            for file_path in glob.glob(f"{submission_dir}/**/{file_extension}", recursive=True):
                relative_path = os.path.relpath(file_path, submission_dir)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        tokens = tokenizer.tokenize(content)
                        all_files.append({
                            "submission": submission_name,
                            "path": relative_path,
                            "full_path": file_path,
                            "content": content,
                            "tokens": tokens
                        })
                except Exception as e:
                    print(f"Error leyendo {file_path}: {str(e)}")
        
        # Calcular similitudes entre archivos
        num_files = len(all_files)
        high_similarity_pairs = []
        
        for i in range(num_files):
            for j in range(i+1, num_files):
                # Evitar comparar archivos de la misma entrega
                if all_files[i]["submission"] == all_files[j]["submission"]:
                    continue
                
                # Calcular similitud basada en tokens
                token_similarity_result = self.token_detector.calculate_similarity(
                    all_files[i]["tokens"], 
                    all_files[j]["tokens"],
                    weights=self.token_weights
                )
                token_similarity = token_similarity_result["combined"]
                
                # Calcular similitud basada en ML (solo si los archivos son suficientemente grandes)
                if len(all_files[i]["content"]) > 100 and len(all_files[j]["content"]) > 100:
                    ml_similarity_results = self.ml_detector.calculate_similarity_matrix(
                        [all_files[i]["content"], all_files[j]["content"]]
                    )
                    ml_similarity = ml_similarity_results["combined"][0][1]
                    ml_details = {
                        "tfidf": float(ml_similarity_results["tfidf"][0][1]),
                        "char_ngram": float(ml_similarity_results["char_ngram"][0][1]),
                        "word_ngram": float(ml_similarity_results["word_ngram"][0][1]),
                        "embedding": float(ml_similarity_results["embedding"][0][1])
                    }
                else:
                    ml_similarity = 0
                    ml_details = {"tfidf": 0, "char_ngram": 0, "word_ngram": 0, "embedding": 0}
                
                # Combinar ambas similitudes usando los pesos configurados
                combined_similarity = (
                    self.token_ml_weights["token"] * token_similarity + 
                    self.token_ml_weights["ml"] * ml_similarity
                )
                
                # Crear diccionario con todas las características para el aprendizaje adaptativo
                all_features = {
                    **token_similarity_result["individual"],
                    "ml_tfidf": ml_details["tfidf"],
                    "ml_char_ngram": ml_details["char_ngram"],
                    "ml_word_ngram": ml_details["word_ngram"],
                    "ml_embedding": ml_details["embedding"]
                }
                
                # Usar el modelo adaptativo para predecir si es plagio
                prediction = self.adaptive_learner.predict_plagiarism(all_features)
                
                # Si la similitud es alta o el modelo predice plagio, analizar en detalle
                if combined_similarity > self.similarity_threshold or prediction["is_plagiarism"]:
                    # Encontrar fragmentos similares
                    similar_fragments = self.token_detector.find_similar_fragments(
                        all_files[i]["tokens"],
                        all_files[j]["tokens"],
                        threshold=self.fragment_threshold
                    )
                    
                    # Reconstruir fragmentos de código original
                    for fragment in similar_fragments:
                        start1, end1 = fragment["fragment1"]
                        start2, end2 = fragment["fragment2"]
                        
                        # Extraer tokens de los fragmentos
                        tokens1 = all_files[i]["tokens"][start1:end1]
                        tokens2 = all_files[j]["tokens"][start2:end2]
                        
                        # Reconstruir código
                        code1 = self.reconstruct_code_fragment(tokens1, all_files[i]["content"], start1, end1)
                        code2 = self.reconstruct_code_fragment(tokens2, all_files[j]["content"], start2, end2)
                        
                        fragment["code1"] = code1
                        fragment["code2"] = code2
                    
                    # Buscar bloques similares usando ML
                    if len(all_files[i]["content"]) > 100 and len(all_files[j]["content"]) > 100:
                        ml_similar_blocks = self.ml_detector.find_similar_code_blocks(
                            all_files[i]["content"],
                            all_files[j]["content"],
                            threshold=self.fragment_threshold
                        )
                        
                        # Convertir bloques ML a formato compatible
                        ml_fragments = []
                        for block in ml_similar_blocks:
                            ml_fragments.append({
                                "type": "ml_block",
                                "block1": block["block1"],
                                "block2": block["block2"],
                                "similarity": block["similarity"],
                                "similarity_details": block["similarity_details"]
                            })
                    else:
                        ml_fragments = []
                    
                    high_similarity_pairs.append({
                        "file1": {
                            "submission": all_files[i]["submission"],
                            "path": all_files[i]["path"],
                            "full_path": all_files[i]["full_path"]
                        },
                        "file2": {
                            "submission": all_files[j]["submission"],
                            "path": all_files[j]["path"],
                            "full_path": all_files[j]["full_path"]
                        },
                        "token_similarity": float(token_similarity),
                        "token_similarity_details": token_similarity_result["individual"],
                        "ml_similarity": float(ml_similarity),
                        "ml_similarity_details": ml_details,
                        "combined_similarity": float(combined_similarity),
                        "similar_fragments": similar_fragments,
                        "ml_fragments": ml_fragments,
                        "is_plagiarism": prediction["is_plagiarism"],
                        "plagiarism_probability": prediction["probability"],
                        "all_features": all_features
                    })
        
        # Ordenar por similitud combinada descendente
        high_similarity_pairs.sort(key=lambda x: x["combined_similarity"], reverse=True)
        
        # Realizar clustering de todas las entregas
        if len(all_files) >= 3:
            # Preparar datos para clustering
            all_contents = [f["content"] for f in all_files]
            metadata = [{"submission": f["submission"], "path": f["path"]} for f in all_files]
            
            # Realizar clustering
            clustering_results = self.ml_detector.cluster_submissions(all_contents)
            
            # Añadir metadatos a los clusters
            for cluster_id, indices in clustering_results["clusters"].items():
                clustering_results["clusters"][cluster_id] = [
                    {"index": idx, "metadata": metadata[idx]} for idx in indices
                ]
        else:
            clustering_results = {"error": "Se necesitan al menos 3 archivos para clustering"}
        
        # Agrupar por entregas
        submission_similarities = {}
        for pair in high_similarity_pairs:
            sub1 = pair["file1"]["submission"]
            sub2 = pair["file2"]["submission"]
            
            key = f"{sub1} - {sub2}" if sub1 < sub2 else f"{sub2} - {sub1}"
            
            if key not in submission_similarities:
                submission_similarities[key] = {
                    "submission1": sub1,
                    "submission2": sub2,
                    "max_similarity": pair["combined_similarity"],
                    "is_plagiarism": pair["is_plagiarism"],
                    "plagiarism_probability": pair["plagiarism_probability"],
                    "similar_files": []
                }
            
            submission_similarities[key]["similar_files"].append({
                "file1": pair["file1"]["path"],
                "file2": pair["file2"]["path"],
                "token_similarity": pair["token_similarity"],
                "ml_similarity": pair["ml_similarity"],
                "combined_similarity": pair["combined_similarity"],
                "is_plagiarism": pair["is_plagiarism"],
                "plagiarism_probability": pair["plagiarism_probability"],
                "fragments": [
                    {
                        "fragment1_start": frag["fragment1"][0],
                        "fragment1_end": frag["fragment1"][1],
                        "fragment2_start": frag["fragment2"][0],
                        "fragment2_end": frag["fragment2"][1],
                        "similarity": frag["similarity"],
                        "code1": frag.get("code1", ""),
                        "code2": frag.get("code2", "")
                    }
                    for frag in pair["similar_fragments"]
                ],
                "ml_fragments": pair["ml_fragments"]
            })
        
        # Convertir a lista y ordenar por similitud máxima
        result = list(submission_similarities.values())
        result.sort(key=lambda x: x["max_similarity"], reverse=True)
        
        # Verificar si hay pares en el ground truth
        for pair in high_similarity_pairs:
            file1 = pair["file1"]["full_path"]
            file2 = pair["file2"]["full_path"]
            
            # Buscar en ground truth
            gt_pairs = self.ground_truth_manager.get_ground_truth_pairs()
            for gt_pair in gt_pairs:
                if (gt_pair["file1"] == file1 and gt_pair["file2"] == file2) or \
                   (gt_pair["file1"] == file2 and gt_pair["file2"] == file1):
                    pair["ground_truth"] = {
                        "is_plagiarism": gt_pair["is_plagiarism"],
                        "plagiarism_type": gt_pair["plagiarism_type"],
                        "confidence": gt_pair["confidence"]
                    }
                    
                    # Añadir al aprendizaje adaptativo
                    self.adaptive_learner.add_training_example(
                        pair["all_features"],
                        gt_pair["is_plagiarism"]
                    )
                    break
        
        return {
            "session_id": session_id,
            "language": language,
            "num_submissions": len(submission_dirs),
            "num_files_analyzed": len(all_files),
            "similarity_threshold": self.similarity_threshold,
            "fragment_threshold": self.fragment_threshold,
            "similarity_results": result[:20],  # Top 20 resultados
            "detailed_pairs": high_similarity_pairs[:50],  # Top 50 pares para análisis detallado
            "clustering": clustering_results,
            "weights": {
                "token": self.token_weights,
                "combined": self.token_ml_weights
            }
        }
    
    def reconstruct_code_fragment(self, tokens, original_code, start_idx, end_idx):
        """
        Reconstruye un fragmento de código a partir de tokens
        """
        if not tokens:
            return ""
        
        # Método simple: buscar líneas que contengan los tokens
        lines = original_code.split('\n')
        token_str = ' '.join(str(t) for t in tokens)
        
        # Intentar encontrar las líneas que contienen estos tokens
        # Esta es una aproximación y podría mejorarse
        relevant_lines = []
        in_fragment = False
        line_start = -1
        line_end = -1
        
        # Buscar líneas que podrían contener los tokens
        for i, line in enumerate(lines):
            contains_tokens = any(str(token) in line for token in tokens[:5]) or \
                             any(str(token) in line for token in tokens[-5:])
            
            if contains_tokens:
                if line_start == -1:
                    line_start = i
                line_end = i
        
        # Si encontramos líneas, extraerlas
        if line_start >= 0 and line_end >= line_start:
            # Añadir un poco de contexto
            line_start = max(0, line_start - 1)
            line_end = min(len(lines) - 1, line_end + 1)
            return '\n'.join(lines[line_start:line_end+1])
        
        # Si no podemos reconstruir, devolver los tokens
        return token_str
    
    def add_to_ground_truth(self, file1_path: str, file2_path: str, is_plagiarism: bool, 
                           plagiarism_type: str = "unknown", confidence: float = 1.0,
                           fragments: List[Dict] = None, notes: str = "", features: Dict = None):
        """
        Añade un par de archivos al conjunto de ground truth
        
        Args:
            file1_path: Ruta al primer archivo
            file2_path: Ruta al segundo archivo
            is_plagiarism: True si es plagio, False si no
            plagiarism_type: Tipo de plagio
            confidence: Nivel de confianza
            fragments: Fragmentos similares
            notes: Notas adicionales
            features: Características de similitud para aprendizaje adaptativo
        """
        self.ground_truth_manager.add_ground_truth_pair(
            file1_path=file1_path,
            file2_path=file2_path,
            is_plagiarism=is_plagiarism,
            plagiarism_type=plagiarism_type,
            confidence=confidence,
            fragments=fragments,
            notes=notes
        )
        
        # Añadir al aprendizaje adaptativo si hay características
        if features:
            self.adaptive_learner.add_training_example(features, is_plagiarism)
        
        return {"status": "success", "message": "Par añadido al ground truth"}
    
    def optimize_thresholds(self, results: List[Dict]) -> Dict:
        """
        Optimiza los umbrales basándose en el ground truth
        """
        from services.evaluation.evaluator import PlagiarismDetectorEvaluator
        
        evaluator = PlagiarismDetectorEvaluator(self.ground_truth_manager)
        optimal_threshold = evaluator.find_optimal_threshold(results)
        
        return {
            "optimal_threshold": optimal_threshold,
            "previous_threshold": self.similarity_threshold,
            "recommendation": f"Se recomienda usar un umbral de {optimal_threshold:.2f} para la detección de plagio"
        }
    
    def get_adaptive_learning_stats(self):
        """
        Obtiene estadísticas del aprendizaje adaptativo
        """
        return {
            "model_trained": hasattr(self.adaptive_learner.model, 'classes_'),
            "optimal_weights": self.adaptive_learner.get_optimal_weights(),
            "weight_history_plot": self.adaptive_learner.get_weight_history_plot()
        }
