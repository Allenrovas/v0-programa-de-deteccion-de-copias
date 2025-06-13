import os
import glob
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import pickle
from pathlib import Path

from services.tokenizer import get_tokenizer_for_language
from services.similarity.enhanced_token_based import EnhancedTokenBasedSimilarity
from services.similarity.ml_based import MLSimilarityDetector
from services.similarity.adaptive_learning import AdaptivePlagiarismLearner
from services.ground_truth.ground_truth_manager import GroundTruthManager
from config.detector_config import get_config

@dataclass
class FileInfo:
    """Información optimizada de archivo"""
    submission: str
    path: str
    full_path: str
    content: str
    tokens: List
    content_hash: str
    line_map: List[int]  # Mapeo de token a línea
    char_map: List[int]  # Mapeo de token a posición de caracter

class FragmentReconstructor:
    """Clase especializada para reconstruir fragmentos de código de forma precisa"""
    
    @staticmethod
    def create_token_mapping(content: str, tokens: List) -> Tuple[List[int], List[int]]:
        """
        Crea mapeo preciso de tokens a líneas y caracteres
        """
        lines = content.split('\n')
        line_map = []
        char_map = []
        
        current_pos = 0
        current_line = 0
        
        for token in tokens:
            token_str = str(token)
            
            # Buscar el token en el contenido desde la posición actual
            while current_pos < len(content):
                if content[current_pos:current_pos + len(token_str)] == token_str:
                    # Encontrado el token
                    char_map.append(current_pos)
                    
                    # Calcular la línea
                    while current_line < len(lines) and \
                          sum(len(line) + 1 for line in lines[:current_line + 1]) <= current_pos:
                        current_line += 1
                    
                    line_map.append(current_line)
                    current_pos += len(token_str)
                    break
                
                current_pos += 1
                if content[current_pos - 1] == '\n':
                    current_line += 1
        
        return line_map, char_map
    
    @staticmethod
    def reconstruct_fragment(tokens: List, original_content: str, 
                           start_idx: int, end_idx: int, 
                           line_map: List[int], char_map: List[int]) -> Dict:
        """
        Reconstruye un fragmento de código de forma precisa
        """
        if not tokens or start_idx >= len(tokens) or end_idx > len(tokens):
            return {"code": "", "start_line": 0, "end_line": 0, "start_char": 0, "end_char": 0}
        
        # Obtener posiciones exactas
        start_line = line_map[start_idx] if start_idx < len(line_map) else 0
        end_line = line_map[end_idx - 1] if end_idx - 1 < len(line_map) else start_line
        start_char = char_map[start_idx] if start_idx < len(char_map) else 0
        end_char = char_map[end_idx - 1] if end_idx - 1 < len(char_map) else start_char
        
        # Extraer el fragmento exacto
        lines = original_content.split('\n')
        
        # Añadir contexto (1 línea antes y después)
        context_start = max(0, start_line - 1)
        context_end = min(len(lines), end_line + 2)
        
        fragment_lines = lines[context_start:context_end]
        code_fragment = '\n'.join(fragment_lines)
        
        # Marcar las líneas exactas del fragmento
        marked_lines = []
        for i, line in enumerate(fragment_lines):
            line_num = context_start + i
            if start_line <= line_num <= end_line:
                marked_lines.append(f">>> {line}")  # Marcar líneas del fragmento
            else:
                marked_lines.append(f"    {line}")  # Líneas de contexto
        
        return {
            "code": '\n'.join(marked_lines),
            "raw_code": code_fragment,
            "start_line": start_line + 1,  # 1-indexed para el usuario
            "end_line": end_line + 1,
            "start_char": start_char,
            "end_char": end_char,
            "context_start": context_start + 1,
            "context_end": context_end
        }

class OptimizedPlagiarismProcessor:
    """
    Procesador optimizado para detectar similitudes y posibles plagios
    """
    def __init__(self, ground_truth_path: str = "data/ground_truth", 
                 model_path: str = "data/models", cache_dir: str = "data/cache"):
        self.token_detector = EnhancedTokenBasedSimilarity()
        self.ml_detector = MLSimilarityDetector()
        self.ground_truth_manager = GroundTruthManager(ground_truth_path)
        self.adaptive_learner = AdaptivePlagiarismLearner(model_path)
        self.fragment_reconstructor = FragmentReconstructor()
        
        # Cache para optimizar velocidad
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Configuración
        config = get_config()
        self.similarity_threshold = config["similarity_threshold"]
        self.fragment_threshold = config["fragment_threshold"]
        self.token_weights = config["similarity_weights"]
        self.token_ml_weights = config["token_ml_weights"]
        
        # Configuración de paralelización
        self.max_workers = min(32, (os.cpu_count() or 1) + 4)
        self.chunk_size = 100  # Tamaño de chunk para procesamiento en lotes
    
    def _get_file_hash(self, file_path: str) -> str:
        """Genera hash del archivo para cache"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _cache_file_analysis(self, file_path: str, analysis: Dict) -> None:
        """Guarda análisis en cache"""
        cache_file = self.cache_dir / f"{self._get_file_hash(file_path)}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(analysis, f)
    
    def _load_cached_analysis(self, file_path: str) -> Optional[Dict]:
        """Carga análisis desde cache"""
        try:
            cache_file = self.cache_dir / f"{self._get_file_hash(file_path)}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            pass
        return None
    
    def _process_single_file(self, file_path: str, submission_name: str, 
                           relative_path: str, tokenizer) -> Optional[FileInfo]:
        """
        Procesa un solo archivo de forma optimizada
        """
        try:
            # Verificar cache primero
            cached = self._load_cached_analysis(file_path)
            if cached:
                return FileInfo(**cached)
            
            # Leer archivo
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Filtrar archivos muy pequeños o muy grandes
            if len(content) < 50 or len(content) > 1000000:  # 1MB límite
                return None
            
            # Tokenizar
            tokens = tokenizer.tokenize(content)
            if len(tokens) < 10:  # Muy pocos tokens
                return None
            
            # Crear mapeos precisos
            line_map, char_map = self.fragment_reconstructor.create_token_mapping(content, tokens)
            
            file_info = FileInfo(
                submission=submission_name,
                path=relative_path,
                full_path=file_path,
                content=content,
                tokens=tokens,
                content_hash=self._get_file_hash(file_path),
                line_map=line_map,
                char_map=char_map
            )
            
            # Guardar en cache
            cache_data = {
                'submission': submission_name,
                'path': relative_path,
                'full_path': file_path,
                'content': content,
                'tokens': tokens,
                'content_hash': file_info.content_hash,
                'line_map': line_map,
                'char_map': char_map
            }
            self._cache_file_analysis(file_path, cache_data)
            
            return file_info
            
        except Exception as e:
            print(f"Error procesando {file_path}: {str(e)}")
            return None
    
    async def _collect_files_parallel(self, submission_dirs: List[str], 
                                    language: str) -> List[FileInfo]:
        """
        Recopila y procesa archivos en paralelo
        """
        tokenizer = get_tokenizer_for_language(language)
        file_extension = {
            "python": "*.py",
            "java": "*.java", 
            "cpp": "*.cpp",
            "c": "*.c",
            "javascript": "*.js",
            "typescript": "*.ts"
        }.get(language, "*.py")
        
        # Recopilar todas las rutas de archivos
        file_tasks = []
        for submission_dir in submission_dirs:
            submission_name = os.path.basename(submission_dir)
            for file_path in glob.glob(f"{submission_dir}/**/{file_extension}", recursive=True):
                relative_path = os.path.relpath(file_path, submission_dir)
                file_tasks.append((file_path, submission_name, relative_path))
        
        # Procesar archivos en paralelo
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    executor, 
                    self._process_single_file, 
                    file_path, submission_name, relative_path, tokenizer
                )
                for file_path, submission_name, relative_path in file_tasks
            ]
            
            results = await asyncio.gather(*tasks)
        
        # Filtrar resultados válidos
        return [result for result in results if result is not None]
    
    def _calculate_similarity_batch(self, file_pairs: List[Tuple[int, int]], 
                                  all_files: List[FileInfo]) -> List[Dict]:
        """
        Calcula similitudes en lotes para optimizar velocidad
        """
        results = []
        
        for i, j in file_pairs:
            file1, file2 = all_files[i], all_files[j]
            
            # Evitar comparar archivos de la misma entrega
            if file1.submission == file2.submission:
                continue
            
            # Calcular similitud basada en tokens
            token_similarity_result = self.token_detector.calculate_similarity(
                file1.tokens, file2.tokens, weights=self.token_weights
            )
            token_similarity = token_similarity_result["combined"]
            
            # Calcular similitud ML solo para archivos grandes
            if len(file1.content) > 100 and len(file2.content) > 100:
                try:
                    ml_similarity_results = self.ml_detector.calculate_similarity_matrix(
                        [file1.content, file2.content]
                    )
                    ml_similarity = ml_similarity_results["combined"][0][1]
                    ml_details = {
                        "tfidf": float(ml_similarity_results["tfidf"][0][1]),
                        "char_ngram": float(ml_similarity_results["char_ngram"][0][1]),
                        "word_ngram": float(ml_similarity_results["word_ngram"][0][1]),
                        "embedding": float(ml_similarity_results["embedding"][0][1])
                    }
                except Exception:
                    ml_similarity = 0
                    ml_details = {"tfidf": 0, "char_ngram": 0, "word_ngram": 0, "embedding": 0}
            else:
                ml_similarity = 0
                ml_details = {"tfidf": 0, "char_ngram": 0, "word_ngram": 0, "embedding": 0}
            
            # Combinar similitudes
            combined_similarity = (
                self.token_ml_weights["token"] * token_similarity + 
                self.token_ml_weights["ml"] * ml_similarity
            )
            
            # Crear características para aprendizaje adaptativo
            all_features = {
                **token_similarity_result["individual"],
                "ml_tfidf": ml_details["tfidf"],
                "ml_char_ngram": ml_details["char_ngram"],
                "ml_word_ngram": ml_details["word_ngram"],
                "ml_embedding": ml_details["embedding"]
            }
            
            # Predicción con modelo adaptativo
            prediction = self.adaptive_learner.predict_plagiarism(all_features)
            
            results.append({
                "indices": (i, j),
                "token_similarity": token_similarity,
                "token_similarity_result": token_similarity_result,
                "ml_similarity": ml_similarity,
                "ml_details": ml_details,
                "combined_similarity": combined_similarity,
                "prediction": prediction,
                "all_features": all_features
            })
        
        return results
    
    def _find_and_reconstruct_fragments(self, file1: FileInfo, file2: FileInfo, 
                                      similarity_data: Dict) -> List[Dict]:
        """
        Encuentra y reconstruye fragmentos similares de forma precisa
        """
        # Encontrar fragmentos similares con tokens
        similar_fragments = self.token_detector.find_similar_fragments(
            file1.tokens, file2.tokens, threshold=self.fragment_threshold
        )
        
        reconstructed_fragments = []
        for fragment in similar_fragments:
            start1, end1 = fragment["fragment1"]
            start2, end2 = fragment["fragment2"]
            
            # Reconstruir fragmentos de forma precisa
            frag1 = self.fragment_reconstructor.reconstruct_fragment(
                file1.tokens, file1.content, start1, end1, 
                file1.line_map, file1.char_map
            )
            
            frag2 = self.fragment_reconstructor.reconstruct_fragment(
                file2.tokens, file2.content, start2, end2,
                file2.line_map, file2.char_map
            )
            
            reconstructed_fragments.append({
                "similarity": fragment["similarity"],
                "fragment1": {
                    "start_token": start1,
                    "end_token": end1,
                    "start_line": frag1["start_line"],
                    "end_line": frag1["end_line"],
                    "code": frag1["code"],
                    "raw_code": frag1["raw_code"]
                },
                "fragment2": {
                    "start_token": start2,
                    "end_token": end2,
                    "start_line": frag2["start_line"],
                    "end_line": frag2["end_line"],
                    "code": frag2["code"],
                    "raw_code": frag2["raw_code"]
                }
            })
        
        return reconstructed_fragments
    
    async def process_submissions(self, submission_dirs: List[str], 
                                language: str, session_id: str) -> Dict:
        """
        Procesa las entregas para detectar similitudes de forma optimizada
        """
        print(f"Iniciando análisis de {len(submission_dirs)} entregas...")
        
        # Recopilar archivos en paralelo
        print("Recopilando y procesando archivos...")
        all_files = await self._collect_files_parallel(submission_dirs, language)
        print(f"Procesados {len(all_files)} archivos válidos")
        
        if len(all_files) < 2:
            return {
                "error": "Se necesitan al menos 2 archivos para comparar",
                "num_files_found": len(all_files)
            }
        
        # Generar pares de comparación
        num_files = len(all_files)
        file_pairs = [(i, j) for i in range(num_files) for j in range(i+1, num_files)]
        
        print(f"Calculando similitudes para {len(file_pairs)} pares...")
        
        # Procesar similitudes en lotes
        batch_results = []
        for i in range(0, len(file_pairs), self.chunk_size):
            batch = file_pairs[i:i + self.chunk_size]
            batch_result = self._calculate_similarity_batch(batch, all_files)
            batch_results.extend(batch_result)
        
        # Filtrar pares con alta similitud
        high_similarity_pairs = []
        for result in batch_results:
            if (result["combined_similarity"] > self.similarity_threshold or 
                result["prediction"]["is_plagiarism"]):
                
                i, j = result["indices"]
                file1, file2 = all_files[i], all_files[j]
                
                # Encontrar fragmentos similares precisos
                fragments = self._find_and_reconstruct_fragments(file1, file2, result)
                
                # Buscar bloques ML si es necesario
                ml_fragments = []
                if (len(file1.content) > 100 and len(file2.content) > 100 and 
                    result["ml_similarity"] > self.fragment_threshold):
                    try:
                        ml_similar_blocks = self.ml_detector.find_similar_code_blocks(
                            file1.content, file2.content, threshold=self.fragment_threshold
                        )
                        ml_fragments = [{
                            "type": "ml_block",
                            "block1": block["block1"],
                            "block2": block["block2"],
                            "similarity": block["similarity"],
                            "similarity_details": block["similarity_details"]
                        } for block in ml_similar_blocks]
                    except Exception as e:
                        print(f"Error en fragmentos ML: {e}")
                
                high_similarity_pairs.append({
                    "file1": {
                        "submission": file1.submission,
                        "path": file1.path,
                        "full_path": file1.full_path
                    },
                    "file2": {
                        "submission": file2.submission,
                        "path": file2.path,
                        "full_path": file2.full_path
                    },
                    "token_similarity": float(result["token_similarity"]),
                    "token_similarity_details": result["token_similarity_result"]["individual"],
                    "ml_similarity": float(result["ml_similarity"]),
                    "ml_similarity_details": result["ml_details"],
                    "combined_similarity": float(result["combined_similarity"]),
                    "similar_fragments": fragments,
                    "ml_fragments": ml_fragments,
                    "is_plagiarism": result["prediction"]["is_plagiarism"],
                    "plagiarism_probability": result["prediction"]["probability"],
                    "all_features": result["all_features"]
                })
        
        # Ordenar por similitud
        high_similarity_pairs.sort(key=lambda x: x["combined_similarity"], reverse=True)
        
        # Clustering optimizado
        clustering_results = await self._perform_clustering(all_files)
        
        # Agrupar por entregas
        submission_similarities = self._group_by_submissions(high_similarity_pairs)
        
        # Verificar ground truth y entrenar modelo
        await self._process_ground_truth(high_similarity_pairs)
        
        print(f"Análisis completado. Encontrados {len(high_similarity_pairs)} pares sospechosos")
        
        return {
            "session_id": session_id,
            "language": language,
            "num_submissions": len(submission_dirs),
            "num_files_analyzed": len(all_files),
            "similarity_threshold": self.similarity_threshold,
            "fragment_threshold": self.fragment_threshold,
            "similarity_results": list(submission_similarities.values())[:20],
            "detailed_pairs": high_similarity_pairs[:50],
            "clustering": clustering_results,
            "weights": {
                "token": self.token_weights,
                "combined": self.token_ml_weights
            },
            "performance_stats": {
                "total_comparisons": len(file_pairs),
                "high_similarity_pairs": len(high_similarity_pairs),
                "cache_hits": getattr(self, '_cache_hits', 0)
            }
        }
    
    async def _perform_clustering(self, all_files: List[FileInfo]) -> Dict:
        """Realiza clustering optimizado"""
        if len(all_files) < 3:
            return {"error": "Se necesitan al menos 3 archivos para clustering"}
        
        try:
            all_contents = [f.content for f in all_files]
            metadata = [{"submission": f.submission, "path": f.path} for f in all_files]
            
            # Ejecutar clustering en thread separado para no bloquear
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                clustering_results = await loop.run_in_executor(
                    executor, self.ml_detector.cluster_submissions, all_contents
                )
            
            # Añadir metadatos
            for cluster_id, indices in clustering_results["clusters"].items():
                clustering_results["clusters"][cluster_id] = [
                    {"index": idx, "metadata": metadata[idx]} for idx in indices
                ]
            
            return clustering_results
        except Exception as e:
            return {"error": f"Error en clustering: {str(e)}"}
    
    def _group_by_submissions(self, high_similarity_pairs: List[Dict]) -> Dict:
        """Agrupa resultados por entregas"""
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
            
            # Actualizar similitud máxima si es necesario
            if pair["combined_similarity"] > submission_similarities[key]["max_similarity"]:
                submission_similarities[key]["max_similarity"] = pair["combined_similarity"]
                submission_similarities[key]["is_plagiarism"] = pair["is_plagiarism"]
                submission_similarities[key]["plagiarism_probability"] = pair["plagiarism_probability"]
            
            submission_similarities[key]["similar_files"].append({
                "file1": pair["file1"]["path"],
                "file2": pair["file2"]["path"],
                "token_similarity": pair["token_similarity"],
                "ml_similarity": pair["ml_similarity"],
                "combined_similarity": pair["combined_similarity"],
                "is_plagiarism": pair["is_plagiarism"],
                "plagiarism_probability": pair["plagiarism_probability"],
                "fragments": pair["similar_fragments"],
                "ml_fragments": pair["ml_fragments"]
            })
        
        return submission_similarities
    
    async def _process_ground_truth(self, high_similarity_pairs: List[Dict]):
        """Procesa ground truth de forma asíncrona"""
        loop = asyncio.get_event_loop()
        
        def process_gt():
            gt_pairs = self.ground_truth_manager.get_ground_truth_pairs()
            for pair in high_similarity_pairs:
                file1 = pair["file1"]["full_path"]
                file2 = pair["file2"]["full_path"]
                
                for gt_pair in gt_pairs:
                    if ((gt_pair["file1"] == file1 and gt_pair["file2"] == file2) or
                        (gt_pair["file1"] == file2 and gt_pair["file2"] == file1)):
                        
                        pair["ground_truth"] = {
                            "is_plagiarism": gt_pair["is_plagiarism"],
                            "plagiarism_type": gt_pair["plagiarism_type"],
                            "confidence": gt_pair["confidence"]
                        }
                        
                        self.adaptive_learner.add_training_example(
                            pair["all_features"], gt_pair["is_plagiarism"]
                        )
                        break
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            await loop.run_in_executor(executor, process_gt)
    
    # Métodos adicionales conservados del original
    def set_thresholds(self, similarity_threshold: float = None, fragment_threshold: float = None):
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
        if fragment_threshold is not None:
            self.fragment_threshold = fragment_threshold
    
    def set_weights(self, token_weights: Dict = None, ml_weights: Dict = None, combined_weights: Dict = None):
        if token_weights is not None:
            self.token_weights = token_weights
        if combined_weights is not None:
            self.token_ml_weights = combined_weights
    
    def use_optimal_weights(self):
        optimal_weights = self.adaptive_learner.get_optimal_weights()
        self.token_weights = optimal_weights["token_based"]
        self.token_ml_weights = optimal_weights["combined"]
        return optimal_weights
    
    def add_to_ground_truth(self, file1_path: str, file2_path: str, is_plagiarism: bool, 
                           plagiarism_type: str = "unknown", confidence: float = 1.0,
                           fragments: List[Dict] = None, notes: str = "", features: Dict = None):
        self.ground_truth_manager.add_ground_truth_pair(
            file1_path=file1_path, file2_path=file2_path, is_plagiarism=is_plagiarism,
            plagiarism_type=plagiarism_type, confidence=confidence,
            fragments=fragments, notes=notes
        )
        
        if features:
            self.adaptive_learner.add_training_example(features, is_plagiarism)
        
        return {"status": "success", "message": "Par añadido al ground truth"}
    
    def optimize_thresholds(self, results: List[Dict]) -> Dict:
        from services.evaluation.evaluator import PlagiarismDetectorEvaluator
        evaluator = PlagiarismDetectorEvaluator(self.ground_truth_manager)
        optimal_threshold = evaluator.find_optimal_threshold(results)
        
        return {
            "optimal_threshold": optimal_threshold,
            "previous_threshold": self.similarity_threshold,
            "recommendation": f"Se recomienda usar un umbral de {optimal_threshold:.2f}"
        }
    
    def get_adaptive_learning_stats(self):
        return {
            "model_trained": hasattr(self.adaptive_learner.model, 'classes_'),
            "optimal_weights": self.adaptive_learner.get_optimal_weights(),
            "weight_history_plot": self.adaptive_learner.get_weight_history_plot()
        }