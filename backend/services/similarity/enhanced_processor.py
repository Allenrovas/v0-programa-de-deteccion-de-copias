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
import time
import multiprocessing as mp

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
    file_size: int
    token_count: int

class FastFragmentReconstructor:
    """Clase optimizada para reconstruir fragmentos de código de forma precisa y rápida"""
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def create_token_mapping_cached(content_hash: str, content: str, tokens_tuple: tuple) -> Tuple[List[int], List[int]]:
        """Versión cacheada del mapeo de tokens"""
        tokens = list(tokens_tuple)
        return FastFragmentReconstructor.create_token_mapping(content, tokens)
    
    @staticmethod
    def create_token_mapping(content: str, tokens: List) -> Tuple[List[int], List[int]]:
        """Crea mapeo optimizado de tokens a líneas y caracteres"""
        lines = content.split('\n')
        line_lengths = [len(line) + 1 for line in lines]  # +1 para \n
        line_starts = [0]
        for length in line_lengths[:-1]:
            line_starts.append(line_starts[-1] + length)
        
        line_map = []
        char_map = []
        
        current_pos = 0
        token_strings = [str(token) for token in tokens]
        
        for token_str in token_strings:
            # Buscar token de forma más eficiente
            found_pos = content.find(token_str, current_pos)
            if found_pos != -1:
                char_map.append(found_pos)
                # Encontrar línea usando búsqueda binaria
                line_num = 0
                for i, start in enumerate(line_starts):
                    if start <= found_pos:
                        line_num = i
                    else:
                        break
                line_map.append(line_num)
                current_pos = found_pos + len(token_str)
            else:
                # Fallback si no se encuentra
                char_map.append(current_pos)
                line_map.append(len(line_map) - 1 if line_map else 0)
                current_pos += 1
        
        return line_map, char_map
    
    @staticmethod
    def reconstruct_fragment_with_highlights(tokens: List, original_content: str, 
                                           start_idx: int, end_idx: int, 
                                           line_map: List[int], char_map: List[int]) -> Dict:
        """Reconstruye fragmento con información de resaltado para visualización completa"""
        if not tokens or start_idx >= len(tokens) or end_idx > len(tokens):
            return {"code": "", "start_line": 0, "end_line": 0, "highlights": []}
        
        start_line = line_map[start_idx] if start_idx < len(line_map) else 0
        end_line = line_map[end_idx - 1] if end_idx - 1 < len(line_map) else start_line
        start_char = char_map[start_idx] if start_idx < len(char_map) else 0
        end_char = char_map[end_idx - 1] if end_idx - 1 < len(char_map) else start_char
        
        lines = original_content.split('\n')
        
        # Crear información de resaltado para toda la visualización
        highlights = []
        for line_num in range(start_line, end_line + 1):
            if line_num < len(lines):
                line_start_char = sum(len(lines[i]) + 1 for i in range(line_num))
                line_end_char = line_start_char + len(lines[line_num])
                
                # Calcular posiciones de resaltado dentro de la línea
                highlight_start = max(0, start_char - line_start_char) if line_num == start_line else 0
                highlight_end = min(len(lines[line_num]), end_char - line_start_char) if line_num == end_line else len(lines[line_num])
                
                highlights.append({
                    "line": line_num + 1,  # 1-indexed
                    "start_col": highlight_start,
                    "end_col": highlight_end,
                    "content": lines[line_num]
                })
        
        return {
            "start_line": start_line + 1,
            "end_line": end_line + 1,
            "start_char": start_char,
            "end_char": end_char,
            "highlights": highlights,
            "fragment_tokens": tokens[start_idx:end_idx]
        }

class UltraFastPlagiarismProcessor:
    """Procesador ultra-optimizado para detectar similitudes"""
    
    def __init__(self, ground_truth_path: str = "data/ground_truth", 
                 model_path: str = "data/models", cache_dir: str = "data/cache"):
        self.token_detector = EnhancedTokenBasedSimilarity()
        self.ml_detector = MLSimilarityDetector()
        self.ground_truth_manager = GroundTruthManager(ground_truth_path)
        self.adaptive_learner = AdaptivePlagiarismLearner(model_path)
        self.fragment_reconstructor = FastFragmentReconstructor()
        
        # Cache optimizado
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._cache_hits = 0
        
        # Configuración optimizada
        config = get_config()
        self.similarity_threshold = config["similarity_threshold"]
        self.fragment_threshold = config["fragment_threshold"]
        self.token_weights = config["similarity_weights"]
        self.token_ml_weights = config["token_ml_weights"]
        
        # Configuración de paralelización optimizada
        self.max_workers = min(16, (os.cpu_count() or 1) * 2)  # Reducido para evitar overhead
        self.chunk_size = 50  # Optimizado
        self.process_pool_size = min(4, os.cpu_count() or 1)  # Para tareas CPU-intensivas
        
        # Filtros de optimización
        self.min_file_size = 100  # Bytes mínimos
        self.max_file_size = 500000  # 500KB máximo
        self.min_tokens = 20  # Tokens mínimos
        self.max_tokens = 10000  # Tokens máximos para evitar archivos enormes
        
        # Cache en memoria para comparaciones frecuentes
        self._similarity_cache = {}
        self._max_cache_size = 10000
    
    def _get_file_hash(self, file_path: str) -> str:
        """Genera hash optimizado del archivo"""
        try:
            stat = os.stat(file_path)
            # Usar tamaño y tiempo de modificación para hash rápido
            quick_hash = f"{stat.st_size}_{stat.st_mtime}_{file_path}"
            return hashlib.md5(quick_hash.encode()).hexdigest()
        except:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
    
    def _should_skip_file(self, file_path: str) -> bool:
        """Determina si un archivo debe ser omitido por optimización"""
        try:
            stat = os.stat(file_path)
            return (stat.st_size < self.min_file_size or 
                   stat.st_size > self.max_file_size)
        except:
            return True
    
    def _cache_file_analysis(self, file_path: str, analysis: Dict) -> None:
        """Guarda análisis en cache de forma asíncrona"""
        try:
            cache_file = self.cache_dir / f"{self._get_file_hash(file_path)}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(analysis, f, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            pass  # Ignorar errores de cache
    
    def _load_cached_analysis(self, file_path: str) -> Optional[Dict]:
        """Carga análisis desde cache"""
        try:
            cache_file = self.cache_dir / f"{self._get_file_hash(file_path)}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    self._cache_hits += 1
                    return pickle.load(f)
        except:
            pass
        return None
    
    def _process_single_file_optimized(self, args) -> Optional[FileInfo]:
        """Versión optimizada para procesamiento de archivos individuales"""
        file_path, submission_name, relative_path, language = args
        
        try:
            # Verificar si debe omitirse
            if self._should_skip_file(file_path):
                return None
            
            # Verificar cache
            cached = self._load_cached_analysis(file_path)
            if cached:
                return FileInfo(**cached)
            
            # Leer archivo de forma optimizada
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Filtros rápidos
            if len(content) < 50:
                return None
            
            # Tokenizar de forma lazy
            from services.tokenizer import get_tokenizer_for_language
            tokenizer = get_tokenizer_for_language(language)
            tokens = tokenizer.tokenize(content)
            
            if len(tokens) < self.min_tokens or len(tokens) > self.max_tokens:
                return None
            
            # Crear mapeos solo si es necesario
            line_map, char_map = self.fragment_reconstructor.create_token_mapping(content, tokens)
            
            file_info = FileInfo(
                submission=submission_name,
                path=relative_path,
                full_path=file_path,
                content=content,
                tokens=tokens,
                content_hash=self._get_file_hash(file_path),
                line_map=line_map,
                char_map=char_map,
                file_size=len(content),
                token_count=len(tokens)
            )
            
            # Cache asíncrono
            cache_data = {
                'submission': submission_name,
                'path': relative_path,
                'full_path': file_path,
                'content': content,
                'tokens': tokens,
                'content_hash': file_info.content_hash,
                'line_map': line_map,
                'char_map': char_map,
                'file_size': len(content),
                'token_count': len(tokens)
            }
            self._cache_file_analysis(file_path, cache_data)
            
            return file_info
            
        except Exception as e:
            print(f"Error procesando {file_path}: {str(e)}")
            return None
    
    async def _collect_files_ultra_fast(self, submission_dirs: List[str], 
                                       language: str) -> List[FileInfo]:
        """Recopilación ultra-rápida de archivos"""
        file_extension = {
            "python": "*.py",
            "java": "*.java", 
            "cpp": "*.cpp",
            "c": "*.c",
            "javascript": "*.js",
            "typescript": "*.ts"
        }.get(language, "*.py")
        
        # Recopilar rutas de forma más eficiente
        file_tasks = []
        for submission_dir in submission_dirs:
            submission_name = os.path.basename(submission_dir)
            pattern = f"{submission_dir}/**/{file_extension}"
            for file_path in glob.glob(pattern, recursive=True):
                if not self._should_skip_file(file_path):
                    relative_path = os.path.relpath(file_path, submission_dir)
                    file_tasks.append((file_path, submission_name, relative_path, language))
        
        print(f"Procesando {len(file_tasks)} archivos candidatos...")
        
        # Usar ProcessPoolExecutor para CPU-intensive tasks
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=self.process_pool_size) as executor:
            # Procesar en chunks para mejor rendimiento
            results = []
            for i in range(0, len(file_tasks), self.chunk_size):
                chunk = file_tasks[i:i + self.chunk_size]
                chunk_results = await loop.run_in_executor(
                    executor, self._process_file_chunk, chunk
                )
                results.extend(chunk_results)
        
        # Filtrar resultados válidos
        valid_files = [result for result in results if result is not None]
        print(f"Archivos válidos procesados: {len(valid_files)}")
        return valid_files
    
    def _process_file_chunk(self, file_chunk):
        """Procesa un chunk de archivos"""
        results = []
        for file_args in file_chunk:
            result = self._process_single_file_optimized(file_args)
            if result:
                results.append(result)
        return results
    
    def _get_similarity_cache_key(self, hash1: str, hash2: str) -> str:
        """Genera clave de cache para similitudes"""
        return f"{min(hash1, hash2)}_{max(hash1, hash2)}"
    
    def _calculate_similarity_ultra_fast(self, file1: FileInfo, file2: FileInfo) -> Optional[Dict]:
        """Cálculo ultra-rápido de similitud con cache inteligente"""
        # Cache de similitudes
        cache_key = self._get_similarity_cache_key(file1.content_hash, file2.content_hash)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        # Early stopping: archivos muy diferentes en tamaño
        size_ratio = min(file1.file_size, file2.file_size) / max(file1.file_size, file2.file_size)
        if size_ratio < 0.1:  # Muy diferentes en tamaño
            return None
        
        # Early stopping: muy pocos tokens en común
        tokens1_set = set(str(t) for t in file1.tokens[:100])  # Solo primeros 100 tokens
        tokens2_set = set(str(t) for t in file2.tokens[:100])
        jaccard_quick = len(tokens1_set & tokens2_set) / len(tokens1_set | tokens2_set)
        if jaccard_quick < 0.1:  # Muy pocos tokens en común
            return None
        
        # Calcular similitud basada en tokens (optimizada)
        token_similarity_result = self.token_detector.calculate_similarity(
            file1.tokens, file2.tokens, weights=self.token_weights
        )
        token_similarity = token_similarity_result["combined"]
        
        # Early stopping: similitud de tokens muy baja
        if token_similarity < self.similarity_threshold * 0.5:
            return None
        
        # ML similarity solo para casos prometedores
        ml_similarity = 0
        ml_details = {"tfidf": 0, "char_ngram": 0, "word_ngram": 0, "embedding": 0}
        
        if (token_similarity > self.similarity_threshold * 0.7 and 
            file1.file_size > 200 and file2.file_size > 200):
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
                pass
        
        # Combinar similitudes
        combined_similarity = (
            self.token_ml_weights["token"] * token_similarity + 
            self.token_ml_weights["ml"] * ml_similarity
        )
        
        result = {
            "token_similarity": token_similarity,
            "token_similarity_result": token_similarity_result,
            "ml_similarity": ml_similarity,
            "ml_details": ml_details,
            "combined_similarity": combined_similarity,
            "jaccard_quick": jaccard_quick
        }
        
        # Cache result si hay espacio
        if len(self._similarity_cache) < self._max_cache_size:
            self._similarity_cache[cache_key] = result
        
        return result
    
    def _find_and_reconstruct_fragments_optimized(self, file1: FileInfo, file2: FileInfo, 
                                                 similarity_data: Dict) -> List[Dict]:
        """Encuentra y reconstruye fragmentos de forma optimizada para visualización completa"""
        # Solo buscar fragmentos si la similitud es suficientemente alta
        if similarity_data["combined_similarity"] < self.fragment_threshold:
            return []
        
        # Encontrar fragmentos similares con parámetros optimizados
        similar_fragments = self.token_detector.find_similar_fragments(
            file1.tokens, file2.tokens, 
            threshold=self.fragment_threshold,
            window_size=min(30, len(file1.tokens) // 4, len(file2.tokens) // 4),
            stride=15
        )
        
        reconstructed_fragments = []
        for fragment in similar_fragments[:10]:  # Limitar a 10 fragmentos por par
            start1, end1 = fragment["fragment1"]
            start2, end2 = fragment["fragment2"]
            
            # Reconstruir con información de resaltado
            frag1 = self.fragment_reconstructor.reconstruct_fragment_with_highlights(
                file1.tokens, file1.content, start1, end1, 
                file1.line_map, file1.char_map
            )
            
            frag2 = self.fragment_reconstructor.reconstruct_fragment_with_highlights(
                file2.tokens, file2.content, start2, end2,
                file2.line_map, file2.char_map
            )
            
            reconstructed_fragments.append({
                "similarity": fragment["similarity"],
                "fragment1": frag1,
                "fragment2": frag2
            })
        
        return reconstructed_fragments
    
    async def process_submissions(self, submission_dirs: List[str], 
                                language: str, session_id: str) -> Dict:
        """Procesamiento ultra-optimizado de entregas"""
        start_time = time.time()
        print(f"Iniciando análisis ultra-rápido de {len(submission_dirs)} entregas...")
        
        # Recopilar archivos de forma ultra-rápida
        collect_start = time.time()
        all_files = await self._collect_files_ultra_fast(submission_dirs, language)
        collect_time = time.time() - collect_start
        print(f"⚡ Archivos procesados en {collect_time:.2f}s - Cache hits: {self._cache_hits}")
        
        if len(all_files) < 2:
            return {
                "error": "Se necesitan al menos 2 archivos para comparar",
                "num_files_found": len(all_files)
            }
        
        # Generar pares de comparación con filtrado inteligente
        comparison_start = time.time()
        high_similarity_pairs = []
        
        # Comparar archivos de forma optimizada
        total_comparisons = 0
        skipped_comparisons = 0
        
        for i in range(len(all_files)):
            for j in range(i + 1, len(all_files)):
                file1, file2 = all_files[i], all_files[j]
                
                # Evitar comparar archivos de la misma entrega
                if file1.submission == file2.submission:
                    skipped_comparisons += 1
                    continue
                
                total_comparisons += 1
                
                # Calcular similitud de forma ultra-rápida
                similarity_result = self._calculate_similarity_ultra_fast(file1, file2)
                
                if similarity_result is None:
                    continue
                
                # Solo procesar si supera el umbral
                if similarity_result["combined_similarity"] > self.similarity_threshold:
                    # Predicción rápida con modelo adaptativo
                    all_features = {
                        **similarity_result["token_similarity_result"]["individual"],
                        "ml_tfidf": similarity_result["ml_details"]["tfidf"],
                        "ml_char_ngram": similarity_result["ml_details"]["char_ngram"],
                        "ml_word_ngram": similarity_result["ml_details"]["word_ngram"],
                        "ml_embedding": similarity_result["ml_details"]["embedding"]
                    }
                    
                    prediction = self.adaptive_learner.predict_plagiarism(all_features)
                    
                    # Encontrar fragmentos para visualización completa
                    fragments = self._find_and_reconstruct_fragments_optimized(
                        file1, file2, similarity_result
                    )
                    
                    high_similarity_pairs.append({
                        "file1": {
                            "submission": file1.submission,
                            "path": file1.path,
                            "full_path": file1.full_path,
                            "content": file1.content,  # Contenido completo para visualización
                            "file_size": file1.file_size,
                            "token_count": file1.token_count
                        },
                        "file2": {
                            "submission": file2.submission,
                            "path": file2.path,
                            "full_path": file2.full_path,
                            "content": file2.content,  # Contenido completo para visualización
                            "file_size": file2.file_size,
                            "token_count": file2.token_count
                        },
                        "token_similarity": float(similarity_result["token_similarity"]),
                        "token_similarity_details": similarity_result["token_similarity_result"]["individual"],
                        "ml_similarity": float(similarity_result["ml_similarity"]),
                        "ml_similarity_details": similarity_result["ml_details"],
                        "combined_similarity": float(similarity_result["combined_similarity"]),
                        "similar_fragments": fragments,  # Fragmentos con información de resaltado
                        "is_plagiarism": prediction["is_plagiarism"],
                        "plagiarism_probability": prediction["probability"],
                        "all_features": all_features
                    })
        
        comparison_time = time.time() - comparison_start
        print(f"⚡ Comparaciones completadas en {comparison_time:.2f}s")
        print(f"Total: {total_comparisons}, Omitidas: {skipped_comparisons}, Sospechosas: {len(high_similarity_pairs)}")
        
        # Ordenar por similitud
        high_similarity_pairs.sort(key=lambda x: x["combined_similarity"], reverse=True)
        
        # Agrupar por entregas de forma optimizada
        submission_similarities = self._group_by_submissions_fast(high_similarity_pairs)
        
        total_time = time.time() - start_time
        print(f"🎉 Análisis completado en {total_time:.2f}s")
        
        return {
            "session_id": session_id,
            "language": language,
            "num_submissions": len(submission_dirs),
            "num_files_analyzed": len(all_files),
            "similarity_threshold": self.similarity_threshold,
            "fragment_threshold": self.fragment_threshold,
            "similarity_results": list(submission_similarities.values())[:20],
            "detailed_pairs": high_similarity_pairs[:50],
            "weights": {
                "token": self.token_weights,
                "combined": self.token_ml_weights
            },
            "performance_stats": {
                "total_time": total_time,
                "collection_time": collect_time,
                "comparison_time": comparison_time,
                "total_comparisons": total_comparisons,
                "skipped_comparisons": skipped_comparisons,
                "high_similarity_pairs": len(high_similarity_pairs),
                "cache_hits": self._cache_hits,
                "files_processed": len(all_files)
            }
        }
    
    def _group_by_submissions_fast(self, high_similarity_pairs: List[Dict]) -> Dict:
        """Agrupación rápida por entregas"""
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
            
            # Actualizar similitud máxima
            if pair["combined_similarity"] > submission_similarities[key]["max_similarity"]:
                submission_similarities[key]["max_similarity"] = pair["combined_similarity"]
                submission_similarities[key]["is_plagiarism"] = pair["is_plagiarism"]
                submission_similarities[key]["plagiarism_probability"] = pair["plagiarism_probability"]
            
            submission_similarities[key]["similar_files"].append({
                "file1": pair["file1"]["path"],
                "file2": pair["file2"]["path"],
                "file1_content": pair["file1"]["content"],  # Para visualización completa
                "file2_content": pair["file2"]["content"],  # Para visualización completa
                "token_similarity": pair["token_similarity"],
                "ml_similarity": pair["ml_similarity"],
                "combined_similarity": pair["combined_similarity"],
                "is_plagiarism": pair["is_plagiarism"],
                "plagiarism_probability": pair["plagiarism_probability"],
                "fragments": pair["similar_fragments"]  # Con información de resaltado
            })
        
        return submission_similarities

# Alias para compatibilidad
OptimizedPlagiarismProcessor = UltraFastPlagiarismProcessor
