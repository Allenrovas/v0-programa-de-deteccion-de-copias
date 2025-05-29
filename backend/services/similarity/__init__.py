import os
import glob
from services.tokenizer import get_tokenizer_for_language
from services.similarity.ml_based import MLSimilarityDetector
from services.similarity.token_based import TokenBasedSimilarity
import pandas as pd
import numpy as np

async def process_submissions(submission_dirs, language, session_id):
    """
    Procesa las entregas para detectar similitudes
    """
    # Obtener el tokenizador adecuado para el lenguaje
    tokenizer = get_tokenizer_for_language(language)
    
    # Inicializar detectores de similitudes
    ml_detector = MLSimilarityDetector()
    token_detector = TokenBasedSimilarity()
    
    # Recopilar todos los archivos de código, javascript puede ser framework o librería
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
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                try:
                    content = f.read()
                    tokens = tokenizer.tokenize(content)
                    all_files.append({
                        "submission": submission_name,
                        "path": relative_path,
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
            token_similarity = token_detector.calculate_similarity(
                all_files[i]["tokens"], 
                all_files[j]["tokens"]
            )
            
            # Calcular similitud basada en ML (solo si los archivos son suficientemente grandes)
            if len(all_files[i]["content"]) > 100 and len(all_files[j]["content"]) > 100:
                ml_similarity_matrix = ml_detector.calculate_similarity_matrix(
                    [all_files[i]["content"], all_files[j]["content"]]
                )
                ml_similarity = ml_similarity_matrix[0][1]
            else:
                ml_similarity = 0
            
            # Combinar ambas similitudes (ponderación ajustable)
            combined_similarity = 0.6 * token_similarity + 0.4 * ml_similarity
            
            if combined_similarity > 0.6:  # Umbral de similitud
                # Encontrar fragmentos similares
                similar_fragments = token_detector.find_similar_fragments(
                    all_files[i]["tokens"],
                    all_files[j]["tokens"],
                    threshold=0.7
                )
                
                high_similarity_pairs.append({
                    "file1": {
                        "submission": all_files[i]["submission"],
                        "path": all_files[i]["path"]
                    },
                    "file2": {
                        "submission": all_files[j]["submission"],
                        "path": all_files[j]["path"]
                    },
                    "token_similarity": float(token_similarity),
                    "ml_similarity": float(ml_similarity),
                    "combined_similarity": float(combined_similarity),
                    "similar_fragments": similar_fragments
                })
    
    # Ordenar por similitud combinada descendente
    high_similarity_pairs.sort(key=lambda x: x["combined_similarity"], reverse=True)
    
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
                "similar_files": []
            }
        
        submission_similarities[key]["similar_files"].append({
            "file1": pair["file1"]["path"],
            "file2": pair["file2"]["path"],
            "token_similarity": pair["token_similarity"],
            "ml_similarity": pair["ml_similarity"],
            "combined_similarity": pair["combined_similarity"],
            "fragments": [
                {
                    "fragment1_start": frag["fragment1"][0],
                    "fragment1_end": frag["fragment1"][1],
                    "fragment2_start": frag["fragment2"][0],
                    "fragment2_end": frag["fragment2"][1],
                    "similarity": frag["similarity"]
                }
                for frag in pair["similar_fragments"]
            ]
        })
    
    # Convertir a lista y ordenar por similitud máxima
    result = list(submission_similarities.values())
    result.sort(key=lambda x: x["max_similarity"], reverse=True)
    
    return {
        "session_id": session_id,
        "language": language,
        "num_submissions": len(submission_dirs),
        "num_files_analyzed": len(all_files),
        "similarity_results": result[:10]  # Top 10 resultados
    }