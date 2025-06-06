import numpy as np
from difflib import SequenceMatcher
from collections import Counter
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Any

class EnhancedTokenBasedSimilarity:
    """
    Implementa algoritmos mejorados de detección de similitud basados en tokens
    """
    def __init__(self):
        self.n_gram_size = 3  # Tamaño de n-gramas para comparación
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer='word',
            token_pattern=r'\S+',
            ngram_range=(1, 3)
        )
    
    def longest_common_subsequence(self, tokens1, tokens2):
        """
        Calcula la longitud de la subsecuencia común más larga entre dos secuencias de tokens
        """
        m, n = len(tokens1), len(tokens2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if tokens1[i-1] == tokens2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        # Normalizar por la longitud de la secuencia más larga
        return dp[m][n] / max(m, n) if max(m, n) > 0 else 0
    
    def sequence_matcher_similarity(self, tokens1, tokens2):
        """
        Usa difflib.SequenceMatcher para calcular la similitud entre secuencias de tokens
        """
        matcher = SequenceMatcher(None, tokens1, tokens2)
        return matcher.ratio()
    
    def n_gram_similarity(self, tokens1, tokens2):
        """
        Calcula la similitud basada en n-gramas entre dos secuencias de tokens
        """
        # Generar n-gramas para ambas secuencias
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        # Asegurarse de que hay suficientes tokens para formar n-gramas
        n = min(self.n_gram_size, len(tokens1), len(tokens2))
        if n < 1:
            return 0.0
        
        # Obtener n-gramas
        ngrams1 = get_ngrams(tokens1, n)
        ngrams2 = get_ngrams(tokens2, n)
        
        # Contar frecuencias de n-gramas
        counter1 = Counter(ngrams1)
        counter2 = Counter(ngrams2)
        
        # Calcular intersección de n-gramas
        common_ngrams = sum((counter1 & counter2).values())
        total_ngrams = len(ngrams1) + len(ngrams2)
        
        # Coeficiente de Jaccard: intersección / unión
        return 2 * common_ngrams / total_ngrams if total_ngrams > 0 else 0
    
    def levenshtein_similarity(self, tokens1, tokens2):
        """
        Calcula similitud usando distancia de Levenshtein normalizada
        """
        # Convertir tokens a strings para usar Levenshtein
        str1 = ' '.join(str(t) for t in tokens1)
        str2 = ' '.join(str(t) for t in tokens2)
        
        # Calcular distancia
        distance = Levenshtein.distance(str1, str2)
        max_len = max(len(str1), len(str2))
        
        # Normalizar a similitud (1 - distancia_normalizada)
        similarity = 1 - (distance / max_len) if max_len > 0 else 0
        return similarity
    
    def tfidf_cosine_similarity(self, tokens1, tokens2):
        """
        Calcula similitud del coseno usando TF-IDF
        """
        # Convertir tokens a strings
        str1 = ' '.join(str(t) for t in tokens1)
        str2 = ' '.join(str(t) for t in tokens2)
        
        # Calcular TF-IDF y similitud del coseno
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([str1, str2])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(cosine_sim)
        except:
            return 0.0
    
    def winnowing_similarity(self, tokens1, tokens2, k=5, w=10):
        """
        Implementa el algoritmo Winnowing para detección de similitud
        k: tamaño de k-grama
        w: tamaño de ventana
        """
        # Función hash para k-gramas
        def hash_kgram(kgram):
            return hash(''.join(str(k) for k in kgram)) & 0xffffffff
        
        # Generar fingerprints usando Winnowing
        def get_fingerprints(tokens, k, w):
            if len(tokens) < k:
                return set()
            
            # Generar k-gramas y sus hashes
            kgrams = [tuple(tokens[i:i+k]) for i in range(len(tokens) - k + 1)]
            hashes = [hash_kgram(kg) for kg in kgrams]
            
            # Aplicar Winnowing
            fingerprints = set()
            for i in range(len(hashes) - w + 1):
                window = hashes[i:i+w]
                # Seleccionar el hash mínimo en la ventana
                min_hash = min(window)
                min_pos = i + window.index(min_hash)
                fingerprints.add((min_hash, min_pos))
            
            return fingerprints
        
        # Obtener fingerprints para ambas secuencias
        fp1 = get_fingerprints(tokens1, k, w)
        fp2 = get_fingerprints(tokens2, k, w)
        
        # Calcular similitud como coeficiente de Jaccard de los fingerprints
        if not fp1 or not fp2:
            return 0.0
        
        # Extraer solo los hashes (ignorar posiciones)
        hash_fp1 = {h for h, _ in fp1}
        hash_fp2 = {h for h, _ in fp2}
        
        # Calcular coeficiente de Jaccard
        intersection = len(hash_fp1.intersection(hash_fp2))
        union = len(hash_fp1.union(hash_fp2))
        
        return intersection / union if union > 0 else 0
    
    def calculate_similarity(self, tokens1, tokens2, weights=None):
        """
        Calcula la similitud combinada entre dos secuencias de tokens
        
        Args:
            tokens1: Primera secuencia de tokens
            tokens2: Segunda secuencia de tokens
            weights: Diccionario con pesos para cada método de similitud
        """
        # Pesos por defecto
        if weights is None:
            weights = {
                "lcs": 0.15,
                "sequence_matcher": 0.15,
                "ngram": 0.2,
                "levenshtein": 0.2,
                "tfidf_cosine": 0.2,
                "winnowing": 0.1
            }
        
        # Calcular similitud usando diferentes métodos
        similarities = {
            "lcs": self.longest_common_subsequence(tokens1, tokens2),
            "sequence_matcher": self.sequence_matcher_similarity(tokens1, tokens2),
            "ngram": self.n_gram_similarity(tokens1, tokens2),
            "levenshtein": self.levenshtein_similarity(tokens1, tokens2),
            "tfidf_cosine": self.tfidf_cosine_similarity(tokens1, tokens2)
        }
        
        # Si hay suficientes tokens, usar también Winnowing
        if len(tokens1) >= 5 and len(tokens2) >= 5:
            similarities["winnowing"] = self.winnowing_similarity(tokens1, tokens2)
        else:
            similarities["winnowing"] = 0
            # Redistribuir el peso
            if "winnowing" in weights and weights["winnowing"] > 0:
                remaining_weight = weights["winnowing"] / len([k for k in weights if k != "winnowing"])
                for k in weights:
                    if k != "winnowing":
                        weights[k] += remaining_weight
                weights["winnowing"] = 0
        
        # Combinar los resultados usando los pesos
        combined_sim = sum(similarities[method] * weights[method] for method in similarities)
        
        return {
            "combined": combined_sim,
            "individual": similarities
        }
    
    def find_similar_fragments(self, tokens1, tokens2, threshold=0.7, window_size=20, stride=10):
        """
        Encuentra fragmentos similares entre dos secuencias de tokens
        
        Args:
            tokens1: Primera secuencia de tokens
            tokens2: Segunda secuencia de tokens
            threshold: Umbral de similitud para considerar un fragmento como similar
            window_size: Tamaño de la ventana deslizante
            stride: Paso entre ventanas consecutivas
        
        Returns:
            Lista de fragmentos similares con sus índices y puntuaciones
        """
        similar_fragments = []
        
        # Si alguna secuencia es demasiado corta, comparar directamente
        if len(tokens1) < window_size or len(tokens2) < window_size:
            similarity_result = self.calculate_similarity(tokens1, tokens2)
            similarity = similarity_result["combined"]
            
            if similarity >= threshold:
                similar_fragments.append({
                    "fragment1": (0, len(tokens1)),
                    "fragment2": (0, len(tokens2)),
                    "similarity": similarity,
                    "tokens1": tokens1,
                    "tokens2": tokens2,
                    "similarity_details": similarity_result["individual"]
                })
            return similar_fragments
        
        # Usar ventana deslizante para encontrar fragmentos similares
        for i in range(0, len(tokens1) - window_size + 1, stride):
            window1 = tokens1[i:i+window_size]
            
            for j in range(0, len(tokens2) - window_size + 1, stride):
                window2 = tokens2[j:j+window_size]
                
                similarity_result = self.calculate_similarity(window1, window2)
                similarity = similarity_result["combined"]
                
                if similarity >= threshold:
                    similar_fragments.append({
                        "fragment1": (i, i + window_size),
                        "fragment2": (j, j + window_size),
                        "similarity": similarity,
                        "tokens1": window1,
                        "tokens2": window2,
                        "similarity_details": similarity_result["individual"]
                    })
        
        # Fusionar fragmentos superpuestos
        return self._merge_overlapping_fragments(similar_fragments)
    
    def find_exact_matches(self, tokens1, tokens2, min_length=5):
        """
        Encuentra coincidencias exactas de secuencias de tokens
        """
        matches = []
        matcher = SequenceMatcher(None, tokens1, tokens2)
        
        for match in matcher.get_matching_blocks():
            if match.size >= min_length:
                matches.append({
                    "fragment1": (match.a, match.a + match.size),
                    "fragment2": (match.b, match.b + match.size),
                    "length": match.size,
                    "tokens": tokens1[match.a:match.a + match.size]
                })
        
        return matches
    
    def _merge_overlapping_fragments(self, fragments):
        """
        Fusiona fragmentos superpuestos para evitar duplicados
        """
        if not fragments:
            return []
        
        # Ordenar por posición del primer fragmento
        sorted_fragments = sorted(fragments, key=lambda x: (x["fragment1"][0], x["fragment2"][0]))
        
        merged = [sorted_fragments[0]]
        
        for current in sorted_fragments[1:]:
            previous = merged[-1]
            
            # Verificar si hay superposición
            if (current["fragment1"][0] <= previous["fragment1"][1] and
                current["fragment2"][0] <= previous["fragment2"][1]):
                # Fusionar fragmentos
                merged[-1] = {
                    "fragment1": (
                        previous["fragment1"][0],
                        max(previous["fragment1"][1], current["fragment1"][1])
                    ),
                    "fragment2": (
                        previous["fragment2"][0],
                        max(previous["fragment2"][1], current["fragment2"][1])
                    ),
                    "similarity": max(previous["similarity"], current["similarity"]),
                    "tokens1": previous["tokens1"],  # Mantener los tokens del fragmento anterior
                    "tokens2": previous["tokens2"],
                    "similarity_details": previous.get("similarity_details", {})
                }
            else:
                merged.append(current)
        
        return merged
    
    def reconstruct_code_from_tokens(self, tokens, original_code):
        """
        Reconstruye el código original a partir de tokens y código fuente
        
        Args:
            tokens: Lista de tokens
            original_code: Código fuente original
        
        Returns:
            Fragmento de código reconstruido
        """
        # Esta es una aproximación simple, podría mejorarse
        if not tokens:
            return ""
        
        # Unir tokens con espacios
        reconstructed = ' '.join(str(t) for t in tokens)
        
        # Intentar encontrar en el código original
        token_str = str(tokens[0])
        start_pos = original_code.find(token_str)
        
        if start_pos >= 0:
            # Buscar el último token
            last_token = str(tokens[-1])
            end_pos = original_code.find(last_token, start_pos) + len(last_token)
            
            if end_pos > start_pos:
                return original_code[start_pos:end_pos]
        
        return reconstructed
