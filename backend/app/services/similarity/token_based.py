import numpy as np
from difflib import SequenceMatcher
from collections import Counter

class TokenBasedSimilarity:
    """
    Implementa algoritmos de detección de similitud basados en tokens
    """
    def __init__(self):
        self.n_gram_size = 3  # Tamaño de n-gramas para comparación
    
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
    
    def winnowing_similarity(self, tokens1, tokens2, k=5, w=10):
        """
        Implementa el algoritmo Winnowing para detección de similitud
        k: tamaño de k-grama
        w: tamaño de ventana
        """
        # Función hash para k-gramas
        def hash_kgram(kgram):
            return hash(''.join(kgram)) & 0xffffffff
        
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
    
    def calculate_similarity(self, tokens1, tokens2):
        """
        Calcula la similitud combinada entre dos secuencias de tokens
        """
        # Calcular similitud usando diferentes métodos
        lcs_sim = self.longest_common_subsequence(tokens1, tokens2)
        seq_sim = self.sequence_matcher_similarity(tokens1, tokens2)
        ngram_sim = self.n_gram_similarity(tokens1, tokens2)
        
        # Si hay suficientes tokens, usar también Winnowing
        if len(tokens1) >= 5 and len(tokens2) >= 5:
            win_sim = self.winnowing_similarity(tokens1, tokens2)
        else:
            win_sim = 0
        
        # Combinar los resultados (ponderación ajustable)
        combined_sim = 0.25 * lcs_sim + 0.25 * seq_sim + 0.25 * ngram_sim + 0.25 * win_sim
        
        return combined_sim
    
    def find_similar_fragments(self, tokens1, tokens2, threshold=0.7, window_size=20):
        """
        Encuentra fragmentos similares entre dos secuencias de tokens
        """
        similar_fragments = []
        
        # Si alguna secuencia es demasiado corta, comparar directamente
        if len(tokens1) < window_size or len(tokens2) < window_size:
            similarity = self.calculate_similarity(tokens1, tokens2)
            if similarity >= threshold:
                similar_fragments.append({
                    "fragment1": (0, len(tokens1)),
                    "fragment2": (0, len(tokens2)),
                    "similarity": similarity
                })
            return similar_fragments
        
        # Usar ventana deslizante para encontrar fragmentos similares
        for i in range(0, len(tokens1) - window_size + 1, window_size // 2):
            window1 = tokens1[i:i+window_size]
            
            for j in range(0, len(tokens2) - window_size + 1, window_size // 2):
                window2 = tokens2[j:j+window_size]
                
                similarity = self.calculate_similarity(window1, window2)
                
                if similarity >= threshold:
                    similar_fragments.append({
                        "fragment1": (i, i + window_size),
                        "fragment2": (j, j + window_size),
                        "similarity": similarity
                    })
        
        # Fusionar fragmentos superpuestos
        return self._merge_overlapping_fragments(similar_fragments)
    
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
                    "similarity": max(previous["similarity"], current["similarity"])
                }
            else:
                merged.append(current)
        
        return merged