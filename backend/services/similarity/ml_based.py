import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModel
import re
import io
import base64
from typing import List, Dict, Tuple, Any, Optional
import umap
import hdbscan
from collections import defaultdict

class MLSimilarityDetector:
    """
    Detector de similitudes basado en machine learning con técnicas avanzadas
    """
    def __init__(self, model_name="microsoft/codebert-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        # Vectorizador TF-IDF para análisis de texto
        self.tfidf = TfidfVectorizer(
            analyzer='word',
            token_pattern=r'\S+',
            ngram_range=(1, 3),
            max_features=10000
        )
        
        # Vectorizador TF-IDF para n-gramas de caracteres (captura estructura)
        self.char_tfidf = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 5),
            max_features=10000
        )
        
        # Vectorizador para n-gramas de palabras (captura lógica y flujo)
        self.ngram_tfidf = TfidfVectorizer(
            analyzer='word',
            token_pattern=r'\S+',
            ngram_range=(2, 4),
            max_features=10000
        )
        
        # Cache para embeddings
        self.embedding_cache = {}
    
    def preprocess_code(self, code):
        """
        Preprocesa el código para mejorar la detección
        """
        # Eliminar comentarios
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)  # Python
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)  # Java/C++/JS
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)  # Comentarios multilínea
        
        # Normalizar espacios en blanco
        code = re.sub(r'\s+', ' ', code)
        
        return code.strip()
    
    def load_model(self):
        """Carga el modelo de transformers para embeddings de código"""
        if self.tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except Exception as e:
                print(f"Error cargando tokenizer: {str(e)}")
                return False
        
        if self.model is None:
            try:
                self.model = AutoModel.from_pretrained(self.model_name)
                # Poner el modelo en modo evaluación
                self.model.eval()
            except Exception as e:
                print(f"Error cargando modelo: {str(e)}")
                return False
        
        return True
    
    def get_embeddings(self, code_snippets, use_cache=True):
        """
        Obtiene embeddings para fragmentos de código usando un modelo preentrenado
        
        Args:
            code_snippets: Lista de fragmentos de código
            use_cache: Si se debe usar caché para embeddings ya calculados
        
        Returns:
            Array numpy con embeddings
        """
        # Verificar si el modelo está cargado
        if not self.load_model():
            print("No se pudo cargar el modelo, usando fallback")
            return self._get_fallback_embeddings(code_snippets)
        
        # Preprocesar código
        preprocessed_snippets = [self.preprocess_code(code) for code in code_snippets]
        
        embeddings = []
        for i, code in enumerate(preprocessed_snippets):
            # Verificar caché
            code_hash = hash(code)
            if use_cache and code_hash in self.embedding_cache:
                embeddings.append(self.embedding_cache[code_hash])
                continue
            
            try:
                # Tokenizar y obtener embeddings
                inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512, padding=True)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Usar el embedding del token [CLS] como representación del fragmento
                embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
                
                # Guardar en caché
                if use_cache:
                    self.embedding_cache[code_hash] = embedding
                
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error obteniendo embedding para fragmento {i}: {str(e)}")
                # Usar fallback
                fallback = self._get_fallback_embeddings([code])[0]
                embeddings.append(fallback)
        
        return np.array(embeddings)
    
    def _get_fallback_embeddings(self, code_snippets):
        """
        Método de respaldo para obtener embeddings cuando falla el modelo
        """
        # Usar TF-IDF como fallback
        preprocessed_snippets = [self.preprocess_code(code) for code in code_snippets]
        
        try:
            tfidf_matrix = self.tfidf.fit_transform(preprocessed_snippets)
            return tfidf_matrix.toarray()
        except:
            # Si todo falla, devolver vectores aleatorios
            return np.random.rand(len(code_snippets), 100)
    
    def calculate_similarity_matrix(self, code_snippets, weights=None):
        """
        Calcula matriz de similitud entre fragmentos de código usando múltiples técnicas
        
        Args:
            code_snippets: Lista de fragmentos de código
            weights: Pesos para cada método de similitud
        
        Returns:
            Matriz de similitud combinada
        """
        # Pesos por defecto
        if weights is None:
            weights = {
                "tfidf": 0.2,
                "char_ngram": 0.2,
                "word_ngram": 0.2,
                "embedding": 0.4
            }
        
        # Preprocesar código
        preprocessed_snippets = [self.preprocess_code(code) for code in code_snippets]
        
        # 1. Método TF-IDF estándar (captura vocabulario)
        try:
            tfidf_matrix = self.tfidf.fit_transform(preprocessed_snippets)
            tfidf_similarity = cosine_similarity(tfidf_matrix)
        except Exception as e:
            print(f"Error en TF-IDF: {str(e)}")
            tfidf_similarity = np.ones((len(code_snippets), len(code_snippets)))
        
        # 2. N-gramas de caracteres (captura estructura)
        try:
            char_matrix = self.char_tfidf.fit_transform(preprocessed_snippets)
            char_similarity = cosine_similarity(char_matrix)
        except Exception as e:
            print(f"Error en n-gramas de caracteres: {str(e)}")
            char_similarity = np.ones((len(code_snippets), len(code_snippets)))
        
        # 3. N-gramas de palabras (captura lógica y flujo)
        try:
            ngram_matrix = self.ngram_tfidf.fit_transform(preprocessed_snippets)
            ngram_similarity = cosine_similarity(ngram_matrix)
        except Exception as e:
            print(f"Error en n-gramas de palabras: {str(e)}")
            ngram_similarity = np.ones((len(code_snippets), len(code_snippets)))
        
        # 4. Embeddings del modelo + Similitud del coseno
        try:
            embeddings = self.get_embeddings(code_snippets)
            embedding_similarity = cosine_similarity(embeddings)
        except Exception as e:
            print(f"Error en embeddings: {str(e)}")
            embedding_similarity = np.ones((len(code_snippets), len(code_snippets)))
        
        # Combinar todos los métodos (promedio ponderado)
        combined_similarity = (
            weights["tfidf"] * tfidf_similarity +
            weights["char_ngram"] * char_similarity +
            weights["word_ngram"] * ngram_similarity +
            weights["embedding"] * embedding_similarity
        )
        
        # Normalizar para asegurar valores entre 0 y 1
        combined_similarity = np.clip(combined_similarity, 0, 1)
        
        return {
            "combined": combined_similarity,
            "tfidf": tfidf_similarity,
            "char_ngram": char_similarity,
            "word_ngram": ngram_similarity,
            "embedding": embedding_similarity
        }
    
    def cluster_submissions(self, code_snippets, method="dbscan", params=None):
        """
        Agrupa entregas similares usando clustering no supervisado
        
        Args:
            code_snippets: Lista de fragmentos de código
            method: Método de clustering ('dbscan', 'kmeans', 'hdbscan')
            params: Parámetros para el algoritmo de clustering
        
        Returns:
            Diccionario con clusters y metadatos
        """
        # Parámetros por defecto
        if params is None:
            params = {
                "dbscan": {"eps": 0.2, "min_samples": 2},
                "kmeans": {"n_clusters": max(2, len(code_snippets) // 5)},
                "hdbscan": {"min_cluster_size": 2, "min_samples": 1}
            }
        
        # Obtener embeddings
        embeddings = self.get_embeddings(code_snippets)
        
        # Aplicar clustering según el método elegido
        labels = None
        if method == "dbscan":
            clustering = DBSCAN(
                eps=params["dbscan"]["eps"], 
                min_samples=params["dbscan"]["min_samples"], 
                metric='cosine'
            ).fit(embeddings)
            labels = clustering.labels_
        
        elif method == "kmeans":
            n_clusters = min(params["kmeans"]["n_clusters"], len(code_snippets) - 1)
            if n_clusters < 2:
                n_clusters = 2
            
            clustering = KMeans(
                n_clusters=n_clusters, 
                random_state=42
            ).fit(embeddings)
            labels = clustering.labels_
        
        elif method == "hdbscan":
            try:
                clustering = hdbscan.HDBSCAN(
                    min_cluster_size=params["hdbscan"]["min_cluster_size"],
                    min_samples=params["hdbscan"]["min_samples"],
                    metric='euclidean'
                ).fit(embeddings)
                labels = clustering.labels_
            except Exception as e:
                print(f"Error en HDBSCAN: {str(e)}")
                # Fallback a DBSCAN
                clustering = DBSCAN(eps=0.2, min_samples=2, metric='cosine').fit(embeddings)
                labels = clustering.labels_
        
        # Agrupar por cluster
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            if label != -1:  # -1 es ruido en DBSCAN/HDBSCAN
                clusters[int(label)].append(i)
        
        # Calcular similitud intra-cluster
        intra_cluster_similarities = {}
        for label, indices in clusters.items():
            if len(indices) > 1:
                # Extraer embeddings del cluster
                cluster_embeddings = embeddings[indices]
                # Calcular similitud del coseno entre todos los pares
                sim_matrix = cosine_similarity(cluster_embeddings)
                # Calcular similitud promedio (excluyendo la diagonal)
                n = len(indices)
                sim_sum = np.sum(sim_matrix) - n  # Restar la diagonal (siempre 1)
                avg_sim = sim_sum / (n * n - n) if n > 1 else 0
                intra_cluster_similarities[label] = float(avg_sim)
        
        # Visualizar clusters
        visualization = self._visualize_clusters(embeddings, labels)
        
        return {
            "clusters": {str(k): v for k, v in clusters.items()},
            "noise": [i for i, label in enumerate(labels) if label == -1],
            "intra_cluster_similarities": intra_cluster_similarities,
            "visualization": visualization,
            "num_clusters": len(clusters),
            "method": method,
            "params": params
        }
    
    def _visualize_clusters(self, embeddings, labels):
        """
        Visualiza clusters usando reducción de dimensionalidad
        
        Args:
            embeddings: Matriz de embeddings
            labels: Etiquetas de cluster
        
        Returns:
            Imagen en base64 de la visualización
        """
        # Reducir dimensionalidad para visualización
        if len(embeddings) < 3:
            # No hay suficientes muestras para visualizar
            return None
        
        try:
            # Usar UMAP si está disponible (mejor para visualización)
            reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42)
            reduced_embeddings = reducer.fit_transform(embeddings)
        except:
            # Fallback a t-SNE
            try:
                reducer = TSNE(n_components=2, random_state=42)
                reduced_embeddings = reducer.fit_transform(embeddings)
            except:
                # Si todo falla, no visualizar
                return None
        
        # Crear gráfico
        plt.figure(figsize=(10, 8))
        
        # Colores para clusters
        unique_labels = set(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Negro para ruido
                col = [0, 0, 0, 1]
            
            class_member_mask = (labels == k)
            xy = reduced_embeddings[class_member_mask]
            plt.scatter(
                xy[:, 0], xy[:, 1],
                s=50, c=[col], label=f'Cluster {k}' if k != -1 else 'Ruido',
                alpha=0.8
            )
        
        plt.title('Visualización de Clusters')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convertir a base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_str
    
    def find_similar_code_blocks(self, code1, code2, window_size=50, stride=25, threshold=0.8):
        """
        Encuentra bloques de código similares entre dos archivos
        
        Args:
            code1: Primer fragmento de código
            code2: Segundo fragmento de código
            window_size: Tamaño de la ventana deslizante (en líneas)
            stride: Paso entre ventanas consecutivas
            threshold: Umbral de similitud
        
        Returns:
            Lista de bloques similares con metadatos
        """
        # Preprocesar código
        code1 = self.preprocess_code(code1)
        code2 = self.preprocess_code(code2)
        
        # Dividir en líneas
        lines1 = code1.split('\n')
        lines2 = code2.split('\n')
        
        similar_blocks = []
        
        # Usar ventana deslizante para encontrar bloques similares
        for i in range(0, max(1, len(lines1) - window_size + 1), stride):
            block1 = '\n'.join(lines1[i:i+window_size])
            
            for j in range(0, max(1, len(lines2) - window_size + 1), stride):
                block2 = '\n'.join(lines2[j:j+window_size])
                
                # Calcular similitud usando múltiples métodos
                similarity_results = self.calculate_similarity_matrix([block1, block2])
                similarity = similarity_results["combined"][0][1]
                
                if similarity >= threshold:
                    similar_blocks.append({
                        "block1": {
                            "start_line": i,
                            "end_line": i + window_size,
                            "code": block1
                        },
                        "block2": {
                            "start_line": j,
                            "end_line": j + window_size,
                            "code": block2
                        },
                        "similarity": float(similarity),
                        "similarity_details": {
                            "tfidf": float(similarity_results["tfidf"][0][1]),
                            "char_ngram": float(similarity_results["char_ngram"][0][1]),
                            "word_ngram": float(similarity_results["word_ngram"][0][1]),
                            "embedding": float(similarity_results["embedding"][0][1])
                        }
                    })
        
        # Ordenar por similitud
        similar_blocks.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Fusionar bloques superpuestos
        merged_blocks = self._merge_overlapping_blocks(similar_blocks)
        
        return merged_blocks
    
    def _merge_overlapping_blocks(self, blocks, overlap_threshold=0.7):
        """
        Fusiona bloques de código superpuestos
        
        Args:
            blocks: Lista de bloques similares
            overlap_threshold: Umbral para considerar superposición
        
        Returns:
            Lista de bloques fusionados
        """
        if not blocks:
            return []
        
        # Ordenar por posición del primer bloque
        sorted_blocks = sorted(blocks, key=lambda x: (x["block1"]["start_line"], x["block2"]["start_line"]))
        
        merged = [sorted_blocks[0]]
        
        for current in sorted_blocks[1:]:
            previous = merged[-1]
            
            # Calcular superposición para el primer bloque
            overlap1_start = max(previous["block1"]["start_line"], current["block1"]["start_line"])
            overlap1_end = min(previous["block1"]["end_line"], current["block1"]["end_line"])
            overlap1 = max(0, overlap1_end - overlap1_start)
            
            # Calcular superposición para el segundo bloque
            overlap2_start = max(previous["block2"]["start_line"], current["block2"]["start_line"])
            overlap2_end = min(previous["block2"]["end_line"], current["block2"]["end_line"])
            overlap2 = max(0, overlap2_end - overlap2_start)
            
            # Calcular porcentaje de superposición
            len1_prev = previous["block1"]["end_line"] - previous["block1"]["start_line"]
            len1_curr = current["block1"]["end_line"] - current["block1"]["start_line"]
            len2_prev = previous["block2"]["end_line"] - previous["block2"]["start_line"]
            len2_curr = current["block2"]["end_line"] - current["block2"]["start_line"]
            
            overlap1_pct = overlap1 / min(len1_prev, len1_curr) if min(len1_prev, len1_curr) > 0 else 0
            overlap2_pct = overlap2 / min(len2_prev, len2_curr) if min(len2_prev, len2_curr) > 0 else 0
            
            # Si hay suficiente superposición en ambos bloques, fusionar
            if overlap1_pct >= overlap_threshold and overlap2_pct >= overlap_threshold:
                merged[-1] = {
                    "block1": {
                        "start_line": min(previous["block1"]["start_line"], current["block1"]["start_line"]),
                        "end_line": max(previous["block1"]["end_line"], current["block1"]["end_line"]),
                        "code": previous["block1"]["code"]  # Mantener el código del bloque anterior
                    },
                    "block2": {
                        "start_line": min(previous["block2"]["start_line"], current["block2"]["start_line"]),
                        "end_line": max(previous["block2"]["end_line"], current["block2"]["end_line"]),
                        "code": previous["block2"]["code"]  # Mantener el código del bloque anterior
                    },
                    "similarity": max(previous["similarity"], current["similarity"]),
                    "similarity_details": previous["similarity_details"]  # Mantener detalles del bloque anterior
                }
            else:
                merged.append(current)
        
        return merged
    
    def analyze_submission_group(self, code_snippets, metadata=None):
        """
        Analiza un grupo de entregas para detectar patrones de similitud
        
        Args:
            code_snippets: Lista de fragmentos de código
            metadata: Metadatos asociados a cada fragmento (opcional)
        
        Returns:
            Análisis detallado del grupo
        """
        if len(code_snippets) < 2:
            return {"error": "Se necesitan al menos 2 fragmentos para el análisis"}
        
        # Calcular matriz de similitud
        similarity_results = self.calculate_similarity_matrix(code_snippets)
        similarity_matrix = similarity_results["combined"]
        
        # Realizar clustering
        cluster_results = self.cluster_submissions(code_snippets)
        
        # Identificar pares con alta similitud
        high_similarity_pairs = []
        for i in range(len(code_snippets)):
            for j in range(i+1, len(code_snippets)):
                similarity = similarity_matrix[i][j]
                if similarity >= 0.7:  # Umbral configurable
                    pair_info = {
                        "index1": i,
                        "index2": j,
                        "similarity": float(similarity),
                        "similarity_details": {
                            "tfidf": float(similarity_results["tfidf"][i][j]),
                            "char_ngram": float(similarity_results["char_ngram"][i][j]),
                            "word_ngram": float(similarity_results["word_ngram"][i][j]),
                            "embedding": float(similarity_results["embedding"][i][j])
                        }
                    }
                    
                    # Añadir metadatos si están disponibles
                    if metadata:
                        pair_info["metadata1"] = metadata[i]
                        pair_info["metadata2"] = metadata[j]
                    
                    high_similarity_pairs.append(pair_info)
        
        # Ordenar por similitud
        high_similarity_pairs.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Generar visualización de la matriz de similitud
        similarity_heatmap = self._generate_similarity_heatmap(similarity_matrix)
        
        return {
            "num_snippets": len(code_snippets),
            "similarity_matrix": similarity_matrix.tolist(),
            "high_similarity_pairs": high_similarity_pairs,
            "clusters": cluster_results,
            "similarity_heatmap": similarity_heatmap,
            "average_similarity": float(np.mean(similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)]))
        }
    
    def _generate_similarity_heatmap(self, similarity_matrix):
        """
        Genera un mapa de calor para la matriz de similitud
        
        Args:
            similarity_matrix: Matriz de similitud
        
        Returns:
            Imagen en base64 del mapa de calor
        """
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                similarity_matrix,
                annot=True,
                cmap="YlGnBu",
                vmin=0,
                vmax=1,
                fmt=".2f"
            )
            plt.title("Matriz de Similitud")
            plt.tight_layout()
            
            # Convertir a base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str
        except Exception as e:
            print(f"Error generando mapa de calor: {str(e)}")
            return None
    
    def get_feature_importance(self, code_snippets):
        """
        Analiza la importancia de diferentes características en la detección
        
        Args:
            code_snippets: Lista de fragmentos de código
        
        Returns:
            Análisis de importancia de características
        """
        if len(code_snippets) < 2:
            return {"error": "Se necesitan al menos 2 fragmentos para el análisis"}
        
        # Calcular similitud con diferentes pesos
        weight_variations = [
            {"tfidf": 1.0, "char_ngram": 0.0, "word_ngram": 0.0, "embedding": 0.0},
            {"tfidf": 0.0, "char_ngram": 1.0, "word_ngram": 0.0, "embedding": 0.0},
            {"tfidf": 0.0, "char_ngram": 0.0, "word_ngram": 1.0, "embedding": 0.0},
            {"tfidf": 0.0, "char_ngram": 0.0, "word_ngram": 0.0, "embedding": 1.0}
        ]
        
        feature_results = {}
        for weights in weight_variations:
            # Identificar qué característica está activa
            active_feature = next(k for k, v in weights.items() if v > 0)
            
            # Calcular matriz de similitud con estos pesos
            similarity_results = self.calculate_similarity_matrix(code_snippets, weights)
            similarity_matrix = similarity_results["combined"]
            
            # Calcular estadísticas
            upper_triangle = similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)]
            feature_results[active_feature] = {
                "mean": float(np.mean(upper_triangle)),
                "std": float(np.std(upper_triangle)),
                "min": float(np.min(upper_triangle)),
                "max": float(np.max(upper_triangle)),
                "median": float(np.median(upper_triangle))
            }
        
        # Generar visualización comparativa
        comparison_plot = self._generate_feature_comparison_plot(feature_results)
        
        return {
            "feature_stats": feature_results,
            "comparison_plot": comparison_plot
        }
    
    def _generate_feature_comparison_plot(self, feature_results):
        """
        Genera un gráfico comparativo de la importancia de características
        
        Args:
            feature_results: Resultados por característica
        
        Returns:
            Imagen en base64 del gráfico
        """
        try:
            features = list(feature_results.keys())
            means = [feature_results[f]["mean"] for f in features]
            stds = [feature_results[f]["std"] for f in features]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(features, means, yerr=stds, capsize=10, alpha=0.7)
            
            # Añadir etiquetas
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.02,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom'
                )
            
            plt.title("Comparación de Características")
            plt.ylabel("Similitud Media")
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            
            # Convertir a base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str
        except Exception as e:
            print(f"Error generando gráfico comparativo: {str(e)}")
            return None
