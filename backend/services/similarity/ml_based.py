import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from transformers import AutoTokenizer, AutoModel
import torch
import re

class MLSimilarityDetector:
    """
    Detector de similitudes basado en machine learning
    """
    def __init__(self, model_name="microsoft/codebert-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.tfidf = TfidfVectorizer(
            analyzer='word',
            token_pattern=r'\S+',
            ngram_range=(1, 3),
            max_features=10000
        )
    
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
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.model is None:
            self.model = AutoModel.from_pretrained(self.model_name)
    
    def get_embeddings(self, code_snippets):
        """
        Obtiene embeddings para fragmentos de código usando un modelo preentrenado
        """
        self.load_model()
        
        # Preprocesar código
        preprocessed_snippets = [self.preprocess_code(code) for code in code_snippets]
        
        embeddings = []
        for code in preprocessed_snippets:
            # Tokenizar y obtener embeddings
            inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Usar el embedding del token [CLS] como representación del fragmento
            embeddings.append(outputs.last_hidden_state[:, 0, :].numpy().flatten())
        
        return np.array(embeddings)
    
    def calculate_similarity_matrix(self, code_snippets):
        """
        Calcula matriz de similitud entre fragmentos de código
        """
        # Método 1: TF-IDF + Similitud del coseno
        preprocessed_snippets = [self.preprocess_code(code) for code in code_snippets]
        tfidf_matrix = self.tfidf.fit_transform(preprocessed_snippets)
        tfidf_similarity = cosine_similarity(tfidf_matrix)
        
        # Método 2: Embeddings del modelo + Similitud del coseno
        embeddings = self.get_embeddings(code_snippets)
        embedding_similarity = cosine_similarity(embeddings)
        
        # Combinar ambos métodos (promedio ponderado)
        combined_similarity = 0.3 * tfidf_similarity + 0.7 * embedding_similarity
        
        return combined_similarity
    
    def cluster_submissions(self, code_snippets, eps=0.2, min_samples=2):
        """
        Agrupa entregas similares usando clustering
        """
        # Obtener embeddings
        embeddings = self.get_embeddings(code_snippets)
        
        # Aplicar DBSCAN para clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
        
        # Obtener etiquetas de cluster
        labels = clustering.labels_
        
        # Agrupar por cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:  # Ruido
                continue
            
            if label not in clusters:
                clusters[label] = []
            
            clusters[label].append(i)
        
        return clusters
    
    def find_similar_code_blocks(self, code1, code2, window_size=50, stride=25, threshold=0.8):
        """
        Encuentra bloques de código similares entre dos archivos
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
                
                # Calcular similitud
                similarity = self.calculate_similarity_matrix([block1, block2])[0][1]
                
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
                        "similarity": float(similarity)
                    })
        
        # Ordenar por similitud
        similar_blocks.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similar_blocks
