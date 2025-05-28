import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

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
            ngram_range=(1, 3)
        )
    
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
        
        embeddings = []
        for code in code_snippets:
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
        tfidf_matrix = self.tfidf.fit_transform(code_snippets)
        tfidf_similarity = cosine_similarity(tfidf_matrix)
        
        # Método 2: Embeddings del modelo + Similitud del coseno
        embeddings = self.get_embeddings(code_snippets)
        embedding_similarity = cosine_similarity(embeddings)
        
        # Combinar ambos métodos (promedio ponderado)
        combined_similarity = 0.3 * tfidf_similarity + 0.7 * embedding_similarity
        
        return combined_similarity