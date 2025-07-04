import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import List, Dict, Tuple, Any, Optional

class AdaptivePlagiarismLearner:
    """
    Sistema de aprendizaje adaptativo para mejorar la detección de plagio con el tiempo
    """
    def __init__(self, model_path="data/models"):
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)
        self.model_file = os.path.join(model_path, "plagiarism_classifier.joblib")
        self.feature_importance_file = os.path.join(model_path, "feature_importance.csv")
        self.training_data_file = os.path.join(model_path, "training_data.csv")
        
        # Cargar modelo si existe
        self.model = self._load_model()
        
        # Historial de pesos
        self.weight_history = self._load_weight_history()
    
    def _load_model(self):
        """Carga el modelo de clasificación si existe"""
        if os.path.exists(self.model_file):
            try:
                return joblib.load(self.model_file)
            except:
                print("Error cargando modelo, creando uno nuevo")
        
        # Crear modelo por defecto
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def _load_weight_history(self):
        """Carga el historial de pesos si existe"""
        if os.path.exists(self.feature_importance_file):
            try:
                return pd.read_csv(self.feature_importance_file)
            except:
                print("Error cargando historial de pesos, creando uno nuevo")
        
        # Crear historial por defecto
        return pd.DataFrame({
            'date': [],
            'lcs': [],
            'sequence_matcher': [],
            'ngram': [],
            'levenshtein': [],
            'tfidf_cosine': [],
            'winnowing': [],
            'ml_tfidf': [],
            'ml_char_ngram': [],
            'ml_word_ngram': [],
            'ml_embedding': []
        })
    
    def _load_training_data(self):
        """Carga los datos de entrenamiento si existen"""
        if os.path.exists(self.training_data_file):
            try:
                return pd.read_csv(self.training_data_file)
            except:
                print("Error cargando datos de entrenamiento, creando nuevos")
        
        # Crear datos por defecto
        return pd.DataFrame({
            'lcs': [],
            'sequence_matcher': [],
            'ngram': [],
            'levenshtein': [],
            'tfidf_cosine': [],
            'winnowing': [],
            'ml_tfidf': [],
            'ml_char_ngram': [],
            'ml_word_ngram': [],
            'ml_embedding': [],
            'is_plagiarism': []
        })
    
    def add_training_example(self, features: Dict[str, float], is_plagiarism: bool):
        """
        Añade un ejemplo de entrenamiento al sistema
        
        Args:
            features: Diccionario con valores de similitud para diferentes métricas
            is_plagiarism: True si es plagio, False si no
        """
        # Cargar datos existentes
        df = self._load_training_data()
        
        # Crear nueva fila
        new_row = {
            'lcs': features.get('lcs', 0),
            'sequence_matcher': features.get('sequence_matcher', 0),
            'ngram': features.get('ngram', 0),
            'levenshtein': features.get('levenshtein', 0),
            'tfidf_cosine': features.get('tfidf_cosine', 0),
            'winnowing': features.get('winnowing', 0),
            'ml_tfidf': features.get('ml_tfidf', 0),
            'ml_char_ngram': features.get('ml_char_ngram', 0),
            'ml_word_ngram': features.get('ml_word_ngram', 0),
            'ml_embedding': features.get('ml_embedding', 0),
            'is_plagiarism': int(is_plagiarism)
        }
        
        # Añadir a DataFrame
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Guardar
        df.to_csv(self.training_data_file, index=False)
        
        # Reentrenar modelo si hay suficientes datos
        if len(df) >= 10:
            self.train_model()
    
    def train_model(self):
        """Entrena el modelo con los datos disponibles"""
        # Cargar datos
        df = self._load_training_data()
        
        if len(df) < 10:
            return {"error": "Datos insuficientes para entrenar"}
        
        # Separar características y etiquetas
        X = df.drop('is_plagiarism', axis=1)
        y = df['is_plagiarism']
        
        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar modelo
        self.model.fit(X_train, y_train)
        
        # Evaluar
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Guardar modelo
        joblib.dump(self.model, self.model_file)
        
        # Actualizar importancia de características
        self._update_feature_importance()
        
        return {
            "accuracy": report['accuracy'],
            "precision": report['weighted avg']['precision'],
            "recall": report['weighted avg']['recall'],
            "f1": report['weighted avg']['f1-score'],
            "num_samples": len(df)
        }
    
    def _update_feature_importance(self):
        """Actualiza la importancia de las características"""
        if not hasattr(self.model, 'feature_importances_'):
            return
        
        # Obtener importancia de características
        feature_names = [
            'lcs', 'sequence_matcher', 'ngram', 'levenshtein', 'tfidf_cosine',
            'winnowing', 'ml_tfidf', 'ml_char_ngram', 'ml_word_ngram', 'ml_embedding'
        ]
        importances = self.model.feature_importances_
        
        # Crear nueva fila para el historial
        new_row = {name: imp for name, imp in zip(feature_names, importances)}
        new_row['date'] = pd.Timestamp.now().isoformat()
        
        # Añadir al historial
        self.weight_history = pd.concat([self.weight_history, pd.DataFrame([new_row])], ignore_index=True)
        
        # Guardar
        self.weight_history.to_csv(self.feature_importance_file, index=False)
    
    def predict_plagiarism(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predice si un par de archivos es plagio basado en sus características
        
        Args:
            features: Diccionario con valores de similitud para diferentes métricas
        
        Returns:
            Diccionario con predicción y probabilidad
        """
        # Verificar si el modelo está entrenado
        if not hasattr(self.model, 'classes_'):
            # Modelo no entrenado, usar regla simple
            combined = sum(features.values()) / len(features)
            return {
                "is_plagiarism": combined > 0.6,
                "probability": combined,
                "method": "simple_average"
            }
        
        # Preparar características en el orden correcto
        feature_names = [
            'lcs', 'sequence_matcher', 'ngram', 'levenshtein', 'tfidf_cosine',
            'winnowing', 'ml_tfidf', 'ml_char_ngram', 'ml_word_ngram', 'ml_embedding'
        ]
        
        X = np.array([[features.get(name, 0) for name in feature_names]])
        
        # Predecir
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        return {
            "is_plagiarism": bool(prediction),
            "probability": float(probabilities[1] if prediction else probabilities[0]),
            "method": "trained_model",
            "feature_importance": {
                name: float(imp) for name, imp in zip(feature_names, self.model.feature_importances_)
            } if hasattr(self.model, 'feature_importances_') else {}
        }
    
    def get_optimal_weights(self) -> Dict[str, float]:
        """
        Obtiene los pesos óptimos basados en la importancia de características
        
        Returns:
            Diccionario con pesos para diferentes métricas
        """
        if not hasattr(self.model, 'feature_importances_'):
            # Modelo no entrenado, usar pesos por defecto
            return {
                "token_based": {
                    "lcs": 0.15,
                    "sequence_matcher": 0.15,
                    "ngram": 0.2,
                    "levenshtein": 0.2,
                    "tfidf_cosine": 0.2,
                    "winnowing": 0.1
                },
                "ml_based": {
                    "tfidf": 0.2,
                    "char_ngram": 0.2,
                    "word_ngram": 0.2,
                    "embedding": 0.4
                },
                "combined": {
                    "token": 0.6,
                    "ml": 0.4
                }
            }
        
        # Obtener importancia de características
        feature_names = [
            'lcs', 'sequence_matcher', 'ngram', 'levenshtein', 'tfidf_cosine',
            'winnowing', 'ml_tfidf', 'ml_char_ngram', 'ml_word_ngram', 'ml_embedding'
        ]
        importances = self.model.feature_importances_
        
        # Normalizar importancias por grupo
        token_features = ['lcs', 'sequence_matcher', 'ngram', 'levenshtein', 'tfidf_cosine', 'winnowing']
        ml_features = ['ml_tfidf', 'ml_char_ngram', 'ml_word_ngram', 'ml_embedding']
        
        token_importances = [importances[feature_names.index(f)] for f in token_features]
        ml_importances = [importances[feature_names.index(f)] for f in ml_features]
        
        # Normalizar a suma 1
        token_sum = sum(token_importances)
        ml_sum = sum(ml_importances)
        
        if token_sum > 0:
            token_weights = {f: float(importances[feature_names.index(f)] / token_sum) for f in token_features}
        else:
            token_weights = {f: 1.0 / len(token_features) for f in token_features}
        
        if ml_sum > 0:
            ml_weights = {
                "tfidf": float(importances[feature_names.index('ml_tfidf')] / ml_sum),
                "char_ngram": float(importances[feature_names.index('ml_char_ngram')] / ml_sum),
                "word_ngram": float(importances[feature_names.index('ml_word_ngram')] / ml_sum),
                "embedding": float(importances[feature_names.index('ml_embedding')] / ml_sum)
            }
        else:
            ml_weights = {"tfidf": 0.2, "char_ngram": 0.2, "word_ngram": 0.2, "embedding": 0.4}
        
        # Calcular peso relativo entre token y ml
        total_importance = token_sum + ml_sum
        if total_importance > 0:
            combined_weights = {
                "token": float(token_sum / total_importance),
                "ml": float(ml_sum / total_importance)
            }
        else:
            combined_weights = {"token": 0.6, "ml": 0.4}
        
        return {
            "token_based": token_weights,
            "ml_based": ml_weights,
            "combined": combined_weights
        }
    
    def get_weight_history_plot(self):
        """
        Genera un gráfico de la evolución de los pesos con el tiempo
        
        Returns:
            Imagen en base64 del gráfico
        """
        import matplotlib.pyplot as plt
        import io
        import base64
        
        if len(self.weight_history) < 2:
            return None
        
        try:
            # Convertir fecha a datetime
            self.weight_history['date'] = pd.to_datetime(self.weight_history['date'])
            
            # Seleccionar columnas para graficar
            feature_cols = [
                'lcs', 'sequence_matcher', 'ngram', 'levenshtein', 'tfidf_cosine',
                'winnowing', 'ml_tfidf', 'ml_char_ngram', 'ml_word_ngram', 'ml_embedding'
            ]
            
            # Crear gráfico
            plt.figure(figsize=(12, 8))
            
            for col in feature_cols:
                if col in self.weight_history.columns:
                    plt.plot(self.weight_history['date'], self.weight_history[col], label=col)
            
            plt.title('Evolución de la Importancia de Características')
            plt.xlabel('Fecha')
            plt.ylabel('Importancia')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Convertir a base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str
        except Exception as e:
            print(f"Error generando gráfico de historial: {str(e)}")
            return None
