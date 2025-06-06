"""
Configuración para el detector de plagio
"""

# Configuración general
DEFAULT_CONFIG = {
    # Umbrales
    "similarity_threshold": 0.6,  # Umbral para considerar plagio
    "fragment_threshold": 0.7,    # Umbral para fragmentos similares
    
    # Pesos para diferentes métricas de similitud
    "similarity_weights": {
        "lcs": 0.15,              # Longest Common Subsequence
        "sequence_matcher": 0.15, # SequenceMatcher
        "ngram": 0.2,             # N-gram
        "levenshtein": 0.2,       # Levenshtein
        "tfidf_cosine": 0.2,      # TF-IDF Cosine
        "winnowing": 0.1          # Winnowing
    },
    
    # Ponderación entre similitud basada en tokens y ML
    "token_ml_weights": {
        "token": 0.6,             # Peso para similitud basada en tokens
        "ml": 0.4                 # Peso para similitud basada en ML
    },
    
    # Parámetros para fragmentos
    "fragment_params": {
        "window_size": 20,        # Tamaño de ventana para buscar fragmentos
        "stride": 10              # Paso entre ventanas consecutivas
    },
    
    # Parámetros para n-gramas
    "ngram_params": {
        "size": 3                 # Tamaño de n-gramas
    }
}

# Configuraciones específicas por lenguaje
LANGUAGE_CONFIGS = {
    "python": {
        "similarity_weights": {
            "lcs": 0.15,
            "sequence_matcher": 0.15,
            "ngram": 0.25,
            "levenshtein": 0.15,
            "tfidf_cosine": 0.2,
            "winnowing": 0.1
        }
    },
    "java": {
        "similarity_weights": {
            "lcs": 0.15,
            "sequence_matcher": 0.15,
            "ngram": 0.2,
            "levenshtein": 0.15,
            "tfidf_cosine": 0.25,
            "winnowing": 0.1
        }
    },
    "cpp": {
        "similarity_weights": {
            "lcs": 0.15,
            "sequence_matcher": 0.15,
            "ngram": 0.2,
            "levenshtein": 0.15,
            "tfidf_cosine": 0.25,
            "winnowing": 0.1
        }
    },
    "javascript": {
        "similarity_weights": {
            "lcs": 0.15,
            "sequence_matcher": 0.15,
            "ngram": 0.25,
            "levenshtein": 0.15,
            "tfidf_cosine": 0.2,
            "winnowing": 0.1
        }
    }
}

def get_config(language=None):
    """
    Obtiene la configuración para el detector
    
    Args:
        language: Lenguaje de programación (opcional)
    
    Returns:
        Configuración para el detector
    """
    config = DEFAULT_CONFIG.copy()

    if language:
        lang_config = LANGUAGE_CONFIGS.get(language, {})
        config.update(lang_config)

    return config