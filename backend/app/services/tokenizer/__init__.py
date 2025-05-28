from app.services.tokenizer.python_tokenizer import PythonTokenizer
from app.services.tokenizer.java_tokenizer import JavaTokenizer
from app.services.tokenizer.cpp_tokenizer import CppTokenizer
from app.services.tokenizer.js_tokenizer import JavaScriptTokenizer

def get_tokenizer_for_language(language):
    """
    Devuelve el tokenizador apropiado para el lenguaje especificado
    """
    tokenizers = {
        "python": PythonTokenizer(),
        "java": JavaTokenizer(),
        "cpp": CppTokenizer(),
        "javascript": JavaScriptTokenizer()
    }
    
    return tokenizers.get(language, PythonTokenizer())  # Default a Python si no se encuentra