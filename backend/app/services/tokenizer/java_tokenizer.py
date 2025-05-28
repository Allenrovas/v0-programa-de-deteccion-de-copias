import re
import keyword
import javalang

class JavaTokenizer:
    """
    Tokeniza código Java para análisis de similitud
    """
    def __init__(self):
        self.ignore_comments = True
        self.normalize_identifiers = True
        # Palabras clave de Java
        self.java_keywords = {
            'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 
            'class', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum',
            'extends', 'final', 'finally', 'float', 'for', 'goto', 'if', 'implements',
            'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new', 'package',
            'private', 'protected', 'public', 'return', 'short', 'static', 'strictfp',
            'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient',
            'try', 'void', 'volatile', 'while', 'true', 'false', 'null'
        }
    
    def tokenize(self, code):
        """
        Convierte código Java en una secuencia de tokens
        """
        tokens = []
        try:
            # Eliminar comentarios si está configurado
            if self.ignore_comments:
                # Eliminar comentarios de línea
                code = re.sub(r'//.*', '', code)
                # Eliminar comentarios de bloque
                code = re.sub(r'/\*[\s\S]*?\*/', '', code)
            
            # Tokenizar usando javalang
            token_stream = list(javalang.tokenizer.tokenize(code))
            
            for token in token_stream:
                token_type = token.__class__.__name__
                token_value = token.value
                
                # Normalizar identificadores si está configurado
                if self.normalize_identifiers and token_type == 'Identifier':
                    if token_value not in self.java_keywords:
                        tokens.append(f"ID_{token_type}")
                    else:
                        tokens.append(token_value)
                else:
                    # Para otros tipos de tokens, usar su valor
                    tokens.append(token_value)
            
            return tokens
        except Exception as e:
            print(f"Error tokenizando código Java: {str(e)}")
            # Fallback: tokenización simple basada en espacios y puntuación
            if not tokens:
                simple_tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[^\s]', code)
                for token in simple_tokens:
                    if token.strip():
                        tokens.append(token)
            return tokens
    
    def get_ast(self, code):
        """
        Obtiene el AST (Abstract Syntax Tree) del código Java
        """
        try:
            return javalang.parse.parse(code)
        except:
            return None