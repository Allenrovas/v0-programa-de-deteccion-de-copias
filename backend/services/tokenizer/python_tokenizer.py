import ast
import tokenize
import keyword 
from io import BytesIO

class PythonTokenizer:
    """
    Tokeniza código Python para análisis de similitud
    """
    def __init__(self):
        self.ignore_comments = True
        self.normalize_identifiers = True
    
    def tokenize(self, code):
        """
        Convierte código Python en una secuencia de tokens
        """
        tokens = []
        try:
            # Normalizar saltos de línea
            code_bytes = code.encode('utf-8')
            
            # Tokenizar usando la biblioteca estándar
            for tok in tokenize.tokenize(BytesIO(code_bytes).readline):
                # Ignorar comentarios si está configurado
                if self.ignore_comments and tok.type == tokenize.COMMENT:
                    continue
                
                # Normalizar identificadores si está configurado
                if self.normalize_identifiers and tok.type == tokenize.NAME:
                    if tok.string not in ['True', 'False', 'None'] and not tok.string.startswith('__'):
                        # Preservar palabras clave pero normalizar otros identificadores
                        if tok.string in keyword.kwlist:
                            tokens.append(tok.string)
                        else:
                            tokens.append(f"ID_{tok.type}")
                        continue
                
                # Agregar el token a la lista
                if tok.type != tokenize.ENCODING and tok.type != tokenize.ENDMARKER:
                    tokens.append(tok.string)
            
            return tokens
        except Exception as e:
            print(f"Error tokenizando código Python: {str(e)}")
            return []
    
    def get_ast(self, code):
        """
        Obtiene el AST (Abstract Syntax Tree) del código
        """
        try:
            return ast.parse(code)
        except SyntaxError:
            return None
