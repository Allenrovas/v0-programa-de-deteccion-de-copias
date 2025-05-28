import re
import esprima

class JavaScriptTokenizer:
    """
    Tokeniza código JavaScript para análisis de similitud
    """
    def __init__(self):
        self.ignore_comments = True
        self.normalize_identifiers = True
        # Palabras clave de JavaScript
        self.js_keywords = {
            'abstract', 'arguments', 'await', 'boolean', 'break', 'byte', 'case', 'catch',
            'char', 'class', 'const', 'continue', 'debugger', 'default', 'delete', 'do',
            'double', 'else', 'enum', 'eval', 'export', 'extends', 'false', 'final',
            'finally', 'float', 'for', 'function', 'goto', 'if', 'implements', 'import',
            'in', 'instanceof', 'int', 'interface', 'let', 'long', 'native', 'new', 'null',
            'package', 'private', 'protected', 'public', 'return', 'short', 'static',
            'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient',
            'true', 'try', 'typeof', 'var', 'void', 'volatile', 'while', 'with', 'yield'
        }
    
    def tokenize(self, code):
        """
        Convierte código JavaScript en una secuencia de tokens
        """
        tokens = []
        try:
            # Tokenizar usando esprima
            token_list = esprima.tokenize(code)
            
            for token in token_list:
                token_type = token.type
                token_value = token.value
                
                # Ignorar comentarios si está configurado
                if self.ignore_comments and token_type == 'CommentLine' or token_type == 'CommentBlock':
                    continue
                
                # Normalizar identificadores si está configurado
                if self.normalize_identifiers and token_type == 'Identifier':
                    if token_value not in self.js_keywords:
                        tokens.append(f"ID_{token_type}")
                    else:
                        tokens.append(token_value)
                else:
                    # Para otros tipos de tokens, usar su valor
                    tokens.append(token_value)
            
            return tokens
        except Exception as e:
            print(f"Error tokenizando código JavaScript: {str(e)}")
            # Fallback: tokenización simple basada en regex
            if not tokens:
                # Eliminar comentarios si está configurado
                if self.ignore_comments:
                    # Eliminar comentarios de línea
                    code = re.sub(r'//.*', '', code)
                    # Eliminar comentarios de bloque
                    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
                
                simple_tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[^\s]', code)
                for token in simple_tokens:
                    if token.strip():
                        if self.normalize_identifiers and token not in self.js_keywords and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
                            tokens.append("ID_IDENTIFIER")
                        else:
                            tokens.append(token)
            return tokens
    
    def get_ast(self, code):
        """
        Obtiene el AST (Abstract Syntax Tree) del código JavaScript
        """
        try:
            return esprima.parseScript(code, {'loc': True, 'range': True})
        except:
            return None