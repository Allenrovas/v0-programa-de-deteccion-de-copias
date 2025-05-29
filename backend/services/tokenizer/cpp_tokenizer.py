import re
import clang.cindex

class CppTokenizer:
    """
    Tokeniza código C++ para análisis de similitud
    """
    def __init__(self):
        self.ignore_comments = True
        self.normalize_identifiers = True
        # Palabras clave de C++
        self.cpp_keywords = {
            'alignas', 'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor',
            'bool', 'break', 'case', 'catch', 'char', 'char8_t', 'char16_t', 'char32_t',
            'class', 'compl', 'concept', 'const', 'consteval', 'constexpr', 'constinit',
            'const_cast', 'continue', 'co_await', 'co_return', 'co_yield', 'decltype',
            'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum', 'explicit',
            'export', 'extern', 'false', 'float', 'for', 'friend', 'goto', 'if', 'inline',
            'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq',
            'nullptr', 'operator', 'or', 'or_eq', 'private', 'protected', 'public',
            'register', 'reinterpret_cast', 'requires', 'return', 'short', 'signed',
            'sizeof', 'static', 'static_assert', 'static_cast', 'struct', 'switch',
            'template', 'this', 'thread_local', 'throw', 'true', 'try', 'typedef',
            'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void',
            'volatile', 'wchar_t', 'while', 'xor', 'xor_eq'
        }
        
        # Inicializar clang si está disponible
        try:
            clang.cindex.Config.set_library_file('/usr/lib/llvm-10/lib/libclang.so.1')
            self.use_clang = True
        except:
            self.use_clang = False
    
    def tokenize(self, code):
        """
        Convierte código C++ en una secuencia de tokens
        """
        tokens = []
        
        # Eliminar comentarios si está configurado
        if self.ignore_comments:
            # Eliminar comentarios de línea
            code = re.sub(r'//.*', '', code)
            # Eliminar comentarios de bloque
            code = re.sub(r'/\*[\s\S]*?\*/', '', code)
        
        try:
            if self.use_clang:
                # Usar clang para tokenización avanzada
                index = clang.cindex.Index.create()
                tu = index.parse('temp.cpp', unsaved_files=[('temp.cpp', code)], 
                                args=['-std=c++17'])
                
                for token in tu.cursor.get_tokens():
                    token_spelling = token.spelling
                    token_kind = token.kind
                    
                    # Normalizar identificadores
                    if self.normalize_identifiers and token_kind == clang.cindex.TokenKind.IDENTIFIER:
                        if token_spelling not in self.cpp_keywords:
                            tokens.append(f"ID_{token_kind.name}")
                        else:
                            tokens.append(token_spelling)
                    else:
                        tokens.append(token_spelling)
            else:
                # Fallback: tokenización simple basada en regex
                simple_tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[^\s]', code)
                for token in simple_tokens:
                    if token.strip():
                        if self.normalize_identifiers and token not in self.cpp_keywords and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
                            tokens.append("ID_IDENTIFIER")
                        else:
                            tokens.append(token)
            
            return tokens
        except Exception as e:
            print(f"Error tokenizando código C++: {str(e)}")
            # Tokenización simple como fallback
            simple_tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[^\s]', code)
            for token in simple_tokens:
                if token.strip():
                    tokens.append(token)
            return tokens
    
    def preprocess_code(self, code):
        """
        Preprocesa el código C++ para eliminar directivas de preprocesador
        """
        # Eliminar directivas #include, #define, etc.
        code = re.sub(r'#\s*include\s*[<"].*[>"]', '', code)
        code = re.sub(r'#\s*define\s+\w+($$.*?$$)?\s+.*', '', code)
        code = re.sub(r'#\s*(if|ifdef|ifndef|else|elif|endif|pragma).*', '', code)
        return code