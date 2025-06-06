import os
import json
import argparse
import difflib
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from pathlib import Path

def load_results(results_file):
    """Carga los resultados de similitud desde un archivo JSON"""
    with open(results_file, 'r') as f:
        return json.load(f)

def get_file_content(file_path):
    """Lee el contenido de un archivo"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        return f"Error leyendo archivo: {str(e)}"

def highlight_code(code, language):
    """Resalta el código usando Pygments"""
    lexer = get_lexer_by_name(language)
    formatter = HtmlFormatter(style='colorful')
    return highlight(code, lexer, formatter)

def generate_diff(code1, code2):
    """Genera un diff HTML entre dos fragmentos de código"""
    diff = difflib.HtmlDiff()
    return diff.make_table(code1.splitlines(), code2.splitlines())

def visualize_fragments(results_file, output_dir, language):
    """
    Visualiza los fragmentos de código similares
    
    Args:
        results_file: Archivo JSON con los resultados de similitud
        output_dir: Directorio donde guardar los archivos HTML
        language: Lenguaje de programación
    """
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar resultados
    results = load_results(results_file)
    
    # Crear archivo HTML principal
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, 'w', encoding='utf-8') as index_file:
        # Escribir encabezado
        index_file.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Visualización de Fragmentos Similares</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .submission-pair {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .file-pair {{ margin-bottom: 20px; background-color: #f9f9f9; padding: 10px; border-radius: 5px; }}
                .fragment {{ margin-bottom: 15px; }}
                .similarity {{ font-weight: bold; color: #c00; }}
                .code-container {{ display: flex; margin-top: 10px; }}
                .code-block {{ flex: 1; margin-right: 10px; overflow-x: auto; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 3px; }}
                .diff-view {{ margin-top: 10px; overflow-x: auto; }}
                table.diff {{ font-family: monospace; border-collapse: collapse; }}
                .diff td {{ padding: 0 3px; }}
                .diff_header {{ background-color: #e0e0e0; }}
                .diff_add {{ background-color: #aaffaa; }}
                .diff_chg {{ background-color: #ffff77; }}
                .diff_sub {{ background-color: #ffaaaa; }}
            </style>
        </head>
        <body>
            <h1>Visualización de Fragmentos Similares</h1>
            <p>Lenguaje: {language}</p>
            <p>Total de pares de entregas: {len(results.get('similarity_results', []))}</p>
        """)
        
        # Escribir pares de entregas
        for i, pair in enumerate(results.get('similarity_results', [])):
            index_file.write(f"""
            <div class="submission-pair">
                <h2>Par {i+1}: {pair['submission1']} - {pair['submission2']}</h2>
                <p class="similarity">Similitud máxima: {pair['max_similarity']:.2f}</p>
            """)
            
            # Escribir pares de archivos
            for j, file_pair in enumerate(pair.get('similar_files', [])):
                index_file.write(f"""
                <div class="file-pair">
                    <h3>Archivos similares {j+1}</h3>
                    <p>Archivo 1: {file_pair['file1']}</p>
                    <p>Archivo 2: {file_pair['file2']}</p>
                    <p class="similarity">Similitud combinada: {file_pair['combined_similarity']:.2f}</p>
                    <p>Similitud por tokens: {file_pair['token_similarity']:.2f}</p>
                    <p>Similitud por ML: {file_pair['ml_similarity']:.2f}</p>
                """)
                
                # Escribir fragmentos
                for k, fragment in enumerate(file_pair.get('fragments', [])):
                    code1 = fragment.get('code1', '')
                    code2 = fragment.get('code2', '')
                    
                    index_file.write(f"""
                    <div class="fragment">
                        <h4>Fragmento similar {k+1}</h4>
                        <p class="similarity">Similitud: {fragment['similarity']:.2f}</p>
                        <div class="code-container">
                            <div class="code-block">
                                <h5>Código 1 (tokens {fragment['fragment1_start']}-{fragment['fragment1_end']})</h5>
                                <pre>{code1}</pre>
                            </div>
                            <div class="code-block">
                                <h5>Código 2 (tokens {fragment['fragment2_start']}-{fragment['fragment2_end']})</h5>
                                <pre>{code2}</pre>
                            </div>
                        </div>
                        <div class="diff-view">
                            <h5>Vista de diferencias</h5>
                            {generate_diff(code1, code2)}
                        </div>
                    </div>
                    """)
                
                index_file.write("</div>")  # Cerrar file-pair
            
            index_file.write("</div>")  # Cerrar submission-pair
        
        # Cerrar HTML
        index_file.write("""
        </body>
        </html>
        """)
    
    print(f"Visualización generada en {index_path}")
    return index_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualiza fragmentos de código similares")
    parser.add_argument("results_file", help="Archivo JSON con los resultados de similitud")
    parser.add_argument("--output", "-o", default="visualization", help="Directorio de salida")
    parser.add_argument("--language", "-l", default="python", help="Lenguaje de programación")
    
    args = parser.parse_args()
    visualize_fragments(args.results_file, args.output, args.language)
