import os
import json
import argparse
import difflib
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import HtmlFormatter
from pathlib import Path
import re

def load_results(results_file):
    """Carga los resultados de similitud desde un archivo JSON"""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_file_content(file_path):
    """Lee el contenido de un archivo"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        return f"Error leyendo archivo: {str(e)}"

def highlight_code_with_fragments(code, language, fragments):
    """
    Resalta el código completo con fragmentos similares marcados
    Similar a como lo hace GitKraken con diferencias
    """
    try:
        lexer = get_lexer_by_name(language)
    except:
        lexer = guess_lexer(code)
    
    # Crear formatter personalizado con estilos para fragmentos
    formatter = HtmlFormatter(
        style='github',
        linenos=True,
        linenostart=1,
        cssclass="source-code",
        wrapcode=True
    )
    
    # Resaltar código base
    highlighted = highlight(code, lexer, formatter)
    
    # Añadir marcadores de fragmentos similares
    lines = code.split('\n')
    
    # Crear mapa de líneas que contienen fragmentos similares
    fragment_lines = set()
    fragment_details = {}
    
    for i, fragment in enumerate(fragments):
        if 'highlights' in fragment:
            for highlight_info in fragment['highlights']:
                line_num = highlight_info['line']
                fragment_lines.add(line_num)
                if line_num not in fragment_details:
                    fragment_details[line_num] = []
                fragment_details[line_num].append({
                    'fragment_id': i,
                    'similarity': fragment.get('similarity', 0),
                    'start_col': highlight_info.get('start_col', 0),
                    'end_col': highlight_info.get('end_col', len(highlight_info.get('content', ''))),
                    'content': highlight_info.get('content', '')
                })
    
    # Modificar el HTML para incluir marcadores de fragmentos
    if fragment_lines:
        # Añadir CSS personalizado para fragmentos
        fragment_css = """
        <style>
        .fragment-highlight {
            background-color: rgba(255, 235, 59, 0.3) !important;
            border-left: 4px solid #ff9800;
            position: relative;
        }
        .fragment-marker {
            background-color: #ff9800;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            position: absolute;
            right: 5px;
            top: 2px;
        }
        .similarity-high { border-left-color: #f44336; }
        .similarity-medium { border-left-color: #ff9800; }
        .similarity-low { border-left-color: #ffeb3b; }
        </style>
        """
        
        # Insertar CSS en el HTML
        highlighted = highlighted.replace('<div class="source-code">', 
                                        fragment_css + '<div class="source-code">')
        
        # Marcar líneas con fragmentos
        for line_num in fragment_lines:
            details = fragment_details[line_num]
            similarity = max(d['similarity'] for d in details)
            
            # Determinar clase de similitud
            if similarity > 0.8:
                sim_class = "similarity-high"
            elif similarity > 0.6:
                sim_class = "similarity-medium"
            else:
                sim_class = "similarity-low"
            
            # Buscar y reemplazar la línea en el HTML
            line_pattern = f'<span class="linenos">{line_num:4d}</span>'
            if line_pattern in highlighted:
                marker = f'<span class="fragment-marker">{similarity:.2f}</span>'
                replacement = f'<span class="linenos fragment-highlight {sim_class}">{line_num:4d}</span>{marker}'
                highlighted = highlighted.replace(line_pattern, replacement, 1)
    
    return highlighted

def create_side_by_side_view(code1, code2, language, fragments1, fragments2):
    """Crea vista lado a lado de dos archivos con fragmentos resaltados"""
    highlighted1 = highlight_code_with_fragments(code1, language, fragments1)
    highlighted2 = highlight_code_with_fragments(code2, language, fragments2)
    
    return f"""
    <div class="side-by-side-container">
        <div class="file-panel">
            <h4>Archivo 1</h4>
            {highlighted1}
        </div>
        <div class="file-panel">
            <h4>Archivo 2</h4>
            {highlighted2}
        </div>
    </div>
    """

def generate_diff_view(code1, code2):
    """Genera vista de diferencias unificada"""
    diff = difflib.unified_diff(
        code1.splitlines(keepends=True),
        code2.splitlines(keepends=True),
        fromfile="Archivo 1",
        tofile="Archivo 2",
        lineterm=""
    )
    
    diff_html = "<pre class='diff-view'>"
    for line in diff:
        if line.startswith('+'):
            diff_html += f"<span class='diff-add'>{line}</span>"
        elif line.startswith('-'):
            diff_html += f"<span class='diff-remove'>{line}</span>"
        elif line.startswith('@@'):
            diff_html += f"<span class='diff-header'>{line}</span>"
        else:
            diff_html += f"<span class='diff-context'>{line}</span>"
    diff_html += "</pre>"
    
    return diff_html

def visualize_fragments(results_file, output_dir, language):
    """
    Visualiza los fragmentos de código similares con vista completa de archivos
    Similar a GitKraken o herramientas de diff modernas
    """
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar resultados
    results = load_results(results_file)
    
    # CSS mejorado para la visualización
    enhanced_css = """
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f8f9fa;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .stats {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            flex: 1;
        }
        .submission-pair { 
            background: white;
            margin-bottom: 30px; 
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .pair-header {
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 1px solid #dee2e6;
        }
        .similarity-badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            margin-left: 10px;
        }
        .similarity-high { background-color: #dc3545; }
        .similarity-medium { background-color: #fd7e14; }
        .similarity-low { background-color: #ffc107; color: #000; }
        .file-comparison {
            padding: 20px;
        }
        .side-by-side-container {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .file-panel {
            flex: 1;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            overflow: hidden;
        }
        .file-panel h4 {
            background: #e9ecef;
            margin: 0;
            padding: 10px 15px;
            border-bottom: 1px solid #dee2e6;
        }
        .source-code {
            max-height: 600px;
            overflow-y: auto;
            font-size: 13px;
        }
        .view-tabs {
            display: flex;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
            margin-bottom: 20px;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border: none;
            background: none;
            border-bottom: 3px solid transparent;
        }
        .tab.active {
            border-bottom-color: #007bff;
            background: white;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .fragment-summary {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .diff-view {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
        }
        .diff-add { background-color: #d4edda; color: #155724; }
        .diff-remove { background-color: #f8d7da; color: #721c24; }
        .diff-header { background-color: #d1ecf1; color: #0c5460; font-weight: bold; }
        .diff-context { color: #6c757d; }
        .performance-stats {
            background: #e8f5e8;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
    </style>
    """
    
    # JavaScript para interactividad
    javascript = """
    <script>
        function showTab(tabName, element) {
            // Ocultar todos los contenidos de tabs
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Desactivar todos los tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Mostrar el contenido seleccionado
            document.getElementById(tabName).classList.add('active');
            element.classList.add('active');
        }
        
        function toggleFragmentDetails(fragmentId) {
            const details = document.getElementById('fragment-details-' + fragmentId);
            details.style.display = details.style.display === 'none' ? 'block' : 'none';
        }
    </script>
    """
    
    # Crear archivo HTML principal
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, 'w', encoding='utf-8') as index_file:
        # Escribir encabezado
        index_file.write(f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Detector de Similitudes - Visualización Avanzada</title>
            {enhanced_css}
            {javascript}
        </head>
        <body>
            <div class="header">
                <h1>etup-ssl.shDetector de Similitudes de Código</h1>
                <p>Visualización avanzada de fragmentos similares - Lenguaje: <strong>{language}</strong></p>
            </div>
        """)
        
        # Estadísticas generales
        similarity_results = results.get('similarity_results', [])
        detailed_pairs = results.get('detailed_pairs', [])
        performance_stats = results.get('performance_stats', {})
        
        index_file.write(f"""
            <div class="stats">
                <div class="stat-card">
                    <h3>Entregas Analizadas</h3>
                    <p><strong>{results.get('num_submissions', 0)}</strong> entregas</p>
                </div>
                <div class="stat-card">
                    <h3>Archivos Procesados</h3>
                    <p><strong>{results.get('num_files_analyzed', 0)}</strong> archivos</p>
                </div>
                <div class="stat-card">
                    <h3>Pares Sospechosos</h3>
                    <p><strong>{len(detailed_pairs)}</strong> pares encontrados</p>
                </div>
                <div class="stat-card">
                    <h3>Tiempo de Análisis</h3>
                    <p><strong>{performance_stats.get('total_time', 0):.2f}s</strong></p>
                </div>
            </div>
        """)
        
        print(f"Generando visualización para {len(detailed_pairs)} pares sospechosos...")
        
        # Procesar cada par de archivos similares
        for i, pair in enumerate(detailed_pairs[:20]):  # Limitar a 20 para rendimiento
            similarity = pair['combined_similarity']
            
            # Determinar clase de similitud
            if similarity > 0.8:
                sim_class = "similarity-high"
                sim_text = "ALTA"
            elif similarity > 0.6:
                sim_class = "similarity-medium"
                sim_text = "MEDIA"
            else:
                sim_class = "similarity-low"
                sim_text = "BAJA"
            
            index_file.write(f"""
            <div class="submission-pair">
                <div class="pair-header">
                    <h2>🔍 Comparación #{i+1}</h2>
                    <p><strong>Archivo 1:</strong> {pair['file1']['submission']} - {pair['file1']['path']}</p>
                    <p><strong>Archivo 2:</strong> {pair['file2']['submission']} - {pair['file2']['path']}</p>
                    <span class="similarity-badge {sim_class}">
                        Similitud {sim_text}: {similarity:.2%}
                    </span>
                    {f'<span class="similarity-badge similarity-high">PLAGIO DETECTADO</span>' if pair.get('is_plagiarism') else ''}
                </div>
                
                <div class="file-comparison">
                    <div class="view-tabs">
                        <button class="tab active" onclick="showTab('side-by-side-{i}', this)">Vista Lado a Lado</button>
                        <button class="tab" onclick="showTab('diff-view-{i}', this)">Vista de Diferencias</button>
                        <button class="tab" onclick="showTab('fragments-{i}', this)">Fragmentos Detallados</button>
                    </div>
                    
                    <div id="side-by-side-{i}" class="tab-content active">
            """)
            
            # Vista lado a lado
            file1_content = pair['file1'].get('content', '')
            file2_content = pair['file2'].get('content', '')
            fragments1 = pair.get('similar_fragments', [])
            fragments2 = [f for f in fragments1]  # Los mismos fragmentos para ambos archivos
            
            side_by_side = create_side_by_side_view(file1_content, file2_content, language, fragments1, fragments2)
            index_file.write(side_by_side)
            
            index_file.write(f"""
                    </div>
                    
                    <div id="diff-view-{i}" class="tab-content">
                        <div class="fragment-summary">
                            <h4>📋 Resumen de Diferencias</h4>
                            <p>Similitud de tokens: <strong>{pair['token_similarity']:.2%}</strong></p>
                            <p>Similitud ML: <strong>{pair['ml_similarity']:.2%}</strong></p>
                            <p>Fragmentos encontrados: <strong>{len(fragments1)}</strong></p>
                        </div>
                        {generate_diff_view(file1_content, file2_content)}
                    </div>
                    
                    <div id="fragments-{i}" class="tab-content">
            """)
            
            # Fragmentos detallados
            for j, fragment in enumerate(fragments1[:5]):  # Limitar a 5 fragmentos
                index_file.write(f"""
                <div class="fragment-summary">
                    <h4>🔍 Fragmento Similar #{j+1}</h4>
                    <p>Similitud: <strong>{fragment.get('similarity', 0):.2%}</strong></p>
                    <p>Líneas en Archivo 1: {fragment.get('fragment1', {}).get('start_line', 0)} - {fragment.get('fragment1', {}).get('end_line', 0)}</p>
                    <p>Líneas en Archivo 2: {fragment.get('fragment2', {}).get('start_line', 0)} - {fragment.get('fragment2', {}).get('end_line', 0)}</p>
                    <button onclick="toggleFragmentDetails('{i}-{j}')">Ver Detalles</button>
                    <div id="fragment-details-{i}-{j}" style="display: none; margin-top: 10px;">
                        <div class="side-by-side-container">
                            <div class="file-panel">
                                <h5>Fragmento en Archivo 1</h5>
                                <pre>{fragment.get('fragment1', {}).get('code', 'No disponible')}</pre>
                            </div>
                            <div class="file-panel">
                                <h5>Fragmento en Archivo 2</h5>
                                <pre>{fragment.get('fragment2', {}).get('code', 'No disponible')}</pre>
                            </div>
                        </div>
                    </div>
                </div>
                """)
            
            index_file.write("""
                    </div>
                </div>
            </div>
            """)
        
        # Estadísticas de rendimiento
        if performance_stats:
            index_file.write(f"""
            <div class="performance-stats">
                <h3>📈 Estadísticas de Rendimiento</h3>
                <div class="stats">
                    <div class="stat-card">
                        <h4>Tiempo Total</h4>
                        <p>{performance_stats.get('total_time', 0):.2f}s</p>
                    </div>
                    <div class="stat-card">
                        <h4>Comparaciones</h4>
                        <p>{performance_stats.get('total_comparisons', 0)}</p>
                    </div>
                    <div class="stat-card">
                        <h4>Cache Hits</h4>
                        <p>{performance_stats.get('cache_hits', 0)}</p>
                    </div>
                    <div class="stat-card">
                        <h4>Archivos Procesados</h4>
                        <p>{performance_stats.get('files_processed', 0)}</p>
                    </div>
                </div>
            </div>
            """)
        
        # Cerrar HTML
        index_file.write("""
        </body>
        </html>
        """)
    
    print(f"Visualización avanzada generada en {index_path}")
    print(f"Abrir en navegador para ver la visualización interactiva")
    return index_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualiza fragmentos de código similares con vista avanzada")
    parser.add_argument("results_file", help="Archivo JSON con los resultados de similitud")
    parser.add_argument("--output", "-o", default="visualization", help="Directorio de salida")
    parser.add_argument("--language", "-l", default="python", help="Lenguaje de programación")
    
    args = parser.parse_args()
    visualize_fragments(args.results_file, args.output, args.language)
