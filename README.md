# Sistema de Detección de Copias de Código

Un sistema avanzado de detección de plagio en código fuente que utiliza técnicas determinísticas y machine learning para identificar similitudes entre entregas de estudiantes.

## 📋 Índice

- [Características Principales](#-características-principales)
- [Requisitos](#-requisitos)
- [Inicio Rápido](#️-inicio-rápido)
  - [1. Iniciar el Servidor](#1-iniciar-el-servidor)
  - [2. Procesar Entregas](#2-procesar-entregas)
- [Flujo de Trabajo Principal](#-flujo-de-trabajo-principal)
  - [1️⃣ PROCESAR: Subir y Analizar Entregas](#1️⃣-procesar-subir-y-analizar-entregas)
  - [2️⃣ REVISAR: Examinar Fragmentos Sospechosos](#2️⃣-revisar-examinar-fragmentos-sospechosos)
  - [3️⃣ MARCAR: Añadir al Ground Truth](#3️⃣-marcar-añadir-al-ground-truth)
  - [4️⃣ ENTRENAR: Mejorar el Modelo con Feedback](#4️⃣-entrenar-mejorar-el-modelo-con-feedback)
  - [5️⃣ MEJORAR: Optimizar Parámetros](#5️⃣-mejorar-optimizar-parámetros)
  - [6️⃣ REPETIR: Procesar Nuevas Entregas con Sistema Mejorado](#6️⃣-repetir-procesar-nuevas-entregas-con-sistema-mejorado)
- [Guía de Comandos Detallada](#-guía-de-comandos-detallada)
  - [🎯 Gestión de Ground Truth](#-gestión-de-ground-truth)
  - [🧠 Aprendizaje Adaptativo](#-aprendizaje-adaptativo)
  - [📊 Evaluación del Sistema](#-evaluación-del-sistema)
- [Métricas y Evaluación](#-métricas-y-evaluación)
  - [Tipos de Plagio](#tipos-de-plagio)
  - [Métricas de Similitud](#métricas-de-similitud)
- [Consideraciones Importantes](#️-consideraciones-importantes)
  - [🎓 Decisión Final del Profesor](#-decisión-final-del-profesor)
  - [📝 Mejores Prácticas](#-mejores-prácticas)
- [Ejemplos de Casos de Uso](#-ejemplos-de-casos-de-uso)
  - [Caso 1: Detección de Plagio Exacto](#caso-1-detección-de-plagio-exacto)
  - [Caso 2: Código Similar pero Legítimo](#caso-2-código-similar-pero-legítimo)

## 🚀 Características Principales

- **Múltiples Lenguajes**: Soporte para Python, Java, C++ y JavaScript
- **Técnicas Híbridas**: Combina métodos determinísticos (Levenshtein, N-gramas, TF-IDF) con ML (CodeBERT)
- **Aprendizaje Adaptativo**: Mejora automáticamente con el feedback del profesor
- **Ground Truth**: Sistema para marcar ejemplos de plagio/no plagio
- **Clustering Inteligente**: Agrupa entregas similares automáticamente
- **Visualización Rica**: Fragmentos de código lado a lado con diferencias resaltadas

## 📋 Requisitos

```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Inicio Rápido

### 1. Iniciar el Servidor

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Procesar Entregas

```bash
curl -X POST "http://localhost:8000/api/upload/" \
  -F "files=@estudiante1.zip" \
  -F "files=@estudiante2.zip" \
  -F "files=@estudiante3.zip" \
  -F "language=python"
```

## 🔄 Flujo de Trabajo Principal

### 1️⃣ PROCESAR: Subir y Analizar Entregas

```bash
# Subir archivos ZIP de estudiantes
RESPONSE=$(curl -s -X POST "http://localhost:8000/api/upload/" \
  -F "files=@estudiante1.zip" \
  -F "files=@estudiante2.zip" \
  -F "language=python")

# Ver resultados formateados
echo $RESPONSE | jq '.'
```

**Resultado esperado:**

```json
{
  "session_id": "abc123",
  "similarity_results": [
    {
      "submission1": "estudiante1",
      "submission2": "estudiante2",
      "max_similarity": 0.92,
      "is_plagiarism": true,
      "similar_files": [
        {
          "file1": "tarea.py",
          "file2": "tarea.py",
          "combined_similarity": 0.92,
          "fragments": [
            {
              "similarity": 0.95,
              "code1": "def calcular_promedio(numeros):\n    return sum(numeros) / len(numeros)",
              "code2": "def calc_prom(nums):\n    return sum(nums) / len(nums)"
            }
          ]
        }
      ]
    }
  ],
  "detailed_pairs": [
    {
      "file1": {
        "submission": "estudiante1",
        "path": "tarea.py",
        "full_path": "/tmp/plagiarism_checker/abc123/estudiante1/tarea.py"
      },
      "file2": {
        "submission": "estudiante2",
        "path": "tarea.py",
        "full_path": "/tmp/plagiarism_checker/abc123/estudiante2/tarea.py"
      },
      "all_features": {
        "lcs": 0.85,
        "sequence_matcher": 0.88,
        "ngram": 0.82,
        "levenshtein": 0.90,
        "tfidf_cosine": 0.75,
        "winnowing": 0.80,
        "ml_tfidf": 0.78,
        "ml_char_ngram": 0.85,
        "ml_word_ngram": 0.83,
        "ml_embedding": 0.92
      }
    }
  ]
}
```

### 2️⃣ REVISAR: Examinar Fragmentos Sospechosos

#### Visualizar fragmentos de código

```bash
# Guardar resultados en un archivo
echo $RESPONSE > resultados.json

# Generar visualización HTML
python scripts/visualize_fragments.py resultados.json --output visualizacion --language python

# Abrir en navegador
open visualizacion/index.html
```

La visualización HTML mostrará:

- Fragmentos de código lado a lado
- Diferencias resaltadas
- Métricas de similitud por fragmento

### 3️⃣ MARCAR: Añadir al Ground Truth

Después de revisar manualmente, marca los pares como plagio o no plagio:

```bash
# Marcar como PLAGIO (usando las rutas completas del resultado)
curl -X POST "http://localhost:8000/api/ground-truth/" \
  -H "Content-Type: application/json" \
  -d '{
    "file1": "/tmp/plagiarism_checker/abc123/estudiante1/tarea.py",
    "file2": "/tmp/plagiarism_checker/abc123/estudiante2/tarea.py",
    "is_plagiarism": true,
    "plagiarism_type": "renamed",
    "confidence": 1.0,
    "notes": "Mismo algoritmo, solo cambiaron nombres de variables"
  }'

# Marcar como NO PLAGIO
curl -X POST "http://localhost:8000/api/ground-truth/" \
  -H "Content-Type: application/json" \
  -d '{
    "file1": "/tmp/plagiarism_checker/abc123/estudiante3/tarea.py",
    "file2": "/tmp/plagiarism_checker/abc123/estudiante4/tarea.py",
    "is_plagiarism": false,
    "confidence": 1.0,
    "notes": "Similitud alta pero es código estándar del curso"
  }'
```

### 4️⃣ ENTRENAR: Mejorar el Modelo con Feedback

```bash
# Entrenar el modelo con los ejemplos acumulados
curl -X POST "http://localhost:8000/api/adaptive-learning/train-model"
```

**Resultado esperado:**

```json
{
  "accuracy": 0.92,
  "precision": 0.89,
  "recall": 0.94,
  "f1": 0.91,
  "num_samples": 24
}
```

### 5️⃣ MEJORAR: Optimizar Parámetros

```bash
# Aplicar pesos óptimos basados en el aprendizaje
curl -X POST "http://localhost:8000/api/adaptive-learning/use-optimal-weights"
```

**Resultado esperado:**

```json
{
  "status": "success",
  "weights": {
    "token_based": {
      "lcs": 0.12,
      "sequence_matcher": 0.15,
      "ngram": 0.25,
      "levenshtein": 0.18,
      "tfidf_cosine": 0.22,
      "winnowing": 0.08
    },
    "combined": {
      "token": 0.55,
      "ml": 0.45
    }
  }
}
```

### 6️⃣ REPETIR: Procesar Nuevas Entregas con Sistema Mejorado

```bash
# El sistema ahora es más preciso con los nuevos pesos y aprendizaje
curl -X POST "http://localhost:8000/api/upload/" \
  -F "files=@estudiante5.zip" \
  -F "files=@estudiante6.zip" \
  -F "language=python"
```

## 📚 Guía de Comandos Detallada

### 🎯 Gestión de Ground Truth

#### Ver todos los pares marcados

```bash
curl -X GET "http://localhost:8000/api/ground-truth/"
```

#### Ver estadísticas del Ground Truth

```bash
curl -X GET "http://localhost:8000/api/ground-truth/stats"
```

#### Exportar Ground Truth a CSV

```bash
curl -X POST "http://localhost:8000/api/ground-truth/export" \
  -H "Content-Type: application/json" \
  -d '{"output_path": "/path/to/export"}'
```

### 🧠 Aprendizaje Adaptativo

#### Ver estadísticas del modelo

```bash
curl -X GET "http://localhost:8000/api/adaptive-learning/stats"
```

#### Ver pesos óptimos actuales

```bash
curl -X GET "http://localhost:8000/api/adaptive-learning/optimal-weights"
```

#### Ver evolución de pesos en el tiempo

```bash
curl -X GET "http://localhost:8000/api/adaptive-learning/weight-history"
```

#### Actualizar pesos manualmente

```bash
curl -X POST "http://localhost:8000/api/adaptive-learning/update-weights" \
  -H "Content-Type: application/json" \
  -d '{
    "token_weights": {
      "lcs": 0.15,
      "sequence_matcher": 0.15,
      "ngram": 0.25,
      "levenshtein": 0.20,
      "tfidf_cosine": 0.20,
      "winnowing": 0.05
    },
    "combined_weights": {
      "token": 0.60,
      "ml": 0.40
    }
  }'
```

### 📊 Evaluación del Sistema

#### Evaluar rendimiento del detector

```bash
curl -X POST "http://localhost:8000/api/ground-truth/evaluate" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "file1": "/path/to/file1.py",
      "file2": "/path/to/file2.py",
      "similarity": 0.92,
      "is_plagiarism": true
    },
    {
      "file1": "/path/to/file3.py",
      "file2": "/path/to/file4.py",
      "similarity": 0.65,
      "is_plagiarism": false
    }
  ]'
```

#### Encontrar umbral óptimo

```bash
curl -X POST "http://localhost:8000/api/ground-truth/optimize-threshold" \
  -H "Content-Type: application/json" \
  -d '[...]'  # Mismos datos que evaluate
```

## 📈 Métricas y Evaluación

### Tipos de Plagio

- **`exact`**: Código idéntico o casi idéntico
- **`renamed`**: Solo cambiaron nombres de variables/funciones
- **`restructured`**: Reorganizaron el código manteniendo la lógica
- **`logic_changed`**: Cambiaron la lógica superficialmente

### Métricas de Similitud

- **LCS**: Subsecuencia común más larga
- **Levenshtein**: Distancia de edición normalizada
- **N-gramas**: Similitud basada en secuencias de tokens
- **TF-IDF**: Similitud de vocabulario y estructura
- **CodeBERT**: Similitud semántica usando embeddings

## ⚠️ Consideraciones Importantes

### 🎓 Decisión Final del Profesor

- El sistema es una **herramienta de apoyo**, no un juez automático
- **Siempre revisa manualmente** los casos detectados
- Considera el contexto del curso y la dificultad del ejercicio

### 📝 Mejores Prácticas

1. **Comienza con umbrales conservadores** (0.7-0.8)
2. **Añade ejemplos de ground truth gradualmente**
3. **Entrena el modelo regularmente** (cada 10-20 ejemplos)
4. **Documenta tus decisiones** en las notas

## 🔍 Ejemplos de Casos de Uso

### Caso 1: Detección de Plagio Exacto

```python
# Código Original (Estudiante A)
def calcular_factorial(n):
    if n <= 1:
        return 1
    return n * calcular_factorial(n - 1)

# Código Copiado (Estudiante B)  
def factorial(num):
    if num <= 1:
        return 1
    return num * factorial(num - 1)
```

**Resultado**: Similitud 95% - PLAGIO detectado

### Caso 2: Código Similar pero Legítimo

```python
# Estudiante A
for i in range(len(lista)):
    print(lista[i])

# Estudiante B
for elemento in lista:
    print(elemento)
```

**Resultado**: Similitud 60% - NO PLAGIO (diferentes enfoques válidos)

---

**Recuerda**: Este sistema te ayuda a **identificar similitudes**, pero la **decisión final sobre plagio académico siempre debe ser tuya** como educador. Usa el sistema como una herramienta de apoyo para hacer tu trabajo más eficiente y preciso.