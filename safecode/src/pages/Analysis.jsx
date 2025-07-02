"use client"

import { useState } from "react"
import { useDropzone } from "react-dropzone"
import {
  Upload,
  FileCode,
  AlertCircle,
  CheckCircle,
  Loader2,
  Code2,
  FileArchive,
  Trash2,
  ArrowRight,
  GitCompare,
  FileText,
  Info,
} from "lucide-react"
import api from "../service/api" 

function Analysis({ theme }) {
  const [language, setLanguage] = useState("")
  const [files, setFiles] = useState([])
  const [fileNames, setFileNames] = useState([])
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [modalMessage, setModalMessage] = useState("")
  const [modalType, setModalType] = useState("info") // 'info', 'success', 'error'
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [analysisResults, setAnalysisResults] = useState(null) // State for results

  const isDark = theme === "dracula"

  const handleLanguageChange = (event) => {
    setLanguage(event.target.value)
  }

  const onDrop = (acceptedFiles) => {
    const zipFiles = acceptedFiles.filter((file) => file.name.endsWith(".zip"))
    setFiles((prevFiles) => [...prevFiles, ...zipFiles])
    setFileNames((prevNames) => [...prevNames, ...zipFiles.map((file) => file.name)])
  }

  const removeFile = (index) => {
    setFiles((prevFiles) => prevFiles.filter((_, i) => i !== index))
    setFileNames((prevNames) => prevNames.filter((_, i) => i !== index))
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    if (!language || files.length < 2) {
      setModalMessage("Por favor, selecciona un lenguaje y carga al menos dos archivos ZIP.")
      setModalType("error")
      setIsModalOpen(true)
      return
    }

    setIsSubmitting(true)
    setAnalysisResults(null) // Clear previous results

    try {
      const formData = new FormData()
      formData.append("language", language)
      files.forEach((file) => {
        formData.append("files", file)
      })

      const response = await api.post("/upload/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })

      console.log("Respuesta del servidor:", response.data)
      if (response.data && response.data.result_summary) {
        setAnalysisResults(response.data.result_summary)
        setModalMessage("Análisis completado. Revisa los resultados a continuación.")
        setModalType("success")
      } else {
        setModalMessage("El análisis se completó, pero no se recibieron resultados detallados.")
        setModalType("info")
      }
      setIsModalOpen(true)
      setIsSubmitting(false)
      // setFiles([]);
      // setFileNames([]);
    } catch (error) {
      console.error("Error al subir los archivos:", error)
      setModalMessage(
        `Ocurrió un error al subir los archivos: ${error.response?.data?.detail || error.message}. Por favor, inténtalo de nuevo.`,
      )
      setModalType("error")
      setIsModalOpen(true)
      setIsSubmitting(false)
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/zip": [".zip"],
    },
    multiple: true,
  })

  // Helper to format similarity percentage
  const formatSimilarity = (similarity) => {
    return (similarity * 100).toFixed(2)
  }

  // Componente para mostrar comparación de archivos estilo GitKraken
  const FileComparisonCard = ({ pair, isDark, formatSimilarity }) => {
    const [selectedFragment, setSelectedFragment] = useState(0)
    const [viewMode, setViewMode] = useState("side-by-side") // 'side-by-side' o 'unified'

    const getSimilarityColor = (similarity) => {
      if (similarity > 0.8) return isDark ? "text-red-400" : "text-red-600"
      if (similarity > 0.6) return isDark ? "text-orange-400" : "text-orange-600"
      if (similarity > 0.4) return isDark ? "text-yellow-400" : "text-yellow-600"
      return isDark ? "text-green-400" : "text-green-600"
    }

    const renderHighlightedCode = (content, highlights, fileIndex) => {
      if (!highlights || highlights.length === 0) {
        return (
          <pre className={`text-xs leading-relaxed ${isDark ? "text-gray-300" : "text-gray-800"}`}>
            <code>{content}</code>
          </pre>
        )
      }

      const lines = content.split("\n")
      const highlightMap = new Map()

      highlights.forEach((highlight) => {
        const lineNum = highlight.line - 1 // Convert to 0-based
        if (lineNum >= 0 && lineNum < lines.length) {
          highlightMap.set(lineNum, highlight)
        }
      })

      return (
        <div className="text-xs leading-relaxed font-mono">
          {lines.map((line, lineIndex) => {
            const highlight = highlightMap.get(lineIndex)
            const isHighlighted = highlight !== undefined

            return (
              <div
                key={lineIndex}
                className={`flex ${
                  isHighlighted
                    ? fileIndex === 0
                      ? isDark
                        ? "bg-blue-900/30 border-l-4 border-blue-400"
                        : "bg-blue-50 border-l-4 border-blue-500"
                      : isDark
                        ? "bg-purple-900/30 border-l-4 border-purple-400"
                        : "bg-purple-50 border-l-4 border-purple-500"
                    : ""
                }`}
              >
                <span
                  className={`inline-block w-12 text-right pr-2 select-none ${
                    isDark ? "text-gray-500" : "text-gray-400"
                  }`}
                >
                  {lineIndex + 1}
                </span>
                <span
                  className={`flex-1 pl-2 ${
                    isHighlighted
                      ? isDark
                        ? "text-white font-medium"
                        : "text-gray-900 font-medium"
                      : isDark
                        ? "text-gray-300"
                        : "text-gray-700"
                  }`}
                >
                  {line || " "}
                </span>
              </div>
            )
          })}
        </div>
      )
    }

    return (
      <div
        className={`rounded-2xl border shadow-xl overflow-hidden ${
          isDark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"
        }`}
      >
        {/* Header */}
        <div className={`p-6 border-b ${isDark ? "border-gray-700 bg-gray-750" : "border-gray-200 bg-gray-50"}`}>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-4">
              <GitCompare className="w-6 h-6 text-blue-500" />
              <div>
                <h3 className={`text-lg font-semibold ${isDark ? "text-white" : "text-gray-900"}`}>
                  Comparación de Archivos
                </h3>
                <p className={`text-sm ${isDark ? "text-gray-400" : "text-gray-600"}`}>
                  {pair.file1.submission} vs {pair.file2.submission}
                </p>
              </div>
            </div>

            {/* Similarity Badge */}
            <div className="flex items-center space-x-3">
              <span
                className={`px-3 py-1 rounded-full text-sm font-medium ${
                  pair.is_plagiarism
                    ? "bg-red-100 text-red-800 border border-red-200"
                    : "bg-green-100 text-green-800 border border-green-200"
                }`}
              >
                {pair.is_plagiarism ? "Plagio Detectado" : "Sin Plagio"}
              </span>
              <span className={`text-2xl font-bold ${getSimilarityColor(pair.combined_similarity)}`}>
                {formatSimilarity(pair.combined_similarity)}%
              </span>
            </div>
          </div>

          {/* File Names */}
          <div className="grid grid-cols-2 gap-4">
            <div className={`p-3 rounded-lg ${isDark ? "bg-blue-900/20" : "bg-blue-50"}`}>
              <FileText className="w-4 h-4 text-blue-500 mb-1" />
              <p className={`font-medium ${isDark ? "text-blue-300" : "text-blue-700"}`}>{pair.file1.path}</p>
              <p className={`text-xs ${isDark ? "text-gray-400" : "text-gray-600"}`}>
                {pair.file1.file_size} bytes • {pair.file1.token_count} tokens
              </p>
            </div>
            <div className={`p-3 rounded-lg ${isDark ? "bg-purple-900/20" : "bg-purple-50"}`}>
              <FileText className="w-4 h-4 text-purple-500 mb-1" />
              <p className={`font-medium ${isDark ? "text-purple-300" : "text-purple-700"}`}>{pair.file2.path}</p>
              <p className={`text-xs ${isDark ? "text-gray-400" : "text-gray-600"}`}>
                {pair.file2.file_size} bytes • {pair.file2.token_count} tokens
              </p>
            </div>
          </div>

          {/* Similarity Metrics */}
          <div className="mt-4 grid grid-cols-3 gap-4">
            <div className={`text-center p-2 rounded ${isDark ? "bg-gray-700" : "bg-gray-100"}`}>
              <p className={`text-sm font-medium ${isDark ? "text-gray-300" : "text-gray-700"}`}>Token Similarity</p>
              <p className={`text-lg font-bold ${getSimilarityColor(pair.token_similarity)}`}>
                {formatSimilarity(pair.token_similarity)}%
              </p>
            </div>
            <div className={`text-center p-2 rounded ${isDark ? "bg-gray-700" : "bg-gray-100"}`}>
              <p className={`text-sm font-medium ${isDark ? "text-gray-300" : "text-gray-700"}`}>ML Similarity</p>
              <p className={`text-lg font-bold ${getSimilarityColor(pair.ml_similarity)}`}>
                {formatSimilarity(pair.ml_similarity)}%
              </p>
            </div>
            <div className={`text-center p-2 rounded ${isDark ? "bg-gray-700" : "bg-gray-100"}`}>
              <p className={`text-sm font-medium ${isDark ? "text-gray-300" : "text-gray-700"}`}>Probabilidad</p>
              <p className={`text-lg font-bold ${getSimilarityColor(pair.plagiarism_probability)}`}>
                {formatSimilarity(pair.plagiarism_probability)}%
              </p>
            </div>
          </div>
        </div>

        {/* Fragment Navigation */}
        {pair.similar_fragments && pair.similar_fragments.length > 0 && (
          <div className={`p-4 border-b ${isDark ? "border-gray-700 bg-gray-800" : "border-gray-200 bg-gray-50"}`}>
            <div className="flex items-center justify-between mb-3">
              <h4 className={`font-medium ${isDark ? "text-white" : "text-gray-900"}`}>
                Fragmentos Similares ({pair.similar_fragments.length})
              </h4>
              <div className="flex space-x-2">
                <button
                  onClick={() => setViewMode("side-by-side")}
                  className={`px-3 py-1 rounded text-sm ${
                    viewMode === "side-by-side"
                      ? "bg-blue-500 text-white"
                      : isDark
                        ? "bg-gray-700 text-gray-300"
                        : "bg-gray-200 text-gray-700"
                  }`}
                >
                  Lado a Lado
                </button>
                <button
                  onClick={() => setViewMode("unified")}
                  className={`px-3 py-1 rounded text-sm ${
                    viewMode === "unified"
                      ? "bg-blue-500 text-white"
                      : isDark
                        ? "bg-gray-700 text-gray-300"
                        : "bg-gray-200 text-gray-700"
                  }`}
                >
                  Unificado
                </button>
              </div>
            </div>

            <div className="flex space-x-2 overflow-x-auto">
              {pair.similar_fragments.map((fragment, index) => (
                <button
                  key={index}
                  onClick={() => setSelectedFragment(index)}
                  className={`px-3 py-2 rounded text-sm whitespace-nowrap ${
                    selectedFragment === index
                      ? "bg-blue-500 text-white"
                      : isDark
                        ? "bg-gray-700 text-gray-300 hover:bg-gray-600"
                        : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                  }`}
                >
                  Fragmento {index + 1} ({formatSimilarity(fragment.similarity)}%)
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Code Comparison */}
        {pair.similar_fragments && pair.similar_fragments.length > 0 && (
          <div className="p-6">
            {viewMode === "side-by-side" ? (
              <div className="grid grid-cols-2 gap-6">
                {/* File 1 */}
                <div className={`rounded-lg border ${isDark ? "border-gray-600" : "border-gray-300"}`}>
                  <div
                    className={`px-4 py-2 border-b ${isDark ? "border-gray-600 bg-blue-900/20" : "border-gray-300 bg-blue-50"}`}
                  >
                    <p className={`text-sm font-medium ${isDark ? "text-blue-300" : "text-blue-700"}`}>
                      {pair.file1.path}
                    </p>
                    <p className={`text-xs ${isDark ? "text-gray-400" : "text-gray-600"}`}>
                      Líneas {pair.similar_fragments[selectedFragment]?.fragment1?.start_line} -{" "}
                      {pair.similar_fragments[selectedFragment]?.fragment1?.end_line}
                    </p>
                  </div>
                  <div className={`p-4 overflow-x-auto max-h-96 ${isDark ? "bg-gray-900" : "bg-gray-50"}`}>
                    {renderHighlightedCode(
                      pair.file1.content,
                      pair.similar_fragments[selectedFragment]?.fragment1?.highlights,
                      0,
                    )}
                  </div>
                </div>

                {/* File 2 */}
                <div className={`rounded-lg border ${isDark ? "border-gray-600" : "border-gray-300"}`}>
                  <div
                    className={`px-4 py-2 border-b ${isDark ? "border-gray-600 bg-purple-900/20" : "border-gray-300 bg-purple-50"}`}
                  >
                    <p className={`text-sm font-medium ${isDark ? "text-purple-300" : "text-purple-700"}`}>
                      {pair.file2.path}
                    </p>
                    <p className={`text-xs ${isDark ? "text-gray-400" : "text-gray-600"}`}>
                      Líneas {pair.similar_fragments[selectedFragment]?.fragment2?.start_line} -{" "}
                      {pair.similar_fragments[selectedFragment]?.fragment2?.end_line}
                    </p>
                  </div>
                  <div className={`p-4 overflow-x-auto max-h-96 ${isDark ? "bg-gray-900" : "bg-gray-50"}`}>
                    {renderHighlightedCode(
                      pair.file2.content,
                      pair.similar_fragments[selectedFragment]?.fragment2?.highlights,
                      1,
                    )}
                  </div>
                </div>
              </div>
            ) : (
              // Unified view
              <div className={`rounded-lg border ${isDark ? "border-gray-600" : "border-gray-300"}`}>
                <div
                  className={`px-4 py-2 border-b ${isDark ? "border-gray-600 bg-gray-700" : "border-gray-300 bg-gray-100"}`}
                >
                  <p className={`text-sm font-medium ${isDark ? "text-white" : "text-gray-900"}`}>
                    Vista Unificada - Fragmento {selectedFragment + 1}
                  </p>
                </div>
                <div className="grid grid-cols-2 gap-0">
                  <div
                    className={`p-4 border-r ${isDark ? "border-gray-600 bg-blue-900/10" : "border-gray-300 bg-blue-50/50"}`}
                  >
                    <p className={`text-xs font-medium mb-2 ${isDark ? "text-blue-300" : "text-blue-700"}`}>
                      {pair.file1.path}
                    </p>
                    <div className="overflow-x-auto max-h-96">
                      {renderHighlightedCode(
                        pair.file1.content,
                        pair.similar_fragments[selectedFragment]?.fragment1?.highlights,
                        0,
                      )}
                    </div>
                  </div>
                  <div className={`p-4 ${isDark ? "bg-purple-900/10" : "bg-purple-50/50"}`}>
                    <p className={`text-xs font-medium mb-2 ${isDark ? "text-purple-300" : "text-purple-700"}`}>
                      {pair.file2.path}
                    </p>
                    <div className="overflow-x-auto max-h-96">
                      {renderHighlightedCode(
                        pair.file2.content,
                        pair.similar_fragments[selectedFragment]?.fragment2?.highlights,
                        1,
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  return (
    <div
      className={`min-h-screen transition-colors duration-300 ${
        isDark
          ? "bg-gradient-to-br from-gray-900 via-gray-800 to-indigo-900"
          : "bg-gradient-to-br from-blue-50 via-white to-indigo-50"
      }`}
    >
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div
          className={`absolute inset-0 ${
            isDark
              ? "bg-gradient-to-r from-blue-600/20 to-purple-600/20"
              : "bg-gradient-to-r from-blue-600/10 to-purple-600/10"
          }`}
        ></div>
        <div className="relative max-w-7xl mx-auto px-6 py-16">
          <div className="text-center">
            <div className="flex items-center justify-center mb-6">
              <Code2 className="w-12 h-12 text-blue-600 mr-4" />
              <span
                className={`text-sm font-medium px-3 py-1 rounded-full ${
                  isDark ? "bg-purple-900/50 text-purple-300 border border-purple-700" : "bg-purple-100 text-purple-800"
                }`}
              >
                Detección de Similitudes
              </span>
            </div>
            <h1 className={`text-4xl md:text-5xl font-bold mb-6 ${isDark ? "text-white" : "text-gray-900"}`}>
              Análisis de{" "}
              <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Proyectos
              </span>
            </h1>
            <p
              className={`text-xl mb-8 max-w-3xl mx-auto leading-relaxed ${isDark ? "text-gray-300" : "text-gray-700"}`}
            >
              Carga tus proyectos en formato ZIP y nuestro sistema analizará las similitudes entre ellos utilizando
              técnicas avanzadas de machine learning.
            </p>
          </div>
        </div>
      </div>

      {/* Analysis Form Section */}
      <div className="max-w-6xl mx-auto px-6 py-8">
        <div className="grid lg:grid-cols-3 gap-12">
          {/* Left Column - Instructions */}
          <div className="lg:col-span-1 space-y-8">
            <div
              className={`rounded-2xl p-8 border ${isDark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"} shadow-lg`}
            >
              <h2 className={`text-2xl font-bold mb-6 ${isDark ? "text-white" : "text-gray-900"}`}>Cómo Funciona</h2>
              <div className="space-y-6">
                {[
                  { title: "Selecciona el Lenguaje", desc: "Elige el lenguaje de programación de tus proyectos." },
                  {
                    title: "Carga los Archivos",
                    desc: "Sube al menos dos archivos ZIP que contengan los proyectos a comparar.",
                  },
                  {
                    title: "Revisa los Resultados",
                    desc: "El sistema procesará los archivos y mostrará un informe detallado de similitudes.",
                  },
                ].map((step, index) => (
                  <div className="flex items-start" key={index}>
                    <div
                      className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${
                        isDark ? "bg-blue-600" : "bg-blue-500"
                      }`}
                    >
                      {index + 1}
                    </div>
                    <div className="ml-4">
                      <h3 className={`font-semibold mb-1 ${isDark ? "text-white" : "text-gray-900"}`}>{step.title}</h3>
                      <p className={isDark ? "text-gray-300" : "text-gray-600"}>{step.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div
              className={`rounded-2xl p-6 border ${isDark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"} shadow-lg`}
            >
              <h3 className={`text-lg font-semibold mb-4 ${isDark ? "text-white" : "text-gray-900"}`}>
                Lenguajes Soportados
              </h3>
              <div className="grid grid-cols-2 gap-3">
                {[
                  {
                    name: "Python",
                    colorClass: isDark ? "text-blue-400" : "text-blue-700",
                    bgClass: isDark ? "bg-gray-700" : "bg-blue-50",
                  },
                  {
                    name: "Java",
                    colorClass: isDark ? "text-green-400" : "text-green-700",
                    bgClass: isDark ? "bg-gray-700" : "bg-green-50",
                  },
                  {
                    name: "C++",
                    colorClass: isDark ? "text-purple-400" : "text-purple-700",
                    bgClass: isDark ? "bg-gray-700" : "bg-purple-50",
                  },
                  {
                    name: "JavaScript",
                    colorClass: isDark ? "text-amber-400" : "text-amber-700",
                    bgClass: isDark ? "bg-gray-700" : "bg-amber-50",
                  },
                ].map((lang) => (
                  <div
                    key={lang.name}
                    className={`p-3 rounded-lg flex items-center ${lang.bgClass} ${lang.colorClass}`}
                  >
                    <FileCode className="w-5 h-5 mr-2" />
                    <span>{lang.name}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column - Form */}
          <div className="lg:col-span-2">
            <div
              className={`rounded-2xl shadow-xl p-8 border ${isDark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}
            >
              <form onSubmit={handleSubmit} className="space-y-8">
                <div>
                  <label className={`block text-lg font-medium mb-3 ${isDark ? "text-white" : "text-gray-900"}`}>
                    <FileCode className="w-5 h-5 inline mr-2" />
                    Selecciona el Lenguaje de Programación
                  </label>
                  <select
                    value={language}
                    onChange={handleLanguageChange}
                    className={`w-full px-4 py-3 rounded-lg border transition-colors duration-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                      isDark ? "bg-gray-700 border-gray-600 text-white" : "bg-white border-gray-300 text-gray-900"
                    }`}
                    required
                  >
                    <option value="" disabled>
                      Selecciona un lenguaje
                    </option>
                    <option value="python">Python</option>
                    <option value="java">Java</option>
                    <option value="cpp">C++</option>
                    <option value="javascript">JavaScript</option>
                  </select>
                </div>

                <div>
                  <label className={`block text-lg font-medium mb-3 ${isDark ? "text-white" : "text-gray-900"}`}>
                    <FileArchive className="w-5 h-5 inline mr-2" />
                    Carga los Archivos ZIP
                  </label>
                  <div
                    {...getRootProps()}
                    className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors duration-300 ${
                      isDragActive
                        ? isDark
                          ? "border-blue-500 bg-blue-500/10"
                          : "border-blue-500 bg-blue-50"
                        : isDark
                          ? "border-gray-600 hover:border-gray-500"
                          : "border-gray-300 hover:border-gray-400"
                    }`}
                  >
                    <input {...getInputProps()} />
                    <Upload
                      className={`w-12 h-12 mx-auto mb-4 ${
                        isDragActive ? "text-blue-500" : isDark ? "text-gray-400" : "text-gray-500"
                      }`}
                    />
                    {isDragActive ? (
                      <p className={`text-lg font-medium ${isDark ? "text-blue-400" : "text-blue-600"}`}>
                        Suelta los archivos aquí...
                      </p>
                    ) : (
                      <>
                        <p className={`text-lg font-medium mb-2 ${isDark ? "text-gray-300" : "text-gray-700"}`}>
                          Arrastra y suelta tus archivos aquí
                        </p>
                        <p className={isDark ? "text-gray-400" : "text-gray-500"}>o haz clic para seleccionarlos</p>
                        <p className={`text-sm mt-3 ${isDark ? "text-gray-500" : "text-gray-400"}`}>
                          Solo se aceptan archivos .zip
                        </p>
                      </>
                    )}
                  </div>
                </div>

                {fileNames.length > 0 && (
                  <div
                    className={`rounded-lg border p-4 ${
                      isDark ? "bg-gray-700/50 border-gray-600" : "bg-gray-50 border-gray-200"
                    }`}
                  >
                    <h3 className={`text-lg font-medium mb-3 ${isDark ? "text-white" : "text-gray-900"}`}>
                      Archivos seleccionados ({fileNames.length})
                    </h3>
                    <ul className="space-y-2">
                      {fileNames.map((name, index) => (
                        <li
                          key={index}
                          className={`flex justify-between items-center p-3 rounded-lg ${
                            isDark ? "bg-gray-800" : "bg-white"
                          }`}
                        >
                          <div className="flex items-center">
                            <FileArchive className={`w-5 h-5 mr-3 ${isDark ? "text-blue-400" : "text-blue-600"}`} />
                            <span className={isDark ? "text-gray-300" : "text-gray-700"}>{name}</span>
                          </div>
                          <button
                            type="button"
                            onClick={() => removeFile(index)}
                            className={`p-1.5 rounded-full transition-colors duration-300 ${
                              isDark
                                ? "text-red-400 hover:bg-red-900/30 hover:text-red-300"
                                : "text-red-500 hover:bg-red-50 hover:text-red-700"
                            }`}
                            aria-label="Eliminar archivo"
                          >
                            <Trash2 className="w-5 h-5" />
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <button
                  type="submit"
                  disabled={isSubmitting}
                  className={`w-full py-4 px-6 rounded-lg font-semibold text-lg transition-all duration-300 transform hover:scale-105 shadow-lg flex items-center justify-center ${
                    isSubmitting
                      ? isDark
                        ? "bg-gray-600 text-gray-400 cursor-not-allowed"
                        : "bg-gray-300 text-gray-500 cursor-not-allowed"
                      : `bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white`
                  }`}
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                      Procesando...
                    </>
                  ) : (
                    <>
                      Iniciar Análisis
                      <ArrowRight className="w-5 h-5 ml-2" />
                    </>
                  )}
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>

      {/* Results Section */}
      {analysisResults && (
        <div className="max-w-7xl mx-auto px-6 py-12">
          <h2 className={`text-3xl md:text-4xl font-bold mb-10 text-center ${isDark ? "text-white" : "text-gray-900"}`}>
            Resultados del Análisis
          </h2>

          {/* Performance Stats */}
          {analysisResults.performance_stats && (
            <div
              className={`rounded-2xl p-6 border mb-8 ${isDark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"} shadow-lg`}
            >
              <h3 className={`text-xl font-semibold mb-4 flex items-center ${isDark ? "text-white" : "text-gray-900"}`}>
                <Info className="w-6 h-6 mr-3 text-blue-500" />
                Estadísticas de Rendimiento
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div className={`p-3 rounded-lg ${isDark ? "bg-gray-700" : "bg-gray-50"}`}>
                  <p className={`font-semibold ${isDark ? "text-green-400" : "text-green-600"}`}>
                    {analysisResults.performance_stats.total_time?.toFixed(2)}s
                  </p>
                  <p className={isDark ? "text-gray-300" : "text-gray-600"}>Tiempo Total</p>
                </div>
                <div className={`p-3 rounded-lg ${isDark ? "bg-gray-700" : "bg-gray-50"}`}>
                  <p className={`font-semibold ${isDark ? "text-blue-400" : "text-blue-600"}`}>
                    {analysisResults.performance_stats.files_processed}
                  </p>
                  <p className={isDark ? "text-gray-300" : "text-gray-600"}>Archivos Procesados</p>
                </div>
                <div className={`p-3 rounded-lg ${isDark ? "bg-gray-700" : "bg-gray-50"}`}>
                  <p className={`font-semibold ${isDark ? "text-purple-400" : "text-purple-600"}`}>
                    {analysisResults.performance_stats.cache_hits}
                  </p>
                  <p className={isDark ? "text-gray-300" : "text-gray-600"}>Cache Hits</p>
                </div>
                <div className={`p-3 rounded-lg ${isDark ? "bg-gray-700" : "bg-gray-50"}`}>
                  <p className={`font-semibold ${isDark ? "text-red-400" : "text-red-600"}`}>
                    {analysisResults.performance_stats.high_similarity_pairs}
                  </p>
                  <p className={isDark ? "text-gray-300" : "text-gray-600"}>Pares Sospechosos</p>
                </div>
              </div>
            </div>
          )}

          {/* File Comparisons - GitKraken Style */}
          {analysisResults.detailed_pairs && analysisResults.detailed_pairs.length > 0 ? (
            <div className="space-y-8">
              {analysisResults.detailed_pairs.map((pair, pairIndex) => (
                <FileComparisonCard key={pairIndex} pair={pair} isDark={isDark} formatSimilarity={formatSimilarity} />
              ))}
            </div>
          ) : (
            <div
              className={`rounded-2xl p-6 md:p-8 border ${isDark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"} shadow-lg text-center`}
            >
              <Info className={`w-12 h-12 mx-auto mb-4 ${isDark ? "text-blue-400" : "text-blue-500"}`} />
              <p className={`text-lg ${isDark ? "text-gray-300" : "text-gray-700"}`}>
                No se encontraron similitudes significativas en este análisis.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 flex items-center justify-center p-4 bg-black/50 z-50">
          <div
            className={`p-6 rounded-2xl shadow-2xl w-full max-w-md transform transition-all duration-300 ${
              isDark ? "bg-gray-800" : "bg-white"
            }`}
          >
            <div className="flex items-center mb-4">
              {modalType === "success" ? (
                <CheckCircle className="w-8 h-8 text-green-500 mr-3" />
              ) : modalType === "error" ? (
                <AlertCircle className="w-8 h-8 text-red-500 mr-3" />
              ) : (
                <AlertCircle className="w-8 h-8 text-blue-500 mr-3" />
              )}
              <h3 className={`text-lg font-semibold ${isDark ? "text-white" : "text-gray-900"}`}>
                {modalType === "success" ? "¡Éxito!" : modalType === "error" ? "Error" : "Aviso"}
              </h3>
            </div>
            <p className={`mb-6 ${isDark ? "text-gray-300" : "text-gray-600"}`}>{modalMessage}</p>
            <button
              onClick={() => setIsModalOpen(false)}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-300"
            >
              Aceptar
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default Analysis
