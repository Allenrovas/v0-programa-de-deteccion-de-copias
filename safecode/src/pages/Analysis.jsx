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
  Percent,
  Info,
} from "lucide-react"
import api from "../service/api" // Asegúrate que la ruta a tu servicio api sea correcta

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

          <div
            className={`rounded-2xl p-6 md:p-8 border mb-8 ${isDark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"} shadow-lg`}
          >
            <h3 className={`text-xl font-semibold mb-4 flex items-center ${isDark ? "text-white" : "text-gray-900"}`}>
              <Info className="w-6 h-6 mr-3 text-blue-500" />
              Resumen General
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
              <p className={isDark ? "text-gray-300" : "text-gray-700"}>
                <strong>Lenguaje:</strong> {analysisResults.language}
              </p>
              <p className={isDark ? "text-gray-300" : "text-gray-700"}>
                <strong>Proyectos Analizados:</strong> {analysisResults.num_submissions}
              </p>
              <p className={isDark ? "text-gray-300" : "text-gray-700"}>
                <strong>Archivos Analizados:</strong> {analysisResults.num_files_analyzed}
              </p>
              <p className={isDark ? "text-gray-300" : "text-gray-700"}>
                <strong>Umbral Similitud:</strong> {formatSimilarity(analysisResults.similarity_threshold)}%
              </p>
              <p className={isDark ? "text-gray-300" : "text-gray-700"}>
                <strong>Umbral Fragmento:</strong> {formatSimilarity(analysisResults.fragment_threshold)}%
              </p>
            </div>
          </div>

          {analysisResults.similarity_results && analysisResults.similarity_results.length > 0 ? (
            analysisResults.similarity_results.map((submissionPair, pairIndex) => (
              <div
                key={pairIndex}
                className={`rounded-2xl p-6 md:p-8 border mb-8 ${isDark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"} shadow-lg`}
              >
                <h3
                  className={`text-2xl font-semibold mb-2 flex items-center ${isDark ? "text-white" : "text-gray-900"}`}
                >
                  <GitCompare className="w-7 h-7 mr-3 text-purple-500" />
                  {submissionPair.submission1} <span className="mx-2 text-gray-500">vs</span>{" "}
                  {submissionPair.submission2}
                </h3>
                <div className="flex items-center mb-4">
                  <Percent
                    className={`w-5 h-5 mr-2 ${submissionPair.max_similarity > 0.7 ? "text-red-500" : submissionPair.max_similarity > 0.4 ? "text-yellow-500" : "text-green-500"}`}
                  />
                  <span className={`font-medium ${isDark ? "text-gray-300" : "text-gray-700"}`}>
                    Similitud Máxima: {formatSimilarity(submissionPair.max_similarity)}%
                  </span>
                  {submissionPair.is_plagiarism && (
                    <span className="ml-3 text-xs font-semibold px-2 py-1 bg-red-500 text-white rounded-full">
                      Plagio Detectado
                    </span>
                  )}
                </div>

                {submissionPair.similar_files && submissionPair.similar_files.length > 0 ? (
                  submissionPair.similar_files.map((filePair, fileIndex) => (
                    <div
                      key={fileIndex}
                      className={`rounded-xl p-4 mt-4 border ${isDark ? "bg-gray-700/50 border-gray-600" : "bg-gray-50 border-gray-200"}`}
                    >
                      <h4
                        className={`text-lg font-medium mb-3 flex items-center ${isDark ? "text-gray-200" : "text-gray-800"}`}
                      >
                        <FileText className="w-5 h-5 mr-2 text-blue-500" />
                        {filePair.file1} <span className="mx-1 text-gray-500">vs</span> {filePair.file2}
                      </h4>
                      <div className="flex items-center mb-3">
                        <Percent
                          className={`w-4 h-4 mr-1 ${filePair.combined_similarity > 0.7 ? "text-red-400" : filePair.combined_similarity > 0.4 ? "text-yellow-400" : "text-green-400"}`}
                        />
                        <span className={`text-sm ${isDark ? "text-gray-300" : "text-gray-600"}`}>
                          Similitud Combinada: {formatSimilarity(filePair.combined_similarity)}%
                        </span>
                        {filePair.is_plagiarism && (
                          <span className="ml-2 text-xs font-semibold px-1.5 py-0.5 bg-red-600 text-white rounded-full">
                            Plagio
                          </span>
                        )}
                      </div>

                      {filePair.fragments && filePair.fragments.length > 0 ? (
                        filePair.fragments.map((fragment, fragIndex) => (
                          <div
                            key={fragIndex}
                            className={`mt-3 pt-3 border-t ${isDark ? "border-gray-600" : "border-gray-300"}`}
                          >
                            <p className={`text-sm font-medium mb-2 ${isDark ? "text-gray-300" : "text-gray-700"}`}>
                              Fragmento {fragIndex + 1} (Similitud: {formatSimilarity(fragment.similarity)}%)
                            </p>
                            <div className="grid md:grid-cols-2 gap-4">
                              <div>
                                <p className={`text-xs mb-1 font-mono ${isDark ? "text-blue-400" : "text-blue-600"}`}>
                                  {filePair.file1} (Fragmento)
                                </p>
                                <pre
                                  className={`p-3 rounded-md text-xs overflow-x-auto ${isDark ? "bg-gray-900 text-gray-300" : "bg-gray-100 text-gray-800"} max-h-60`}
                                >
                                  <code>{fragment.code1 || "No se pudo reconstruir el fragmento."}</code>
                                </pre>
                              </div>
                              <div>
                                <p
                                  className={`text-xs mb-1 font-mono ${isDark ? "text-purple-400" : "text-purple-600"}`}
                                >
                                  {filePair.file2} (Fragmento)
                                </p>
                                <pre
                                  className={`p-3 rounded-md text-xs overflow-x-auto ${isDark ? "bg-gray-900 text-gray-300" : "bg-gray-100 text-gray-800"} max-h-60`}
                                >
                                  <code>{fragment.code2 || "No se pudo reconstruir el fragmento."}</code>
                                </pre>
                              </div>
                            </div>
                          </div>
                        ))
                      ) : (
                        <p className={`text-sm italic ${isDark ? "text-gray-400" : "text-gray-500"}`}>
                          No se encontraron fragmentos específicos para estos archivos.
                        </p>
                      )}
                    </div>
                  ))
                ) : (
                  <p className={`text-sm italic mt-4 ${isDark ? "text-gray-400" : "text-gray-500"}`}>
                    No se encontraron archivos con similitudes detalladas para este par de proyectos.
                  </p>
                )}
              </div>
            ))
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
