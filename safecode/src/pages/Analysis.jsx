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

    try {
      // Crea un FormData con todos los archivos y el lenguaje seleccionado
      const formData = new FormData()
      formData.append("language", language) // Lenguaje al FormData
      files.forEach((file) => {
        formData.append("files", file) // Se agrega cada archivo al FormData
        console.log("Archivo:", file.name)
      })

      // Enviar el FormData al backend
      const response = await api.post("/upload/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })

      console.log("Respuesta del servidor:", response.data)
      setModalMessage("Archivos subidos exitosamente. El análisis ha comenzado.")
      setModalType("success")
      setIsModalOpen(true)
      setIsSubmitting(false)
    } catch (error) {
      console.error("Error al subir los archivos:", error)
      setModalMessage("Ocurrió un error al subir los archivos. Por favor, inténtalo de nuevo.")
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
                <div className="flex items-start">
                  <div
                    className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${
                      isDark ? "bg-blue-600" : "bg-blue-500"
                    }`}
                  >
                    1
                  </div>
                  <div className="ml-4">
                    <h3 className={`font-semibold mb-1 ${isDark ? "text-white" : "text-gray-900"}`}>
                      Selecciona el Lenguaje
                    </h3>
                    <p className={isDark ? "text-gray-300" : "text-gray-600"}>
                      Elige el lenguaje de programación de tus proyectos.
                    </p>
                  </div>
                </div>

                <div className="flex items-start">
                  <div
                    className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${
                      isDark ? "bg-blue-600" : "bg-blue-500"
                    }`}
                  >
                    2
                  </div>
                  <div className="ml-4">
                    <h3 className={`font-semibold mb-1 ${isDark ? "text-white" : "text-gray-900"}`}>
                      Carga los Archivos
                    </h3>
                    <p className={isDark ? "text-gray-300" : "text-gray-600"}>
                      Sube al menos dos archivos ZIP que contengan los proyectos a comparar.
                    </p>
                  </div>
                </div>

                <div className="flex items-start">
                  <div
                    className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${
                      isDark ? "bg-blue-600" : "bg-blue-500"
                    }`}
                  >
                    3
                  </div>
                  <div className="ml-4">
                    <h3 className={`font-semibold mb-1 ${isDark ? "text-white" : "text-gray-900"}`}>
                      Revisa los Resultados
                    </h3>
                    <p className={isDark ? "text-gray-300" : "text-gray-600"}>
                      El sistema procesará los archivos y mostrará un informe detallado de similitudes.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div
              className={`rounded-2xl p-6 border ${isDark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"} shadow-lg`}
            >
              <h3 className={`text-lg font-semibold mb-4 ${isDark ? "text-white" : "text-gray-900"}`}>
                Lenguajes Soportados
              </h3>
              <div className="grid grid-cols-2 gap-3">
                <div
                  className={`p-3 rounded-lg flex items-center ${
                    isDark ? "bg-gray-700 text-blue-400" : "bg-blue-50 text-blue-700"
                  }`}
                >
                  <FileCode className="w-5 h-5 mr-2" />
                  <span>Python</span>
                </div>
                <div
                  className={`p-3 rounded-lg flex items-center ${
                    isDark ? "bg-gray-700 text-green-400" : "bg-green-50 text-green-700"
                  }`}
                >
                  <FileCode className="w-5 h-5 mr-2" />
                  <span>Java</span>
                </div>
                <div
                  className={`p-3 rounded-lg flex items-center ${
                    isDark ? "bg-gray-700 text-purple-400" : "bg-purple-50 text-purple-700"
                  }`}
                >
                  <FileCode className="w-5 h-5 mr-2" />
                  <span>C++</span>
                </div>
                <div
                  className={`p-3 rounded-lg flex items-center ${
                    isDark ? "bg-gray-700 text-amber-400" : "bg-amber-50 text-amber-700"
                  }`}
                >
                  <FileCode className="w-5 h-5 mr-2" />
                  <span>JavaScript</span>
                </div>
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
                      ? "bg-gray-400 cursor-not-allowed"
                      : "bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white"
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
