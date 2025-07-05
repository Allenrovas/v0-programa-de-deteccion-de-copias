import { useState } from "react"
import { X, Upload, MessageSquare, ImageIcon, Send, CheckCircle, AlertCircle, TestTube } from "lucide-react"
import api from "../service/api"

function Feedback({ theme }) {
  const [comment, setComment] = useState("")
  const [images, setImages] = useState([])
  const [imagePreviews, setImagePreviews] = useState([])
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [modalMessage, setModalMessage] = useState("")
  const [modalType, setModalType] = useState("info") // 'info', 'success', 'error'
  const [isSubmitting, setIsSubmitting] = useState(false)

  const isDark = theme === "dracula"

  const handleCommentChange = (event) => {
    setComment(event.target.value)
  }

  const handleImageChange = (event) => {
    const selectedImages = Array.from(event.target.files)

    // Validar tamaño de archivos (máximo 10MB cada uno)
    const maxSize = 10 * 1024 * 1024 // 10MB
    const validImages = selectedImages.filter((image) => {
      if (image.size > maxSize) {
        setModalMessage(`La imagen "${image.name}" es demasiado grande. El tamaño máximo es 10MB.`)
        setModalType("error")
        setIsModalOpen(true)
        return false
      }
      return true
    })

    if (validImages.length > 0) {
      setImages((prevImages) => [...prevImages, ...validImages])

      const previews = validImages.map((image) => URL.createObjectURL(image))
      setImagePreviews((prevPreviews) => [...prevPreviews, ...previews])
    }
  }

  const handleRemoveImage = (index) => {
    // Liberar URL del objeto para evitar memory leaks
    URL.revokeObjectURL(imagePreviews[index])

    setImages((prevImages) => prevImages.filter((_, i) => i !== index))
    setImagePreviews((prevPreviews) => prevPreviews.filter((_, i) => i !== index))
  }

  const handleSubmit = async (event) => {
    event.preventDefault()

    if (!comment.trim()) {
      setModalMessage("Por favor, escribe un comentario antes de enviar.")
      setModalType("error")
      setIsModalOpen(true)
      return
    }

    setIsSubmitting(true)

    try {
      // Crear FormData para enviar archivos
      const formData = new FormData()
      formData.append("message", comment.trim())

      // Agregar imágenes al FormData
      images.forEach((image, index) => {
        formData.append("images", image)
      })

      const response = await api.post("/feedback/send", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })

      if (response.data.success) {
        setModalMessage(
          images.length === 0
            ? "¡Gracias por tu retroalimentación! Hemos recibido tu comentario correctamente"
            : `¡Gracias por tu retroalimentación! Hemos recibido tu comentario correctamente junto con ${images.length} ${images.length === 1 ? "imagen" : "imágenes"}`
        )
        setModalType("success")
        setIsModalOpen(true)

        // Limpiar formulario
        setComment("")

        // Liberar URLs de objetos antes de limpiar
        imagePreviews.forEach((url) => URL.revokeObjectURL(url))
        setImages([])
        setImagePreviews([])
      } else {
        throw new Error(response.data.message || "Error desconocido")
      }
    } catch (error) {
      console.error("Error enviando feedback:", error)
      let errorMessage = "Hubo un error al enviar tu mensaje. Por favor, intenta nuevamente."

      // Manejo de errores
      if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail
      } else if (error.response?.data?.message) {
        errorMessage = error.response.data.message
      } else if (error.message) {
        errorMessage = error.message
      } else if (error.code === "NETWORK_ERROR") {
        errorMessage = "No se pudo conectar con el servidor. Verifica tu conexión a internet."
      }

      setModalMessage(errorMessage)
      setModalType("error")
      setIsModalOpen(true)
    } finally {
      setIsSubmitting(false)
    }
  }

  const closeModal = () => {
    setIsModalOpen(false)
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
              <MessageSquare className="w-12 h-12 text-blue-600 mr-4" />
              <span
                className={`text-sm font-medium px-3 py-1 rounded-full ${
                  isDark ? "bg-green-900/50 text-green-300 border border-green-700" : "bg-green-100 text-green-800"
                }`}
              >
                Tu Opinión Importa
              </span>
            </div>
            <h1 className={`text-4xl md:text-5xl font-bold mb-6 ${isDark ? "text-white" : "text-gray-900"}`}>
              Enviar{" "}
              <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Retroalimentación
              </span>
            </h1>
            <p
              className={`text-xl mb-8 max-w-3xl mx-auto leading-relaxed ${isDark ? "text-gray-300" : "text-gray-700"}`}
            >
              Ayúdame a mejorar el sistema de detección de copias. Tu retroalimentación es fundamental para el
              desarrollo continuo del proyecto.
            </p>
          </div>
        </div>
      </div>

      {/* Feedback Form Section */}
      <div className="max-w-4xl mx-auto px-6 py-8">
        <div className="grid lg:grid-cols-2 gap-12 items-start">
          {/* Left Column - Information */}
          <div className="space-y-8">
            <div
              className={`rounded-2xl p-8 border ${isDark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"} shadow-lg`}
            >
              <h2 className={`text-2xl font-bold mb-6 ${isDark ? "text-white" : "text-gray-900"}`}>
                ¿Por qué es importante tu feedback?
              </h2>
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <CheckCircle className="w-6 h-6 text-green-500 mt-1 flex-shrink-0" />
                  <div>
                    <h3 className={`font-semibold mb-1 ${isDark ? "text-white" : "text-gray-900"}`}>Mejora Continua</h3>
                    <p className={isDark ? "text-gray-300" : "text-gray-600"}>
                      Tus comentarios me ayudan a identificar áreas de mejora y nuevas funcionalidades.
                    </p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <CheckCircle className="w-6 h-6 text-green-500 mt-1 flex-shrink-0" />
                  <div>
                    <h3 className={`font-semibold mb-1 ${isDark ? "text-white" : "text-gray-900"}`}>
                      Experiencia de Usuario
                    </h3>
                    <p className={isDark ? "text-gray-300" : "text-gray-600"}>
                      Me permite optimizar la interfaz y hacer el sistema más intuitivo.
                    </p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <CheckCircle className="w-6 h-6 text-green-500 mt-1 flex-shrink-0" />
                  <div>
                    <h3 className={`font-semibold mb-1 ${isDark ? "text-white" : "text-gray-900"}`}>
                      Desarrollo Académico
                    </h3>
                    <p className={isDark ? "text-gray-300" : "text-gray-600"}>
                      Como proyecto de graduación, tu feedback contribuye al aprendizaje y desarrollo.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div
              className={`rounded-2xl p-6 border ${isDark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"} shadow-lg`}
            >
              <h3 className={`text-lg font-semibold mb-4 ${isDark ? "text-white" : "text-gray-900"}`}>
                Tipos de Feedback Útil
              </h3>
              <ul className={`space-y-2 text-sm ${isDark ? "text-gray-300" : "text-gray-600"}`}>
                <li>• Errores o bugs encontrados</li>
                <li>• Sugerencias de mejora</li>
                <li>• Problemas de usabilidad</li>
                <li>• Ideas para nuevas funcionalidades</li>
                <li>• Comentarios sobre la precisión del análisis</li>
              </ul>
            </div>
          </div>

          {/* Right Column - Form */}
          <div
            className={`rounded-2xl shadow-xl p-8 border ${isDark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}
          >
            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label className={`block text-sm font-medium mb-3 ${isDark ? "text-white" : "text-gray-900"}`}>
                  <MessageSquare className="w-4 h-4 inline mr-2" />
                  Comentario
                </label>
                <textarea
                  value={comment}
                  onChange={handleCommentChange}
                  className={`w-full px-4 py-3 rounded-lg border transition-colors duration-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none ${
                    isDark
                      ? "bg-gray-700 border-gray-600 text-white placeholder-gray-400"
                      : "bg-white border-gray-300 text-gray-900 placeholder-gray-500"
                  }`}
                  placeholder="Escribe tu comentario aquí... Comparte tus experiencias, sugerencias o reporta cualquier problema que hayas encontrado."
                  rows="6"
                />
                <div className={`text-xs mt-2 ${isDark ? "text-gray-400" : "text-gray-500"}`}>
                  {comment.length}/1000 caracteres
                </div>
              </div>

              <div>
                <label className={`block text-sm font-medium mb-3 ${isDark ? "text-white" : "text-gray-900"}`}>
                  <ImageIcon className="w-4 h-4 inline mr-2" />
                  Subir Imágenes (Opcional)
                </label>
                <div
                  className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors duration-300 ${
                    isDark
                      ? "border-gray-600 hover:border-gray-500 bg-gray-700/50"
                      : "border-gray-300 hover:border-gray-400 bg-gray-50"
                  }`}
                >
                  <Upload className={`w-8 h-8 mx-auto mb-2 ${isDark ? "text-gray-400" : "text-gray-500"}`} />
                  <input
                    type="file"
                    accept="image/*"
                    multiple
                    onChange={handleImageChange}
                    className="hidden"
                    id="image-upload"
                  />
                  <label
                    htmlFor="image-upload"
                    className={`cursor-pointer text-sm ${isDark ? "text-gray-300" : "text-gray-600"}`}
                  >
                    Haz clic para subir imágenes o arrastra y suelta
                  </label>
                  <p className={`text-xs mt-1 ${isDark ? "text-gray-400" : "text-gray-500"}`}>
                    PNG, JPG, GIF hasta 10MB cada una
                  </p>
                </div>
              </div>

              {imagePreviews.length > 0 && (
                <div className="grid grid-cols-2 gap-4">
                  {imagePreviews.map((src, index) => (
                    <div key={index} className="relative group">
                      <img
                        src={src || "/placeholder.svg"}
                        alt={`Preview ${index}`}
                        className="w-full h-32 object-cover rounded-lg border border-gray-200"
                      />
                      <button
                        type="button"
                        onClick={() => handleRemoveImage(index)}
                        className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600 transition-colors duration-300 opacity-0 group-hover:opacity-100"
                      >
                        <X className="w-4 h-4" />
                      </button>
                      <div
                        className={`absolute bottom-0 left-0 right-0 bg-black/50 text-white text-xs p-1 rounded-b-lg ${isDark ? "text-gray-200" : "text-white"}`}
                      >
                        {images[index]?.name}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <button
                type="submit"
                disabled={isSubmitting}
                className={`w-full py-3 px-6 rounded-lg font-semibold transition-all duration-300 transform hover:scale-105 shadow-lg ${
                  isSubmitting ? "bg-gray-400 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700 text-white"
                }`}
              >
                {isSubmitting ? (
                  <div className="flex items-center justify-center">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                    Enviando...
                  </div>
                ) : (
                  <div className="flex items-center justify-center">
                    <Send className="w-5 h-5 mr-2" />
                    Enviar Feedback
                    {images.length > 0 && (
                      <span className="ml-2 bg-blue-500 text-white text-xs px-2 py-1 rounded-full">
                        +{images.length} img
                      </span>
                    )}
                  </div>
                )}
              </button>
            </form>
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
              ) : (
                <AlertCircle className="w-8 h-8 text-red-500 mr-3" />
              )}
              <h3 className={`text-lg font-semibold ${isDark ? "text-white" : "text-gray-900"}`}>
                {modalType === "success" ? "¡Éxito!" : "Aviso"}
              </h3>
            </div>
            <p className={`mb-6 ${isDark ? "text-gray-300" : "text-gray-600"}`}>{modalMessage}</p>
            <button
              onClick={closeModal}
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

export default Feedback
