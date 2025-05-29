import { Heart, Mail, GraduationCap, Code2 } from "lucide-react"

function Footer({ theme }) {
  const isDark = theme === "dracula"

  return (
    <footer
      className={`transition-colors duration-300 border-t ${
        isDark ? "bg-gray-900 border-gray-700 text-gray-300" : "bg-gray-50 border-gray-200 text-gray-700"
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 py-12">
        {/* Main Footer Content */}
        <div className="grid md:grid-cols-2 gap-8 mb-8">
          {/* Left Column - Project Info */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2 mb-4">
              <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-2 rounded-lg">
                <Heart className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Safe Code
              </span>
              <span
                className={`text-xs font-medium px-2 py-1 rounded-full ${
                  isDark ? "bg-red-900/50 text-red-300 border border-red-700" : "bg-red-100 text-red-800"
                }`}
              >
                Sin Fines de Lucro
              </span>
            </div>

            <div className="flex items-start space-x-3">
              <GraduationCap className={`w-5 h-5 mt-1 flex-shrink-0 ${isDark ? "text-blue-400" : "text-blue-600"}`} />
              <p className={`leading-relaxed ${isDark ? "text-gray-300" : "text-gray-600"}`}>
                Prototipo desarrollado como trabajo de graduación para la Facultad de Ingeniería de la Universidad de
                San Carlos de Guatemala.
              </p>
            </div>

            <div className="flex items-start space-x-3">
              <Code2 className={`w-5 h-5 mt-1 flex-shrink-0 ${isDark ? "text-green-400" : "text-green-600"}`} />
              <p className={`leading-relaxed ${isDark ? "text-gray-300" : "text-gray-600"}`}>
                Herramienta de código abierto para detectar similitudes en proyectos de programación y mantener la
                integridad académica.
              </p>
            </div>
          </div>

          {/* Right Column - Contact & Developer */}
          <div className="space-y-4">
            <h3 className={`text-lg font-semibold mb-4 ${isDark ? "text-white" : "text-gray-900"}`}>
              Información del Desarrollador
            </h3>

            <div className="space-y-3">
              <div
                className={`p-4 rounded-lg border ${isDark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"}`}
              >
                <p className={`font-medium mb-2 ${isDark ? "text-white" : "text-gray-900"}`}>
                  Allen Giankarlo Román Vásquez
                </p>
                <p className={`text-sm ${isDark ? "text-gray-400" : "text-gray-600"}`}>
                  Estudiante de Ingeniería en Ciencias y Sistemas
                </p>
              </div>

              <div
                className={`p-4 rounded-lg border ${isDark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"}`}
              >
                <div className="flex items-center space-x-2 mb-2">
                  <Mail className={`w-4 h-4 ${isDark ? "text-blue-400" : "text-blue-600"}`} />
                  <span className={`text-sm font-medium ${isDark ? "text-white" : "text-gray-900"}`}>Contacto</span>
                </div>
                <a
                  href="mailto:2909080001301@ingenieria.usac.edu.gt"
                  className={`text-sm transition-colors duration-300 ${
                    isDark ? "text-blue-400 hover:text-blue-300" : "text-blue-600 hover:text-blue-700"
                  } hover:underline`}
                >
                  2909080001301@ingenieria.usac.edu.gt
                </a>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Section */}
        <div className={`pt-8 border-t ${isDark ? "border-gray-700" : "border-gray-200"}`}>
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <div className="flex items-center space-x-4">
              <p className={`text-sm ${isDark ? "text-gray-400" : "text-gray-600"}`}>
                © 2024 Allen Giankarlo Román Vásquez. Todos los derechos reservados.
              </p>
            </div>

            <div className="flex items-center space-x-4">
              <span className={`text-sm ${isDark ? "text-gray-400" : "text-gray-600"}`}>Hecho con</span>
              <Heart className="w-4 h-4 text-red-500" />
              <span className={`text-sm ${isDark ? "text-gray-400" : "text-gray-600"}`}>para la educación</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default Footer
