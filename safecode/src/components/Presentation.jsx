import { useState } from "react"
import {
  GraduationCap,
  Code2,
  Cpu,
  Globe,
  ChevronLeft,
  ChevronRight,
  Upload,
  Search,
  BarChart3,
  Heart,
} from "lucide-react"

function Presentation({ theme }) {
  const languages = [
    {
      icon: <GraduationCap className="w-12 h-12 text-blue-600 dark:text-blue-400" />,
      title: "Python",
      description:
        "Python es un lenguaje de programación de alto nivel, conocido por su simplicidad y fácil adopción para nuevos usuarios. Es ampliamente utilizado en el ámbito empresarial y académico, especialmente en áreas de ciencia de datos y machine learning.",
      points: [
        "Popularidad en la educación: sintaxis clara y enfoque en la resolución fácil de problemas.",
        "Bibliotecas: amplia gama de bibliotecas para la resolución de problemas.",
        "Facilidad de preprocesamiento: herramientas como Pandas permiten un preprocesamiento eficiente.",
      ],
    },
    {
      icon: <Code2 className="w-12 h-12 text-green-600 dark:text-green-400" />,
      title: "Java",
      description:
        "Java es un lenguaje orientado a objetos conocido por su portabilidad y robustez. Es utilizado en aplicaciones empresariales y por estudiantes para aprender paradigmas como la programación orientada a objetos.",
      points: [
        "Uso extendido en la educación: familiar para los estudiantes.",
        "Ecosistema: robusto con herramientas utilizadas por estudiantes.",
        "Proyectos: relevante para el análisis de similitudes de código.",
      ],
    },
    {
      icon: <Cpu className="w-12 h-12 text-purple-600 dark:text-purple-400" />,
      title: "C++",
      description:
        "C++ es un lenguaje de propósito general conocido por su rendimiento y control sobre los recursos del sistema. Es comúnmente utilizado en cursos de programación avanzada y sistemas.",
      points: [
        "Desarrollo de habilidades técnicas: enseñado en cursos avanzados.",
        "Proyectos de alto rendimiento: relevante para el análisis de similitudes.",
      ],
    },
    {
      icon: <Globe className="w-12 h-12 text-amber-600 dark:text-amber-400" />,
      title: "JavaScript",
      description:
        "JavaScript es un lenguaje de programación utilizado principalmente para el desarrollo web. Es esencial para la creación de aplicaciones interactivas y dinámicas en el navegador.",
      points: [
        "Dominio en el desarrollo web: común en proyectos académicos.",
        "Interactividad en proyectos: mejora la comprensión en estudiantes.",
        "Popularidad entre los estudiantes: a menudo el primer lenguaje aprendido.",
      ],
    },
  ]

  const features = [
    {
      icon: <Upload className="w-8 h-8 text-blue-600 dark:text-blue-400" />,
      title: "Carga Sencilla",
      description: "Simplemente cargue archivos ZIP con los proyectos que desea analizar.",
    },
    {
      icon: <Search className="w-8 h-8 text-green-600 dark:text-green-400" />,
      title: "Análisis Profundo",
      description: "Algoritmo avanzado que analiza la estructura y patrones del código.",
    },
    {
      icon: <BarChart3 className="w-8 h-8 text-purple-600 dark:text-purple-400" />,
      title: "Resultados Detallados",
      description: "Informes completos con porcentajes de similitud y fragmentos coincidentes.",
    },
  ]

  const [currentIndex, setCurrentIndex] = useState(0)

  const handlePrev = () => {
    setCurrentIndex((prevIndex) => (prevIndex === 0 ? languages.length - 1 : prevIndex - 1))
  }

  const handleNext = () => {
    setCurrentIndex((prevIndex) => (prevIndex === languages.length - 1 ? 0 : prevIndex + 1))
  }

  const isDark = theme === "dracula"

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
              <Heart className="w-8 h-8 text-red-500 mr-3" />
              <span
                className={`text-sm font-medium px-3 py-1 rounded-full ${
                  isDark ? "bg-red-900/50 text-red-300 border border-red-700" : "bg-red-100 text-red-800"
                }`}
              >
                Proyecto Sin Fines de Lucro
              </span>
            </div>
            <h1 className={`text-5xl md:text-6xl font-bold mb-6 ${isDark ? "text-white" : "text-gray-900"}`}>
              Sistema de Detección de{" "}
              <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Copias</span>
            </h1>
            <p
              className={`text-xl mb-8 max-w-3xl mx-auto leading-relaxed ${isDark ? "text-gray-300" : "text-gray-700"}`}
            >
              Herramienta gratuita y de código abierto que utiliza técnicas avanzadas para detectar similitudes en
              proyectos de programación, ayudando a instituciones educativas a mantener la integridad académica.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button 
              className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-lg transition-all duration-300 transform hover:scale-105 shadow-lg"
                onClick={() => window.location.href = "/analysis"}
              >
                Comenzar
              </button>
              <button
                className={`border-2 border-blue-600 font-semibold py-3 px-8 rounded-lg transition-all duration-300 ${
                  isDark
                    ? "text-blue-400 hover:bg-blue-600 hover:text-white"
                    : "text-blue-600 hover:bg-blue-600 hover:text-white"
                }`}
                onClick={() => window.location.href = "/info"}
              >
                Ver Información
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="max-w-7xl mx-auto px-6 py-16">
        <h2 className={`text-3xl font-bold text-center mb-12 ${isDark ? "text-white" : "text-gray-900"}`}>
          ¿Por qué elegir nuestro sistema?
        </h2>
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          {features.map((feature, index) => (
            <div
              key={index}
              className={`rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2 border ${
                isDark ? "bg-gray-800 border-gray-700 hover:bg-gray-750" : "bg-white border-gray-100"
              }`}
            >
              <div className={`p-3 rounded-full w-fit mb-4 ${isDark ? "bg-gray-700" : "bg-gray-50"}`}>
                {feature.icon}
              </div>
              <h3 className={`text-xl font-semibold mb-3 ${isDark ? "text-white" : "text-gray-900"}`}>
                {feature.title}
              </h3>
              <p className={isDark ? "text-gray-300" : "text-gray-600"}>{feature.description}</p>
            </div>
          ))}
        </div>

        {/* How to Use Section */}
        <div
          className={`rounded-2xl shadow-xl p-8 mb-16 border ${
            isDark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"
          }`}
        >
          <h2 className={`text-3xl font-bold text-center mb-8 ${isDark ? "text-white" : "text-gray-900"}`}>
            Cómo Usar
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="bg-blue-100 text-blue-600 w-12 h-12 rounded-full flex items-center justify-center font-bold text-xl mx-auto mb-4">
                1
              </div>
              <h3 className={`text-lg font-semibold mb-2 ${isDark ? "text-white" : "text-gray-900"}`}>
                Cargue los Proyectos
              </h3>
              <p className={isDark ? "text-gray-300" : "text-gray-600"}>
                Cargue un archivo ZIP por cada proyecto a analizar.
              </p>
            </div>
            <div className="text-center">
              <div className="bg-green-100 text-green-600 w-12 h-12 rounded-full flex items-center justify-center font-bold text-xl mx-auto mb-4">
                2
              </div>
              <h3 className={`text-lg font-semibold mb-2 ${isDark ? "text-white" : "text-gray-900"}`}>
                Seleccione el Lenguaje
              </h3>
              <p className={isDark ? "text-gray-300" : "text-gray-600"}>
                Seleccione el lenguaje de programación utilizado en los proyectos.
              </p>
            </div>
            <div className="text-center">
              <div className="bg-purple-100 text-purple-600 w-12 h-12 rounded-full flex items-center justify-center font-bold text-xl mx-auto mb-4">
                3
              </div>
              <h3 className={`text-lg font-semibold mb-2 ${isDark ? "text-white" : "text-gray-900"}`}>
                Revise los Resultados
              </h3>
              <p className={isDark ? "text-gray-300" : "text-gray-600"}>
                Revise los resultados del análisis para ver las similitudes detectadas.
              </p>
            </div>
          </div>
        </div>

        {/* Languages Section */}
        <div
          className={`rounded-2xl shadow-xl p-8 border ${
            isDark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"
          }`}
        >
          <h2 className={`text-3xl font-bold text-center mb-8 ${isDark ? "text-white" : "text-gray-900"}`}>
            Lenguajes Soportados
          </h2>

          {/* Language Carousel */}
          <div className="relative">
            <div className="flex items-center justify-between">
              <button
                onClick={handlePrev}
                className="bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-full shadow-lg transition-all duration-300 transform hover:scale-110 z-10"
              >
                <ChevronLeft className="w-6 h-6" />
              </button>

              <div className="flex-1 mx-8">
                <div
                  className={`rounded-xl p-8 shadow-inner ${
                    isDark ? "bg-gradient-to-r from-gray-700 to-gray-600" : "bg-gradient-to-r from-gray-50 to-blue-50"
                  }`}
                >
                  <div className="flex items-start space-x-6">
                    <div className={`p-4 rounded-xl shadow-md ${isDark ? "bg-gray-800" : "bg-white"}`}>
                      {languages[currentIndex].icon}
                    </div>
                    <div className="flex-1">
                      <h3 className={`text-2xl font-bold mb-3 ${isDark ? "text-white" : "text-gray-900"}`}>
                        {languages[currentIndex].title}
                      </h3>
                      <p className={`mb-4 leading-relaxed ${isDark ? "text-gray-300" : "text-gray-700"}`}>
                        {languages[currentIndex].description}
                      </p>
                      <ul className="space-y-2">
                        {languages[currentIndex].points.map((point, index) => (
                          <li key={index} className="flex items-start">
                            <div className="w-2 h-2 bg-blue-600 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                            <span className={`text-sm ${isDark ? "text-gray-300" : "text-gray-600"}`}>{point}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              <button
                onClick={handleNext}
                className="bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-full shadow-lg transition-all duration-300 transform hover:scale-110 z-10"
              >
                <ChevronRight className="w-6 h-6" />
              </button>
            </div>

            {/* Language indicators */}
            <div className="flex justify-center mt-6 space-x-2">
              {languages.map((_, index) => (
                <button
                  key={index}
                  onClick={() => setCurrentIndex(index)}
                  className={`w-3 h-3 rounded-full transition-all duration-300 ${
                    index === currentIndex
                      ? "bg-blue-600 scale-125"
                      : isDark
                        ? "bg-gray-600 hover:bg-gray-500"
                        : "bg-gray-300 hover:bg-gray-400"
                  }`}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Call to Action */}
        <div className="text-center mt-16">
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-8 text-white">
            <h2 className="text-3xl font-bold mb-4">¿Tienes alguna sugerencia o comentario?</h2>
            <p className="text-xl mb-6 opacity-90">
              Tu opinión es importante para mí. Ayúdame a mejorar el sistema con tus comentarios.
            </p>
            <button 
              className="bg-white text-blue-600 hover:bg-gray-100 font-semibold py-3 px-8 rounded-lg transition-all duration-300 transform hover:scale-105 shadow-lg"
              onClick={() => window.location.href = "/feedback"}
            >
              Enviar Comentarios
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Presentation
