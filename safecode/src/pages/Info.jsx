"use client"

import { GraduationCap, Target, Globe, Lightbulb, Calendar, Mail, BookOpen, Code2, Brain, Award } from "lucide-react"

const InfoCard = ({ icon: Icon, title, description, isDark, variant = "default" }) => {
  const variants = {
    default: isDark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100",
    featured: "bg-gradient-to-br from-blue-600 to-purple-600 text-white border-transparent",
    team: isDark
      ? "bg-gradient-to-br from-gray-700 to-gray-600 border-gray-600"
      : "bg-gradient-to-br from-blue-50 to-indigo-50 border-blue-200",
  }

  return (
    <div
      className={`shadow-lg rounded-xl p-6 flex flex-col items-center text-center hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2 border ${variants[variant]}`}
    >
      <div
        className={`w-16 h-16 mb-4 p-3 rounded-full ${
          variant === "featured"
            ? "bg-white/20"
            : variant === "team"
              ? isDark
                ? "bg-gray-800"
                : "bg-white"
              : isDark
                ? "bg-gray-700"
                : "bg-gray-50"
        }`}
      >
        <Icon className={`w-full h-full ${variant === "featured" ? "text-white" : "text-blue-600"}`} />
      </div>
      <h3
        className={`text-xl font-semibold mb-3 ${
          variant === "featured" ? "text-white" : isDark ? "text-white" : "text-gray-900"
        }`}
      >
        {title}
      </h3>
      <p
        className={`leading-relaxed ${
          variant === "featured" ? "text-white/90" : isDark ? "text-gray-300" : "text-gray-600"
        }`}
      >
        {description}
      </p>
    </div>
  )
}

function Info({ theme }) {
  const isDark = theme === "dracula"

  const projectInfo = [
    {
      icon: BookOpen,
      title: "Introducción al Proyecto",
      description:
        "Este proyecto aborda el problema del plagio en proyectos de programación, utilizando técnicas avanzadas de machine learning para detectar similitudes en el código fuente.",
      variant: "featured",
    },
    {
      icon: Target,
      title: "Objetivos del Proyecto",
      description:
        "Desarrollar un sistema de detección de copias en proyectos de programación con una precisión mínima del 70%.",
    },
    {
      icon: Brain,
      title: "Metodología",
      description:
        "El sistema analizará estructuras, patrones y lógicas de programación utilizando técnicas de machine learning.",
    },
  ]

  const technicalInfo = [
    {
      icon: Globe,
      title: "Alcances y Límites",
      description:
        "El sistema se centrará en lenguajes como Python, Java, C++ y JavaScript, para proyectos de tamaño pequeño a mediano.",
    },
    {
      icon: Lightbulb,
      title: "Antecedentes y Contexto",
      description:
        "Basado en investigaciones previas, este proyecto busca integrar enfoques innovadores para mejorar la detección de plagio.",
    },
    {
      icon: Calendar,
      title: "Planificación del Proyecto",
      description: "El proyecto sigue un cronograma detallado para asegurar su finalización exitosa.",
    },
  ]

  const teamInfo = [
    {
      icon: GraduationCap,
      title: "Estudiante",
      description: "Allen Giankarlo Román Vásquez",
      variant: "team",
    },
    {
      icon: Award,
      title: "Asesora",
      description: "Inga. Mirna Ivonne Aldana Larrazabal",
      variant: "team",
    },
  ]

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
                  isDark ? "bg-blue-900/50 text-blue-300 border border-blue-700" : "bg-blue-100 text-blue-800"
                }`}
              >
                Trabajo de Graduación
              </span>
            </div>
            <h1 className={`text-4xl md:text-5xl font-bold mb-6 ${isDark ? "text-white" : "text-gray-900"}`}>
              Información del{" "}
              <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Proyecto
              </span>
            </h1>
            <p
              className={`text-xl mb-8 max-w-3xl mx-auto leading-relaxed ${isDark ? "text-gray-300" : "text-gray-700"}`}
            >
              Sistema de Detección de Copias en Proyectos de Programación utilizando Técnicas de Machine Learning
            </p>
            <div className="flex items-center justify-center space-x-4">
              <div className={`px-4 py-2 rounded-lg ${isDark ? "bg-gray-800" : "bg-white"} shadow-md`}>
                <span className={`text-sm ${isDark ? "text-gray-300" : "text-gray-600"}`}>
                  Universidad de San Carlos de Guatemala
                </span>
              </div>
              <div className={`px-4 py-2 rounded-lg ${isDark ? "bg-gray-800" : "bg-white"} shadow-md`}>
                <span className={`text-sm ${isDark ? "text-gray-300" : "text-gray-600"}`}>Facultad de Ingeniería</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Project Overview Section */}
      <div className="max-w-7xl mx-auto px-6 py-16">
        <h2 className={`text-3xl font-bold text-center mb-12 ${isDark ? "text-white" : "text-gray-900"}`}>
          Resumen del Proyecto
        </h2>
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          {projectInfo.map((item, index) => (
            <InfoCard
              key={index}
              icon={item.icon}
              title={item.title}
              description={item.description}
              isDark={isDark}
              variant={item.variant}
            />
          ))}
        </div>

        {/* Technical Details Section */}
        <h2 className={`text-3xl font-bold text-center mb-12 ${isDark ? "text-white" : "text-gray-900"}`}>
          Detalles Técnicos
        </h2>
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          {technicalInfo.map((item, index) => (
            <InfoCard key={index} icon={item.icon} title={item.title} description={item.description} isDark={isDark} />
          ))}
        </div>

        {/* Team Section */}
        <div
          className={`rounded-2xl shadow-xl p-8 mb-16 border ${
            isDark ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"
          }`}
        >
          <h2 className={`text-3xl font-bold text-center mb-8 ${isDark ? "text-white" : "text-gray-900"}`}>
            Equipo del Proyecto
          </h2>
          <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            {teamInfo.map((item, index) => (
              <InfoCard
                key={index}
                icon={item.icon}
                title={item.title}
                description={item.description}
                isDark={isDark}
                variant={item.variant}
              />
            ))}
          </div>
        </div>

        {/* Contact Section */}
        <div className="text-center">
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-8 text-white">
            <Mail className="w-12 h-12 mx-auto mb-4" />
            <h2 className="text-3xl font-bold mb-4">¿Tienes preguntas?</h2>
            <p className="text-xl mb-6 opacity-90">Para más información sobre el proyecto, no dudes en contactarnos.</p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <a
                href="mailto:2909080001301@ingenieria.usac.edu.gt"
                className="bg-white text-blue-600 hover:bg-gray-100 font-semibold py-3 px-8 rounded-lg transition-all duration-300 transform hover:scale-105 shadow-lg"
              >
                Enviar Email
              </a>
              <span className="text-white/80">2909080001301@ingenieria.usac.edu.gt</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Info
