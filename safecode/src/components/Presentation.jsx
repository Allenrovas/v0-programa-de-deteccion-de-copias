import React, { useState } from 'react';
import { AcademicCapIcon, CodeIcon, ChipIcon, GlobeAltIcon } from '@heroicons/react/outline';

function Presentation() {
  const languages = [
    {
      icon: <AcademicCapIcon className="w-12 h-12 text-primary" />,
      title: 'Python',
      description: 'Python es un lenguaje de programación de alto nivel, conocido por su simplicidad y fácil adopción para nuevos usuarios. Es ampliamente utilizado en el ámbito empresarial y académico, especialmente en áreas de ciencia de datos y machine learning.',
      points: [
        'Popularidad en la educación: sintaxis clara y enfoque en la resolución fácil de problemas.',
        'Bibliotecas: amplia gama de bibliotecas para la resolución de problemas.',
        'Facilidad de preprocesamiento: herramientas como Pandas permiten un preprocesamiento eficiente.',
      ],
    },
    {
      icon: <CodeIcon className="w-12 h-12 text-secondary" />,
      title: 'Java',
      description: 'Java es un lenguaje orientado a objetos conocido por su portabilidad y robustez. Es utilizado en aplicaciones empresariales y por estudiantes para aprender paradigmas como la programación orientada a objetos.',
      points: [
        'Uso extendido en la educación: familiar para los estudiantes.',
        'Ecosistema: robusto con herramientas utilizadas por estudiantes.',
        'Proyectos: relevante para el análisis de similitudes de código.',
      ],
    },
    {
      icon: <ChipIcon className="w-12 h-12 text-accent" />,
      title: 'C++',
      description: 'C++ es un lenguaje de propósito general conocido por su rendimiento y control sobre los recursos del sistema. Es comúnmente utilizado en cursos de programación avanzada y sistemas.',
      points: [
        'Desarrollo de habilidades técnicas: enseñado en cursos avanzados.',
        'Proyectos de alto rendimiento: relevante para el análisis de similitudes.',
      ],
    },
    {
      icon: <GlobeAltIcon className="w-12 h-12 text-info" />,
      title: 'JavaScript',
      description: 'JavaScript es un lenguaje de programación utilizado principalmente para el desarrollo web. Es esencial para la creación de aplicaciones interactivas y dinámicas en el navegador.',
      points: [
        'Dominio en el desarrollo web: común en proyectos académicos.',
        'Interactividad en proyectos: mejora la comprensión en estudiantes.',
        'Popularidad entre los estudiantes: a menudo el primer lenguaje aprendido.',
      ],
    },
  ];

  const [currentIndex, setCurrentIndex] = useState(0);

  const handlePrev = () => {
    setCurrentIndex((prevIndex) => (prevIndex === 0 ? languages.length - 1 : prevIndex - 1));
  };

  const handleNext = () => {
    setCurrentIndex((prevIndex) => (prevIndex === languages.length - 1 ? 0 : prevIndex + 1));
  };

  return (
    <div className="presentation p-6 bg-base-200 text-base-content min-h-screen flex items-center justify-center">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-4xl font-bold mb-4 text-center text-primary">Bienvenido al Sistema de Detección de Copias</h1>
        <p className="mb-6 text-lg text-center">
          Este sistema utiliza técnicas avanzadas para detectar similitudes en proyectos de programación, ayudando a identificar posibles casos de plagio.
        </p>
        <div className="card shadow-lg bg-base-100 mb-6 transition-transform transform hover:scale-105">
          <div className="card-body">
            <h2 className="card-title text-center text-secondary">Cómo Usar</h2>
            <ol className="list-decimal list-inside text-center">
              <li>Cargue un archivo ZIP por cada proyecto a analizar.</li>
              <li>Seleccione el lenguaje de programación utilizado en los proyectos.</li>
              <li>Revise los resultados del análisis para ver las similitudes detectadas.</li>
            </ol>
          </div>
        </div>
        <div className="card shadow-lg bg-base-100 mb-6 transition-transform transform hover:scale-105">
          <div className="card-body">
            <h2 className="card-title text-center text-secondary">Lenguajes Soportados</h2>
            <div className="flex items-center justify-between">
              <button onClick={handlePrev} className="btn btn-secondary btn-circle">
                &lt;
              </button>
              <div className="flex items-center space-x-4 p-4  rounded-lg shadow-md transition-all duration-300 ease-in-out transform hover:scale-105">
                {languages[currentIndex].icon}
                <div>
                  <h3 className="text-xl font-semibold">{languages[currentIndex].title}</h3>
                  <p className="text-sm">{languages[currentIndex].description}</p>
                  <ul className="list-disc list-inside text-sm">
                    {languages[currentIndex].points.map((point, index) => (
                      <li key={index}>{point}</li>
                    ))}
                  </ul>
                </div>
              </div>
              <button onClick={handleNext} className="btn btn-secondary btn-circle">
                &gt;
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Presentation;