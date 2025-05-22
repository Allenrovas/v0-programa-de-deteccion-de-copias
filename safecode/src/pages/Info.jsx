import React from 'react';
import { AcademicCapIcon, ClipboardListIcon, ChartBarIcon, GlobeAltIcon, LightBulbIcon, UserIcon, CalendarIcon, MailIcon } from '@heroicons/react/outline';
const InfoCard = ({ icon: Icon, title, description }) => {
  return (
    <div class="bg-base-100 shadow-lg rounded-lg p-6 flex flex-col items-center text-center hover:shadow-xl transition-shadow duration-300">
      <div class="text-primary w-16 h-16 mb-4">
        <Icon />
 </div>
 <h3 class="text-xl font-semibold mb-2">{title}</h3>
     <p class="text-base-content text-lg">{description}</p>
   </div>
 );
};

function Info() {
  const infoData = [
    {
      icon: AcademicCapIcon,
      title: "Introducción al Proyecto",
      description: "Este proyecto aborda el problema del plagio en proyectos de programación, utilizando técnicas avanzadas de machine learning para detectar similitudes en el código fuente.",
    },
    {
      icon: ClipboardListIcon,
      title: "Objetivos del Proyecto",
      description: "Desarrollar un sistema de detección de copias en proyectos de programación con una precisión mínima del 85%.",
    },
    {
      icon: ChartBarIcon,
      title: "Metodología",
      description: "El sistema analizará estructuras, patrones y lógicas de programación utilizando técnicas de machine learning.",
    },
    {
      icon: GlobeAltIcon,
      title: "Alcances y Límites",
      description: "El sistema se centrará en lenguajes como Python, Java, C++ y JavaScript, para proyectos de tamaño pequeño a mediano.",
    },
    {
      icon: LightBulbIcon,
      title: "Antecedentes y Contexto",
      description: "Basado en investigaciones previas, este proyecto busca integrar enfoques innovadores para mejorar la detección de plagio.",
    },
    {
      icon: UserIcon,
      title: "Estudiante",
      description: "Allen Giankarlo Román Vásquez",
    },
    {
      icon: UserIcon,
      title: "Asesora",
      description: "Inga. Mirna Ivonne Aldana Larrazabal",
    },
    {
      icon: CalendarIcon,
      title: "Planificación del Proyecto",
      description: "El proyecto sigue un cronograma detallado para asegurar su finalización exitosa.",
    },
    {
      icon: MailIcon,
      title: "Contacto",
      description: "Para más información, por favor contacta a Allen Giankarlo Román Vásquez.",
    },
  ];

  return (
    <div class="info-page p-8 bg-base-200 text-base-content min-h-screen">
      <div class="max-w-6xl mx-auto">
        <h1 class="text-4xl font-bold mb-12 text-center text-primary">Información del Proyecto</h1>
       <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
          {infoData.map((item, index) => (
            <InfoCard              key={index}
              icon={item.icon}
              title={item.title}
              description={item.description}
            />
          ))}
        </div>
 </div>
 </div>
 );
}

export default Info;