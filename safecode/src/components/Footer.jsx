import React from 'react';

function Footer() {
  return (
    <footer className="footer bg-base-300 text-base-content py-6">
      <div className="max-w-3xl mx-auto text-center flex flex-col items-center">
        <p className="text-lg leading-relaxed mb-2">
          © 2024 Allen Giankarlo Román Vásquez. Todos los derechos reservados.
        </p>
        <p className="text-lg leading-relaxed mb-2">
          Este es un prototipo desarrollado como trabajo de graduación para la Facultad de Ingeniería de la Universidad de San Carlos de Guatemala.
        </p>
        <p className="text-lg leading-relaxed mb-2">
          Para consultas, contacta a: <a href="mailto:2909080001301@ingenieria.usac.edu.gt" className="text-primary hover:underline">2909080001301@ingenieria.usac.edu.gt</a>
        </p>
        <p className="text-lg leading-relaxed">
          Desarrollado con ❤️ por Allen Giankarlo Román Vásquez.
        </p>
      </div>
    </footer>
  );
}

export default Footer;