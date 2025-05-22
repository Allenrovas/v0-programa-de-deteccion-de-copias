import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { XIcon } from '@heroicons/react/outline';
import { Dialog } from '@headlessui/react';
import api from '../service/api';

function Analysis() {
  const [language, setLanguage] = useState('');
  const [files, setFiles] = useState([]);
  const [fileNames, setFileNames] = useState([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalMessage, setModalMessage] = useState('');

  const handleLanguageChange = (event) => {
    setLanguage(event.target.value);
  };

  const onDrop = (acceptedFiles) => {
    const zipFiles = acceptedFiles.filter((file) => file.name.endsWith('.zip'));
    setFiles((prevFiles) => [...prevFiles, ...zipFiles]);
    setFileNames((prevNames) => [...prevNames, ...zipFiles.map((file) => file.name)]);
  };

  const removeFile = (index) => {
    setFiles((prevFiles) => prevFiles.filter((_, i) => i !== index));
    setFileNames((prevNames) => prevNames.filter((_, i) => i !== index));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!language || files.length < 2) {
   setModalMessage('Por favor, selecciona un lenguaje y carga al menos dos archivos ZIP.');
      setIsModalOpen(true);
      return;
    }
  
    try {
      // Crea un FormData con todos los archivos y el lenguaje seleccionado
      const formData = new FormData();
      formData.append('language', language); // Lenguaje al FormData
      files.forEach((file) => {
        formData.append('files', file); // Se agrega cada archivo al FormData
        console.log('Archivo:', file.name);
      });

      
  
      // Enviar el FormData al backend
      const response = await api.post('/upload/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
  
      console.log('Respuesta del servidor:', response.data);
      setModalMessage('Archivos subidos exitosamente. El análisis ha comenzado.');
      setIsModalOpen(true);
    } catch (error) {
      console.error('Error al subir los archivos:', error);
      setModalMessage('Ocurrió un error al subir los archivos. Por favor, inténtalo de nuevo.');
      setIsModalOpen(true);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: '.zip',
    multiple: true,
  });

  return (
    <div className="p-6 bg-base-200 text-base-content min-h-screen flex items-center justify-center">
      <div className="max-w-lg w-full bg-base-100 shadow-lg rounded-lg p-6">
        <h1 className="text-4xl font-bold mb-12 text-center text-primary">Análisis de Proyectos</h1>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium mb-2">Selecciona el Lenguaje de Programación</label>
            <select value={language} onChange={handleLanguageChange} className="select select-bordered w-full" required>
              <option value="" disabled>Selecciona un lenguaje</option>
              <option value="python">Python</option>
              <option value="java">Java</option>
              <option value="cpp">C++</option>
              <option value="javascript">JavaScript</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Carga los Archivos ZIP</label>
            <div {...getRootProps()} className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer ${isDragActive ? 'border-primary bg-primary/10' : 'border-base-300'}`}>
              <input {...getInputProps()} />
              {isDragActive ? (
                <p className="text-primary">Suelta los archivos aquí...</p>
              ) : (
                <>
                  <p className="text-base-content">Arrastra y suelta tus archivos aquí, o haz clic para seleccionarlos.</p>
                  <p className="text-sm text-gray-500 mt-2">Solo se aceptan archivos .zip</p>
                </>
              )}
            </div>
          </div>

          {fileNames.length > 0 && (
            <div className="mt-4">
              <h2 className="text-lg font-medium mb-2">Archivos seleccionados:</h2>
              <ul className="list-disc list-inside">
                {fileNames.map((name, index) => (
                  <li key={index} className="flex justify-between items-center">
                    {name}
                    <button onClick={() => removeFile(index)} className="ml-2 text-red-500 hover:text-red-700">
                      <XIcon className="w-5 h-5" />
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <button type="submit" className="btn btn-primary w-full">Iniciar Análisis</button>
        </form>
      </div>

      <Dialog open={isModalOpen} onClose={() => setIsModalOpen(false)} className="fixed inset-0 flex items-center justify-center p-4 bg-black/50">
        <div className="bg-white p-6 rounded-lg shadow-lg w-80">
          <Dialog.Title className="text-lg font-semibold">Aviso</Dialog.Title>
          <Dialog.Description className="mt-2">{modalMessage}</Dialog.Description>
          <button onClick={() => setIsModalOpen(false)} className="mt-4 btn btn-primary w-full">Aceptar</button>
        </div>
      </Dialog>
    </div>
  );
}

export default Analysis;
