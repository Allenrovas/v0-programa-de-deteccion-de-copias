import React, { useState } from 'react';
import { XIcon } from '@heroicons/react/outline';
import { Dialog } from '@headlessui/react';

function Feedback() {
  const [comment, setComment] = useState('');
  const [images, setImages] = useState([]);
  const [imagePreviews, setImagePreviews] = useState([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalMessage, setModalMessage] = useState('');

  const handleCommentChange = (event) => {
    setComment(event.target.value);
  };

  const handleImageChange = (event) => {
    const selectedImages = Array.from(event.target.files);
    setImages((prevImages) => [...prevImages, ...selectedImages]);

    const previews = selectedImages.map((image) => URL.createObjectURL(image));
    setImagePreviews((prevPreviews) => [...prevPreviews, ...previews]);
  };

  const handleRemoveImage = (index) => {
    setImages((prevImages) => prevImages.filter((_, i) => i !== index));
    setImagePreviews((prevPreviews) => prevPreviews.filter((_, i) => i !== index));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    if (!comment) {
      setModalMessage('Por favor, escribe un comentario antes de enviar.');
      setIsModalOpen(true);
      return;
    }
    console.log('Comentario:', comment);
    console.log('Imágenes:', images);
  };

  return (
    <div className="feedback-page p-8 bg-base-200 text-base-content min-h-screen flex items-center justify-center">
      <div className="max-w-lg w-full bg-base-100 shadow-lg rounded-lg p-6">
        <h1 className="text-3xl font-bold mb-6 text-center text-primary">Enviar retroalimentación</h1>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium mb-2">Comentario</label>
            <textarea
              value={comment}
              onChange={handleCommentChange}
              className="textarea textarea-bordered w-full"
              placeholder="Escribe tu comentario aquí..."
              rows="4"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Subir Imágenes</label>
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={handleImageChange}
              className="file-input file-input-bordered w-full"
            />
          </div>
          {imagePreviews.length > 0 && (
            <div className="mt-4 grid grid-cols-2 gap-4">
              {imagePreviews.map((src, index) => (
                <div key={index} className="relative">
                  <img src={src} alt={`Preview ${index}`} className="w-full h-32 object-cover rounded-lg" />
                  <button
                    type="button"
                    onClick={() => handleRemoveImage(index)}
                    className="absolute top-1 right-1 bg-red-500 text-white rounded-full p-1 hover:bg-red-700"
                  >
                    <XIcon className="w-5 h-5" />
                  </button>
                </div>
              ))}
            </div>
          )}
          <button type="submit" className="btn btn-primary w-full">Enviar Feedback</button>
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

export default Feedback;
