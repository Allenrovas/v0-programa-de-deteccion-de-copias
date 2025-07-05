// api.js
import ApiService from './apiService';

// Crea una instancia única de ApiService con la URL base de tu API
const api = new ApiService(`http://${import.meta.env.VITE_BACKEND_HOST || 'localhost'}:${import.meta.env.VITE_BACKEND_PORT || 8000}`);

// Exporta la instancia para que pueda ser utilizada en toda la aplicación
export default api;