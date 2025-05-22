// api.js
import ApiService from './apiService';

// Crea una instancia única de ApiService con la URL base de tu API
const api = new ApiService(import.meta.env.VITE_API_URL || 'http://localhost:8000');

// Exporta la instancia para que pueda ser utilizada en toda la aplicación
export default api;