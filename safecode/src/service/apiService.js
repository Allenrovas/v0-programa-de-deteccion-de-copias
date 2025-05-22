import axios from 'axios';

class ApiService {
  constructor(baseURL) {
    this.api = axios.create({
      baseURL: baseURL, 
      // No establecer Content-Type globalmente
    });

    this.prefix = 'api';
  }

  get(endpoint, params = {}) {
    return this.api.get(this.prefix + endpoint, { params }, { headers: { 'Content-Type': 'application/json' } });
  }

  post(endpoint, data, headers = {}) {
    return this.api.post(this.prefix + endpoint, data, { headers });
  }

  put(endpoint, data) {
    return this.api.put(this.prefix + endpoint, data);
  }

  delete(endpoint) {
    return this.api.delete(this.prefix + endpoint);
  }

  uploadFiles(endpoint, files) {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });

    return this.api.post(this.prefix + endpoint, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  }
}

export default ApiService;
