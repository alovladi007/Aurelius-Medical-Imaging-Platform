import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60 seconds for medical image processing
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// API methods
export const api = {
  // Health check
  healthCheck: async () => {
    const response = await apiClient.get('/health');
    return response.data;
  },

  // Model info
  getModelInfo: async () => {
    const response = await apiClient.get('/model_info');
    return response.data;
  },

  // Single prediction
  predict: async (imageFile, clinicalData = {}) => {
    const formData = new FormData();
    formData.append('image', imageFile);

    // Add clinical data fields
    if (clinicalData.clinical_notes) {
      formData.append('clinical_notes', clinicalData.clinical_notes);
    }
    if (clinicalData.patient_age) {
      formData.append('patient_age', clinicalData.patient_age);
    }
    if (clinicalData.patient_gender) {
      formData.append('patient_gender', clinicalData.patient_gender);
    }
    if (clinicalData.smoking_history !== undefined) {
      formData.append('smoking_history', clinicalData.smoking_history);
    }
    if (clinicalData.family_history !== undefined) {
      formData.append('family_history', clinicalData.family_history);
    }

    const response = await apiClient.post('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Batch prediction
  batchPredict: async (imageFiles) => {
    const formData = new FormData();
    imageFiles.forEach((file) => {
      formData.append('images', file);
    });

    const response = await apiClient.post('/batch_predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
};

export default apiClient;
