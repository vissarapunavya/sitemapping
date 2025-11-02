import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const checkHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    return null;
  }
};

export const getAvailableModels = async () => {
  try {
    const response = await api.get('/models');
    return response.data.models;
  } catch (error) {
    console.error('Failed to get models:', error);
    return [];
  }
};

export const predictImage = async (imageFile, modelType) => {
  try {
    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('model_type', modelType);
    
    const response = await api.post('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  } catch (error) {
    console.error('Prediction failed:', error);
    throw error;
  }
};
