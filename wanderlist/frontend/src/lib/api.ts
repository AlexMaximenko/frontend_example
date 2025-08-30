import axios from 'axios';
import type { Destination, DestinationListResponse, ContactFormData } from './types';

const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const destinationsApi = {
  async getList(params?: {
    q?: string;
    country?: string;
    sort?: string;
    page?: number;
    limit?: number;
  }): Promise<DestinationListResponse> {
    const response = await api.get('/destinations', { params });
    return response.data;
  },

  async getById(id: string): Promise<Destination> {
    const response = await api.get(`/destinations/${id}`);
    return response.data;
  },
};

export const contactApi = {
  async submit(data: ContactFormData): Promise<{ ok: boolean }> {
    const response = await api.post('/contact', data);
    return response.data;
  },
};

export const formatImageUrl = (url: string): string => {
  if (url.startsWith('http')) return url;
  return `http://localhost:8000${url}`;
};