import axios from 'axios';
import type { Destination, DestinationList } from './types';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
});

export async function getDestinations(
  params: Record<string, unknown>
): Promise<DestinationList> {
  const { data } = await api.get<DestinationList>('/api/destinations', {
    params,
  });
  return data;
}

export async function getDestination(id: string): Promise<Destination> {
  const { data } = await api.get<Destination>(`/api/destinations/${id}`);
  return data;
}

export async function sendContact(payload: {
  name: string;
  email: string;
  message: string;
}): Promise<{ ok: boolean }> {
  const { data } = await api.post('/api/contact', payload);
  return data;
}
