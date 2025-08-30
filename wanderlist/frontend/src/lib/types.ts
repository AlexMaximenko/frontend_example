export interface Destination {
  id: string;
  name: string;
  country: string;
  rating: number;
  shortDescription: string;
  images: string[];
}

export interface DestinationListResponse {
  items: Destination[];
  page: number;
  totalPages: number;
  total: number;
}

export interface ContactFormData {
  name: string;
  email: string;
  message: string;
}

export interface ApiResponse<T> {
  data?: T;
  error?: string;
}