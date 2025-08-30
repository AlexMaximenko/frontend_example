export interface Destination {
  id: string;
  name: string;
  country: string;
  rating: number;
  shortDescription: string;
  images: string[];
}

export interface DestinationList {
  items: Destination[];
  page: number;
  totalPages: number;
  total: number;
}
