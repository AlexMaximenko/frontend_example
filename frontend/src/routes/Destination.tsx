import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { getDestination } from '../lib/api';
import { Destination } from '../lib/types';
import ImageCarousel from '../components/ImageCarousel';
import Rating from '../components/Rating';
import { useFavorites } from '../lib/FavoritesContext';

export default function DestinationPage() {
  const { id } = useParams();
  const [dest, setDest] = useState<Destination | null>(null);
  const { favorites, toggle } = useFavorites();

  useEffect(() => {
    if (id) getDestination(id).then(setDest);
  }, [id]);

  if (!dest) return <div className="p-4">Loading...</div>;

  const fav = favorites.includes(dest.id);

  return (
    <div className="container mx-auto p-4 space-y-4">
      <ImageCarousel images={dest.images} />
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">{dest.name}</h1>
        <button
          className={`px-3 py-1 rounded ${fav ? 'bg-red-500 text-white' : 'bg-gray-200'}`}
          onClick={() => toggle(dest.id)}
        >
          {fav ? 'Remove Favorite' : 'Add to Favorites'}
        </button>
      </div>
      <span className="inline-block bg-blue-100 text-blue-600 px-2 py-1 rounded text-xs">
        {dest.country}
      </span>
      <Rating value={dest.rating} />
      <p>{dest.shortDescription}</p>
    </div>
  );
}
