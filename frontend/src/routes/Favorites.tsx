import React, { useEffect, useState } from 'react';
import { useFavorites } from '../lib/FavoritesContext';
import { getDestinations } from '../lib/api';
import { Destination } from '../lib/types';
import Card from '../components/Card';

export default function FavoritesPage() {
  const { favorites } = useFavorites();
  const [items, setItems] = useState<Destination[]>([]);

  useEffect(() => {
    getDestinations({ limit: 100 }).then((res) => {
      setItems(res.items.filter((i) => favorites.includes(i.id)));
    });
  }, [favorites]);

  if (!items.length)
    return <div className="p-4">No favorites yet. Browse destinations!</div>;

  return (
    <div className="container mx-auto p-4 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
      {items.map((d) => (
        <Card key={d.id} dest={d} />
      ))}
    </div>
  );
}
