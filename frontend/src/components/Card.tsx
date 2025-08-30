import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Destination } from '../lib/types';
import Rating from './Rating';
import { useFavorites } from '../lib/FavoritesContext';

interface Props {
  dest: Destination;
}

export default function Card({ dest }: Props) {
  const { favorites, toggle } = useFavorites();
  const fav = favorites.includes(dest.id);
  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className="bg-white rounded shadow overflow-hidden"
    >
      <Link to={`/destinations/${dest.id}`}>
        <img
          src={dest.images[0]}
          alt={dest.name}
          className="w-full h-40 object-cover"
        />
      </Link>
      <div className="p-4 space-y-2">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold">
            <Link to={`/destinations/${dest.id}`}>{dest.name}</Link>
          </h3>
          <button
            aria-label="favorite"
            className={`text-xl ${fav ? 'text-red-500' : 'text-gray-400'}`}
            onClick={() => toggle(dest.id)}
          >
            {fav ? '♥' : '♡'}
          </button>
        </div>
        <span className="inline-block bg-blue-100 text-blue-600 px-2 py-1 rounded text-xs">
          {dest.country}
        </span>
        <Rating value={dest.rating} />
      </div>
    </motion.div>
  );
}
