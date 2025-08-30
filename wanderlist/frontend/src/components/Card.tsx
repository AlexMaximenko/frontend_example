import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { favoritesStorage } from '../lib/storage';
import { formatImageUrl } from '../lib/api';
import type { Destination } from '../lib/types';
import Rating from './Rating';
import { useState } from 'react';

interface CardProps {
  destination: Destination;
  onFavoriteToggle?: () => void;
}

const Card: React.FC<CardProps> = ({ destination, onFavoriteToggle }) => {
  const [isFavorite, setIsFavorite] = useState(favoritesStorage.isFavorite(destination.id));

  const handleFavoriteClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const newState = favoritesStorage.toggleFavorite(destination.id);
    setIsFavorite(newState);
    onFavoriteToggle?.();
  };

  return (
    <motion.div
      whileHover={{ y: -4 }}
      whileTap={{ scale: 0.98 }}
      className="card overflow-hidden"
    >
      <Link to={`/destinations/${destination.id}`}>
        <div className="relative aspect-video">
          <img
            src={formatImageUrl(destination.images[0])}
            alt={destination.name}
            className="w-full h-full object-cover"
            loading="lazy"
          />
          <div className="absolute top-2 left-2 bg-white px-2 py-1 rounded-md text-sm font-medium">
            {destination.country}
          </div>
          <button
            onClick={handleFavoriteClick}
            className={`absolute top-2 right-2 p-2 rounded-full transition-colors ${
              isFavorite
                ? 'bg-red-500 text-white hover:bg-red-600'
                : 'bg-white text-gray-600 hover:text-red-500'
            }`}
            aria-label={isFavorite ? 'Remove from favorites' : 'Add to favorites'}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill={isFavorite ? 'currentColor' : 'none'}
              viewBox="0 0 24 24"
              strokeWidth={2}
              stroke="currentColor"
              className="w-5 h-5"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M21 8.5c0-2.485-2.075-4.5-4.64-4.5-1.792 0-3.349.996-4.11 2.451a1 1 0 01-1.5 0C9.99 4.996 8.432 4 6.64 4 4.075 4 2 6.015 2 8.5c0 1.09.425 2.077 1.11 2.808L12 21l8.89-9.692A4.441 4.441 0 0021 8.5z"
              />
            </svg>
          </button>
        </div>
        
        <div className="p-4">
          <h3 className="text-xl font-semibold mb-2">{destination.name}</h3>
          <Rating value={destination.rating} />
          <p className="text-gray-600 mt-2 line-clamp-2">{destination.shortDescription}</p>
        </div>
      </Link>
    </motion.div>
  );
};

export default Card;