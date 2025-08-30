import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import ImageCarousel from '../components/ImageCarousel';
import Rating from '../components/Rating';
import Skeleton from '../components/Skeleton';
import { destinationsApi } from '../lib/api';
import { favoritesStorage } from '../lib/storage';
import type { Destination as DestinationType } from '../lib/types';

const Destination = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [destination, setDestination] = useState<DestinationType | null>(null);
  const [loading, setLoading] = useState(true);
  const [isFavorite, setIsFavorite] = useState(false);

  useEffect(() => {
    const fetchDestination = async () => {
      if (!id) return;
      
      try {
        const data = await destinationsApi.getById(id);
        setDestination(data);
        setIsFavorite(favoritesStorage.isFavorite(id));
      } catch (error) {
        console.error('Failed to fetch destination:', error);
        navigate('/browse');
      } finally {
        setLoading(false);
      }
    };

    fetchDestination();
  }, [id, navigate]);

  const handleFavoriteToggle = () => {
    if (!id) return;
    const newState = favoritesStorage.toggleFavorite(id);
    setIsFavorite(newState);
  };

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <Skeleton className="h-96 mb-8" />
        <div className="max-w-4xl mx-auto">
          <Skeleton className="h-12 w-3/4 mb-4" />
          <Skeleton className="h-6 w-1/4 mb-4" />
          <Skeleton className="h-24 w-full" />
        </div>
      </div>
    );
  }

  if (!destination) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
    >
      {/* Hero Image */}
      <div className="relative h-[50vh] overflow-hidden">
        <img
          src={`http://localhost:8000${destination.images[0]}`}
          alt={destination.name}
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent" />
        <div className="absolute bottom-0 left-0 right-0 p-8 text-white">
          <div className="container mx-auto">
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-5xl font-bold mb-2"
            >
              {destination.name}
            </motion.h1>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="text-xl"
            >
              {destination.country}
            </motion.p>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Actions Bar */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="flex items-center justify-between mb-8"
          >
            <Rating value={destination.rating} />
            <button
              onClick={handleFavoriteToggle}
              className={`flex items-center space-x-2 px-6 py-2 rounded-lg transition-colors ${
                isFavorite
                  ? 'bg-red-500 text-white hover:bg-red-600'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
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
              <span>{isFavorite ? 'Remove from Favorites' : 'Add to Favorites'}</span>
            </button>
          </motion.div>

          {/* Description */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="mb-12"
          >
            <h2 className="text-2xl font-semibold mb-4">About this destination</h2>
            <p className="text-gray-600 text-lg leading-relaxed">{destination.shortDescription}</p>
          </motion.div>

          {/* Image Gallery */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <h2 className="text-2xl font-semibold mb-4">Gallery</h2>
            <ImageCarousel images={destination.images} alt={destination.name} />
          </motion.div>

          {/* Back Button */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="mt-12 text-center"
          >
            <button
              onClick={() => navigate(-1)}
              className="btn-secondary"
            >
              Back to Browse
            </button>
          </motion.div>
        </div>
      </div>
    </motion.div>
  );
};

export default Destination;