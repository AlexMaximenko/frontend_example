import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import Card from '../components/Card';
import { CardSkeleton } from '../components/Skeleton';
import { destinationsApi } from '../lib/api';
import { favoritesStorage } from '../lib/storage';
import type { Destination } from '../lib/types';

const Favorites = () => {
  const [favorites, setFavorites] = useState<Destination[]>([]);
  const [loading, setLoading] = useState(true);

  const loadFavorites = async () => {
    setLoading(true);
    try {
      const favoriteIds = favoritesStorage.getFavorites();
      const favoriteDestinations: Destination[] = [];
      
      // Fetch each favorite destination
      for (const id of favoriteIds) {
        try {
          const destination = await destinationsApi.getById(id);
          favoriteDestinations.push(destination);
        } catch (error) {
          console.error(`Failed to fetch favorite ${id}:`, error);
          // Remove invalid favorite
          favoritesStorage.removeFavorite(id);
        }
      }
      
      setFavorites(favoriteDestinations);
    } catch (error) {
      console.error('Failed to load favorites:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadFavorites();
  }, []);

  const handleFavoriteToggle = () => {
    // Refresh the list when a favorite is toggled
    loadFavorites();
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
      className="container mx-auto px-4 py-8"
    >
      <h1 className="text-4xl font-bold mb-8">My Favorite Destinations</h1>

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[...Array(6)].map((_, i) => (
            <CardSkeleton key={i} />
          ))}
        </div>
      ) : favorites.length === 0 ? (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center py-16"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
            className="w-24 h-24 mx-auto text-gray-400 mb-4"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M21 8.5c0-2.485-2.075-4.5-4.64-4.5-1.792 0-3.349.996-4.11 2.451a1 1 0 01-1.5 0C9.99 4.996 8.432 4 6.64 4 4.075 4 2 6.015 2 8.5c0 1.09.425 2.077 1.11 2.808L12 21l8.89-9.692A4.441 4.441 0 0021 8.5z"
            />
          </svg>
          <h2 className="text-2xl font-semibold mb-4 text-gray-600">No favorites yet</h2>
          <p className="text-gray-500 mb-8">
            Start exploring and add your favorite destinations to this list!
          </p>
          <Link to="/browse" className="btn-primary">
            Browse Destinations
          </Link>
        </motion.div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {favorites.map((destination, index) => (
            <motion.div
              key={destination.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Card destination={destination} onFavoriteToggle={handleFavoriteToggle} />
            </motion.div>
          ))}
        </div>
      )}
    </motion.div>
  );
};

export default Favorites;