import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import Card from '../components/Card';
import { CardSkeleton } from '../components/Skeleton';
import { destinationsApi } from '../lib/api';
import type { Destination } from '../lib/types';

const Browse = () => {
  const [destinations, setDestinations] = useState<Destination[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCountry, setSelectedCountry] = useState('');
  const [sortBy, setSortBy] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [countries, setCountries] = useState<string[]>([]);

  // Debounce timer
  const [debounceTimer, setDebounceTimer] = useState<NodeJS.Timeout | null>(null);

  const fetchDestinations = useCallback(async () => {
    setLoading(true);
    try {
      const response = await destinationsApi.getList({
        q: searchQuery,
        country: selectedCountry,
        sort: sortBy,
        page: currentPage,
        limit: 9,
      });
      setDestinations(response.items);
      setTotalPages(response.totalPages);
    } catch (error) {
      console.error('Failed to fetch destinations:', error);
    } finally {
      setLoading(false);
    }
  }, [searchQuery, selectedCountry, sortBy, currentPage]);

  // Fetch initial data and countries
  useEffect(() => {
    const fetchCountries = async () => {
      try {
        const response = await destinationsApi.getList({ limit: 100 });
        const uniqueCountries = [...new Set(response.items.map((d) => d.country))].sort();
        setCountries(uniqueCountries);
      } catch (error) {
        console.error('Failed to fetch countries:', error);
      }
    };
    fetchCountries();
  }, []);

  // Fetch destinations when filters change
  useEffect(() => {
    fetchDestinations();
  }, [fetchDestinations]);

  // Handle search with debounce
  const handleSearchChange = (value: string) => {
    setSearchQuery(value);
    setCurrentPage(1);
    
    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }
    
    const timer = setTimeout(() => {
      fetchDestinations();
    }, 300);
    
    setDebounceTimer(timer);
  };

  const handleFilterChange = (type: 'country' | 'sort', value: string) => {
    if (type === 'country') {
      setSelectedCountry(value);
    } else {
      setSortBy(value);
    }
    setCurrentPage(1);
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
      className="container mx-auto px-4 py-8"
    >
      <h1 className="text-4xl font-bold mb-8">Browse Destinations</h1>

      {/* Filters */}
      <div className="bg-white p-6 rounded-lg shadow-md mb-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Search */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Search destinations
            </label>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => handleSearchChange(e.target.value)}
              placeholder="Search by name or country..."
              className="input"
            />
          </div>

          {/* Country Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Filter by country
            </label>
            <select
              value={selectedCountry}
              onChange={(e) => handleFilterChange('country', e.target.value)}
              className="input"
            >
              <option value="">All countries</option>
              {countries.map((country) => (
                <option key={country} value={country}>
                  {country}
                </option>
              ))}
            </select>
          </div>

          {/* Sort */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Sort by</label>
            <select
              value={sortBy}
              onChange={(e) => handleFilterChange('sort', e.target.value)}
              className="input"
            >
              <option value="">Default</option>
              <option value="name">Name (A-Z)</option>
              <option value="rating">Rating (High to Low)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Results Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        {loading
          ? [...Array(9)].map((_, i) => <CardSkeleton key={i} />)
          : destinations.map((destination) => (
              <motion.div
                key={destination.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
              >
                <Card destination={destination} />
              </motion.div>
            ))}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex justify-center space-x-2">
          <button
            onClick={() => setCurrentPage((prev) => Math.max(1, prev - 1))}
            disabled={currentPage === 1}
            className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Previous
          </button>
          
          <span className="flex items-center px-4">
            Page {currentPage} of {totalPages}
          </span>
          
          <button
            onClick={() => setCurrentPage((prev) => Math.min(totalPages, prev + 1))}
            disabled={currentPage === totalPages}
            className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next
          </button>
        </div>
      )}
    </motion.div>
  );
};

export default Browse;