import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Destination } from '../lib/types';
import { getDestinations } from '../lib/api';
import Card from '../components/Card';

export default function Home() {
  const [items, setItems] = useState<Destination[]>([]);

  useEffect(() => {
    getDestinations({ sort: 'rating', limit: 4 }).then((res) =>
      setItems(res.items)
    );
  }, []);

  return (
    <div className="container mx-auto p-4 space-y-8">
      <motion.h1
        className="text-3xl md:text-5xl font-bold text-center"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        Discover your next adventure
      </motion.h1>
      <div className="text-center">
        <Link to="/browse" className="bg-blue-500 text-white px-4 py-2 rounded">
          Browse Destinations
        </Link>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
        {items.map((d) => (
          <Card key={d.id} dest={d} />
        ))}
      </div>
    </div>
  );
}
