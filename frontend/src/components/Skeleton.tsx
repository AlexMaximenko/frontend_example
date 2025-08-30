import React from 'react';
import { motion } from 'framer-motion';

export default function Skeleton() {
  return (
    <motion.div
      className="animate-pulse bg-gray-200 h-40 w-full rounded"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    />
  );
}
