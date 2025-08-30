import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Props {
  images: string[];
}

export default function ImageCarousel({ images }: Props) {
  const [index, setIndex] = useState(0);
  const next = () => setIndex((i) => (i + 1) % images.length);
  const prev = () => setIndex((i) => (i - 1 + images.length) % images.length);

  return (
    <div className="relative w-full overflow-hidden">
      <AnimatePresence initial={false} mode="wait">
        <motion.img
          key={images[index]}
          src={images[index]}
          alt="gallery"
          className="w-full h-64 object-cover rounded"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
        />
      </AnimatePresence>
      {images.length > 1 && (
        <div className="absolute inset-0 flex items-center justify-between p-2">
          <button
            onClick={prev}
            className="bg-white/70 rounded-full px-2 py-1 text-sm"
          >
            ‹
          </button>
          <button
            onClick={next}
            className="bg-white/70 rounded-full px-2 py-1 text-sm"
          >
            ›
          </button>
        </div>
      )}
    </div>
  );
}
