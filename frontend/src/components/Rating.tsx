import React from 'react';

export default function Rating({ value }: { value: number }) {
  const stars = Array.from({ length: 5 }, (_, i) =>
    i < Math.round(value) ? '★' : '☆'
  );
  return (
    <div className="text-yellow-500" aria-label={`rating ${value}`}>
      {stars.join('')}
    </div>
  );
}
