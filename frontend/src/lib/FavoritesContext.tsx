import React, { createContext, useContext, useState } from 'react';
import { getFavorites, toggleFavorite } from './storage';

interface FavoritesContextValue {
  favorites: string[];
  toggle: (id: string) => void;
}

const FavoritesContext = createContext<FavoritesContextValue | undefined>(
  undefined
);

export const FavoritesProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [favorites, setFavorites] = useState<string[]>(getFavorites());

  const toggle = (id: string) => {
    setFavorites(toggleFavorite(id));
  };

  return (
    <FavoritesContext.Provider value={{ favorites, toggle }}>
      {children}
    </FavoritesContext.Provider>
  );
};

export function useFavorites() {
  const ctx = useContext(FavoritesContext);
  if (!ctx)
    throw new Error('useFavorites must be used within FavoritesProvider');
  return ctx;
}
