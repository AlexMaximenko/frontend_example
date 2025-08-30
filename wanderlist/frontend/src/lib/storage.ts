const FAVORITES_KEY = 'wanderlist_favorites';

export const favoritesStorage = {
  getFavorites(): string[] {
    const stored = localStorage.getItem(FAVORITES_KEY);
    return stored ? JSON.parse(stored) : [];
  },

  addFavorite(id: string): void {
    const favorites = this.getFavorites();
    if (!favorites.includes(id)) {
      favorites.push(id);
      localStorage.setItem(FAVORITES_KEY, JSON.stringify(favorites));
    }
  },

  removeFavorite(id: string): void {
    const favorites = this.getFavorites();
    const filtered = favorites.filter((fav) => fav !== id);
    localStorage.setItem(FAVORITES_KEY, JSON.stringify(filtered));
  },

  toggleFavorite(id: string): boolean {
    const favorites = this.getFavorites();
    if (favorites.includes(id)) {
      this.removeFavorite(id);
      return false;
    } else {
      this.addFavorite(id);
      return true;
    }
  },

  isFavorite(id: string): boolean {
    return this.getFavorites().includes(id);
  },
};