const KEY = 'favorites';

export function getFavorites(): string[] {
  try {
    return JSON.parse(localStorage.getItem(KEY) || '[]');
  } catch {
    return [];
  }
}

export function toggleFavorite(id: string): string[] {
  const current = new Set(getFavorites());
  if (current.has(id)) {
    current.delete(id);
  } else {
    current.add(id);
  }
  const arr = Array.from(current);
  localStorage.setItem(KEY, JSON.stringify(arr));
  return arr;
}

export function isFavorite(id: string): boolean {
  return getFavorites().includes(id);
}
