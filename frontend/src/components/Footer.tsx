import React from 'react';

export default function Footer() {
  return (
    <footer className="bg-gray-100 mt-auto">
      <div className="container mx-auto p-4 text-center text-sm text-gray-500">
        © {new Date().getFullYear()} WanderList
      </div>
    </footer>
  );
}
