import React from 'react';
import { Link, NavLink } from 'react-router-dom';

export default function NavBar() {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-2 rounded ${isActive ? 'text-blue-500' : 'text-gray-600'}`;
  return (
    <nav className="bg-white shadow">
      <div className="container mx-auto flex items-center justify-between p-4">
        <Link to="/" className="text-xl font-bold text-blue-500">
          WanderList
        </Link>
        <div className="space-x-2">
          <NavLink to="/browse" className={linkClass}>
            Browse
          </NavLink>
          <NavLink to="/favorites" className={linkClass}>
            Favorites
          </NavLink>
          <NavLink to="/contact" className={linkClass}>
            Contact
          </NavLink>
        </div>
      </div>
    </nav>
  );
}
