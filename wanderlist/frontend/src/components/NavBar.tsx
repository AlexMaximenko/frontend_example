import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';

const NavBar = () => {
  const location = useLocation();
  
  const isActive = (path: string) => location.pathname === path;
  
  return (
    <nav className="bg-white shadow-md sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <Link to="/" className="text-2xl font-bold text-blue-600">
            WanderList
          </Link>
          
          <ul className="flex space-x-6">
            {[
              { path: '/', label: 'Home' },
              { path: '/browse', label: 'Browse' },
              { path: '/favorites', label: 'Favorites' },
              { path: '/contact', label: 'Contact' },
            ].map((item) => (
              <li key={item.path}>
                <Link
                  to={item.path}
                  className={`relative px-2 py-1 transition-colors ${
                    isActive(item.path) ? 'text-blue-600' : 'text-gray-600 hover:text-blue-600'
                  }`}
                >
                  {item.label}
                  {isActive(item.path) && (
                    <motion.div
                      layoutId="navbar-indicator"
                      className="absolute -bottom-1 left-0 right-0 h-0.5 bg-blue-600"
                      initial={false}
                    />
                  )}
                </Link>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default NavBar;