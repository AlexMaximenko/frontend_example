import React, { Suspense, lazy } from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import NavBar from './components/NavBar';
import Footer from './components/Footer';
import { FavoritesProvider } from './lib/FavoritesContext';

const Home = lazy(() => import('./routes/Home'));
const Browse = lazy(() => import('./routes/Browse'));
const Destination = lazy(() => import('./routes/Destination'));
const Favorites = lazy(() => import('./routes/Favorites'));
const Contact = lazy(() => import('./routes/Contact'));

const pageVariants = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
};

export default function App() {
  const location = useLocation();
  return (
    <FavoritesProvider>
      <div className="min-h-screen flex flex-col">
        <NavBar />
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            {[
              { path: '/', element: <Home /> },
              { path: '/browse', element: <Browse /> },
              { path: '/destinations/:id', element: <Destination /> },
              { path: '/favorites', element: <Favorites /> },
              { path: '/contact', element: <Contact /> },
            ].map(({ path, element }) => (
              <Route
                key={path}
                path={path}
                element={
                  <Suspense fallback={<div className="p-4">Loading...</div>}>
                    <motion.div
                      variants={pageVariants}
                      initial="initial"
                      animate="animate"
                      exit="exit"
                      transition={{ duration: 0.3 }}
                    >
                      {element}
                    </motion.div>
                  </Suspense>
                }
              />
            ))}
          </Routes>
        </AnimatePresence>
        <Footer />
      </div>
    </FavoritesProvider>
  );
}
