const Footer = () => {
  return (
    <footer className="bg-gray-800 text-white py-8 mt-16">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <h3 className="text-lg font-semibold mb-4">About WanderList</h3>
            <p className="text-gray-300">
              Discover amazing destinations around the world and plan your next adventure.
            </p>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-4">Quick Links</h3>
            <ul className="space-y-2 text-gray-300">
              <li>
                <a href="/browse" className="hover:text-white transition-colors">
                  Browse Destinations
                </a>
              </li>
              <li>
                <a href="/favorites" className="hover:text-white transition-colors">
                  My Favorites
                </a>
              </li>
              <li>
                <a href="/contact" className="hover:text-white transition-colors">
                  Contact Us
                </a>
              </li>
            </ul>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-4">Follow Us</h3>
            <p className="text-gray-300">
              Stay updated with the latest travel tips and destinations.
            </p>
          </div>
        </div>
        
        <div className="mt-8 pt-8 border-t border-gray-700 text-center text-gray-400">
          <p>&copy; 2024 WanderList. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;