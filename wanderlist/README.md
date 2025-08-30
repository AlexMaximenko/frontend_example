# WanderList - Travel Destination Explorer

A modern, responsive travel destination explorer built with React, TypeScript, and FastAPI. Browse beautiful destinations, save your favorites, and plan your next adventure!

## 🚀 Quick Start (Recommended)

### Prerequisites
- Docker and Docker Compose installed
- Node.js 20+ and Python 3.11+ (for manual setup)

### Run with Docker Compose

```bash
# Clone the repository
git clone <repository-url>
cd wanderlist

# Start both services
make up

# Or directly with docker-compose
docker compose up --build
```

The application will be available at:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation (Swagger): http://localhost:8000/docs

### Stop the services

```bash
make down
# Or
docker compose down
```

## 💻 Manual Setup (Without Docker)

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm ci

# Run development server
npm run dev
```

## 🛠️ Development Commands

### Makefile Commands

```bash
make up       # Start all services with Docker
make down     # Stop all services
make fmt      # Format code (Prettier for frontend, ruff for backend)
make lint     # Run linters (ESLint for frontend, ruff for backend)
```

### Frontend Commands

```bash
npm run dev      # Start development server
npm run build    # Build for production
npm run preview  # Preview production build
npm run lint     # Run ESLint
npm run format   # Format code with Prettier
```

### Backend Commands

```bash
uvicorn app.main:app --reload  # Run with hot reload
python -m ruff check .          # Run linter
python -m ruff format .         # Format code
```

## 📁 Project Structure

```
wanderlist/
├── README.md
├── docker-compose.yml
├── Makefile
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI application entry
│   │   ├── data.py           # In-memory data store
│   │   ├── models.py         # Pydantic models
│   │   ├── routers/
│   │   │   ├── destinations.py
│   │   │   └── contact.py
│   │   └── static/
│   │       └── images/       # Sample destination images
│   ├── requirements.txt
│   └── Dockerfile
│
└── frontend/
    ├── index.html
    ├── vite.config.ts
    ├── tsconfig.json
    ├── tailwind.config.js
    ├── package.json
    ├── Dockerfile
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── routes/           # Page components
        │   ├── Home.tsx
        │   ├── Browse.tsx
        │   ├── Destination.tsx
        │   ├── Favorites.tsx
        │   └── Contact.tsx
        ├── components/       # Reusable components
        │   ├── NavBar.tsx
        │   ├── Footer.tsx
        │   ├── Card.tsx
        │   ├── Rating.tsx
        │   ├── ImageCarousel.tsx
        │   ├── Skeleton.tsx
        │   └── Toast.tsx
        ├── lib/              # Utilities and API
        │   ├── api.ts
        │   ├── storage.ts
        │   └── types.ts
        └── styles/
            └── index.css
```

## 🌟 Features

- **Browse Destinations**: Search, filter by country, and sort destinations
- **Destination Details**: View detailed information, image galleries, and ratings
- **Favorites**: Save and manage your favorite destinations (stored in localStorage)
- **Contact Form**: Submit inquiries with client-side validation
- **Responsive Design**: Optimized for mobile, tablet, and desktop
- **Smooth Animations**: Page transitions and micro-interactions with Framer Motion
- **Type Safety**: Full TypeScript support for better developer experience

## 🔧 Configuration

### Environment Variables

Currently, the application uses default values. To customize:

**Backend** (create `.env` in `/backend`):
```env
CORS_ORIGINS=["http://localhost:5173"]
```

**Frontend** (create `.env` in `/frontend`):
```env
VITE_API_URL=http://localhost:8000
```

### Adding More Destinations

Edit `backend/app/data.py` to add more destinations to the in-memory store:

```python
destinations_data.append(
    Destination(
        id="21",
        name="New Destination",
        country="Country",
        rating=4.5,
        shortDescription="Description here",
        images=["/static/images/dest1.jpg", "/static/images/dest2.jpg"]
    )
)
```

## 🐛 Troubleshooting

### Port Already in Use

If ports 5173 or 8000 are already in use:

```bash
# Find process using the port
lsof -i :5173  # or :8000

# Kill the process
kill -9 <PID>
```

### CORS Issues

Ensure the backend CORS configuration includes your frontend URL:

```python
# backend/app/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Docker Build Issues

```bash
# Clean rebuild
docker compose down
docker compose build --no-cache
docker compose up
```

## 📝 API Documentation

Once the backend is running, visit http://localhost:8000/docs for interactive API documentation.

### Main Endpoints

- `GET /api/destinations` - List destinations with pagination and filters
- `GET /api/destinations/{id}` - Get destination details
- `POST /api/contact` - Submit contact form
- `GET /healthz` - Health check endpoint

## 🚦 Testing

### Frontend
```bash
cd frontend
npm test  # If tests are configured
```

### Backend
```bash
cd backend
python -m pytest  # If tests are configured
```

## 📄 License

This project is created for educational purposes.