from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routers import destinations, contact
from app.models import HealthResponse

app = FastAPI(title="WanderList API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include routers
app.include_router(destinations.router, prefix="/api", tags=["destinations"])
app.include_router(contact.router, prefix="/api", tags=["contact"])


@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(ok=True)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to WanderList API", "docs": "/docs"}