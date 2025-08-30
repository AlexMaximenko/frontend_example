from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routers import destinations, contact

app = FastAPI(title="WanderList API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(destinations.router)
app.include_router(contact.router)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/healthz")
def healthz():
    return {"ok": True}
