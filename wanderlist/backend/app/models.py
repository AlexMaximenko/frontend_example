from pydantic import BaseModel, EmailStr
from typing import List, Optional


class Destination(BaseModel):
    id: str
    name: str
    country: str
    rating: float
    shortDescription: str
    images: List[str]


class DestinationList(BaseModel):
    items: List[Destination]
    page: int
    totalPages: int
    total: int


class ContactForm(BaseModel):
    name: str
    email: EmailStr
    message: str


class SuccessResponse(BaseModel):
    ok: bool


class HealthResponse(BaseModel):
    ok: bool