from pydantic import BaseModel, EmailStr, Field
from typing import List


class Destination(BaseModel):
    id: str
    name: str
    country: str
    rating: float = Field(ge=0, le=5)
    shortDescription: str
    images: List[str]


class DestinationListResponse(BaseModel):
    items: List[Destination]
    page: int
    totalPages: int
    total: int


class ContactRequest(BaseModel):
    name: str
    email: EmailStr
    message: str


class ContactResponse(BaseModel):
    ok: bool = True
