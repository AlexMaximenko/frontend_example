from fastapi import APIRouter
from app.models import ContactRequest, ContactResponse

router = APIRouter(prefix="/api", tags=["contact"])


@router.post("/contact", response_model=ContactResponse)
def submit_contact(payload: ContactRequest):
    print("Contact message:", payload.model_dump())
    return ContactResponse()
