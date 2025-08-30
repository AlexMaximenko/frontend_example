from fastapi import APIRouter
import logging
from app.models import ContactForm, SuccessResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/contact", response_model=SuccessResponse)
async def submit_contact(form: ContactForm):
    """Submit a contact form"""
    logger.info(f"Contact form received - Name: {form.name}, Email: {form.email}, Message: {form.message}")
    print(f"Contact form submission:")
    print(f"  Name: {form.name}")
    print(f"  Email: {form.email}")
    print(f"  Message: {form.message}")
    return SuccessResponse(ok=True)