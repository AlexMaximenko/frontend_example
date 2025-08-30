from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.models import DestinationList, Destination
from app.data import get_destinations, get_destination_by_id

router = APIRouter()


@router.get("/destinations", response_model=DestinationList)
async def list_destinations(
    q: Optional[str] = Query(None, description="Search query"),
    country: Optional[str] = Query(None, description="Filter by country"),
    sort: Optional[str] = Query(None, regex="^(name|rating)$", description="Sort by name or rating"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(8, ge=1, le=50, description="Items per page")
):
    """Get paginated list of destinations with optional filters"""
    result = get_destinations(q=q, country=country, sort=sort, page=page, limit=limit)
    return DestinationList(**result)


@router.get("/destinations/{destination_id}", response_model=Destination)
async def get_destination(destination_id: str):
    """Get a specific destination by ID"""
    destination = get_destination_by_id(destination_id)
    if not destination:
        raise HTTPException(status_code=404, detail="Destination not found")
    return destination