from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.models import Destination, DestinationListResponse
from app.data import destinations

router = APIRouter(prefix="/api/destinations", tags=["destinations"])


@router.get("", response_model=DestinationListResponse)
def list_destinations(
    q: Optional[str] = None,
    country: Optional[str] = None,
    sort: Optional[str] = Query(None, pattern="^(name|rating)$"),
    page: int = 1,
    limit: int = 6,
):
    items = destinations.copy()

    if q:
        items = [d for d in items if q.lower() in d["name"].lower()]
    if country:
        items = [d for d in items if d["country"].lower() == country.lower()]
    if sort:
        reverse = sort == "rating"
        items.sort(key=lambda x: x[sort], reverse=reverse)

    total = len(items)
    total_pages = max((total - 1) // limit + 1, 1)
    start = (page - 1) * limit
    paged = items[start : start + limit]

    return DestinationListResponse(
        items=[Destination(**d) for d in paged],
        page=page,
        totalPages=total_pages,
        total=total,
    )


@router.get("/{dest_id}", response_model=Destination)
def get_destination(dest_id: str):
    for d in destinations:
        if d["id"] == dest_id:
            return Destination(**d)
    raise HTTPException(status_code=404, detail="Destination not found")
