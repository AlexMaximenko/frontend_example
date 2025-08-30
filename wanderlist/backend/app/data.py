from typing import List, Optional
from app.models import Destination

destinations_data: List[Destination] = [
    Destination(
        id="1",
        name="Santorini",
        country="Greece",
        rating=4.8,
        shortDescription="Stunning white-washed buildings and breathtaking sunsets over the Aegean Sea.",
        images=["/static/images/santorini1.jpg", "/static/images/santorini2.jpg", "/static/images/santorini3.jpg"]
    ),
    Destination(
        id="2",
        name="Kyoto",
        country="Japan",
        rating=4.9,
        shortDescription="Ancient temples, traditional gardens, and the iconic bamboo groves.",
        images=["/static/images/kyoto1.jpg", "/static/images/kyoto2.jpg", "/static/images/kyoto3.jpg"]
    ),
    Destination(
        id="3",
        name="Reykjavik",
        country="Iceland",
        rating=4.7,
        shortDescription="Gateway to natural wonders, from geysers to the Northern Lights.",
        images=["/static/images/reykjavik1.jpg", "/static/images/reykjavik2.jpg", "/static/images/reykjavik3.jpg"]
    ),
    Destination(
        id="4",
        name="Marrakech",
        country="Morocco",
        rating=4.6,
        shortDescription="Vibrant souks, stunning palaces, and the enchanting Medina.",
        images=["/static/images/marrakech1.jpg", "/static/images/marrakech2.jpg", "/static/images/marrakech3.jpg"]
    ),
    Destination(
        id="5",
        name="Bali",
        country="Indonesia",
        rating=4.8,
        shortDescription="Tropical paradise with ancient temples, rice terraces, and pristine beaches.",
        images=["/static/images/bali1.jpg", "/static/images/bali2.jpg", "/static/images/bali3.jpg"]
    ),
    Destination(
        id="6",
        name="Barcelona",
        country="Spain",
        rating=4.7,
        shortDescription="Gaudí's architectural masterpieces meet Mediterranean beach culture.",
        images=["/static/images/barcelona1.jpg", "/static/images/barcelona2.jpg", "/static/images/barcelona3.jpg"]
    ),
    Destination(
        id="7",
        name="Dubai",
        country="UAE",
        rating=4.5,
        shortDescription="Futuristic skyline, luxury shopping, and desert adventures.",
        images=["/static/images/dubai1.jpg", "/static/images/dubai2.jpg", "/static/images/dubai3.jpg"]
    ),
    Destination(
        id="8",
        name="Cape Town",
        country="South Africa",
        rating=4.9,
        shortDescription="Stunning coastline, Table Mountain, and vibrant cultural heritage.",
        images=["/static/images/capetown1.jpg", "/static/images/capetown2.jpg", "/static/images/capetown3.jpg"]
    ),
    Destination(
        id="9",
        name="Prague",
        country="Czech Republic",
        rating=4.7,
        shortDescription="Medieval charm with Gothic architecture and cobblestone streets.",
        images=["/static/images/prague1.jpg", "/static/images/prague2.jpg", "/static/images/prague3.jpg"]
    ),
    Destination(
        id="10",
        name="Machu Picchu",
        country="Peru",
        rating=4.9,
        shortDescription="Ancient Incan citadel high in the Andes Mountains.",
        images=["/static/images/machupicchu1.jpg", "/static/images/machupicchu2.jpg", "/static/images/machupicchu3.jpg"]
    ),
    Destination(
        id="11",
        name="Amsterdam",
        country="Netherlands",
        rating=4.6,
        shortDescription="Picturesque canals, world-class museums, and cycling culture.",
        images=["/static/images/amsterdam1.jpg", "/static/images/amsterdam2.jpg", "/static/images/amsterdam3.jpg"]
    ),
    Destination(
        id="12",
        name="Sydney",
        country="Australia",
        rating=4.8,
        shortDescription="Iconic Opera House, stunning beaches, and vibrant harbor life.",
        images=["/static/images/sydney1.jpg", "/static/images/sydney2.jpg", "/static/images/sydney3.jpg"]
    ),
    Destination(
        id="13",
        name="Rome",
        country="Italy",
        rating=4.8,
        shortDescription="Eternal city filled with ancient ruins, Renaissance art, and delicious cuisine.",
        images=["/static/images/rome1.jpg", "/static/images/rome2.jpg", "/static/images/rome3.jpg"]
    ),
    Destination(
        id="14",
        name="Istanbul",
        country="Turkey",
        rating=4.7,
        shortDescription="Where East meets West - Byzantine and Ottoman heritage.",
        images=["/static/images/istanbul1.jpg", "/static/images/istanbul2.jpg", "/static/images/istanbul3.jpg"]
    ),
    Destination(
        id="15",
        name="New York City",
        country="USA",
        rating=4.6,
        shortDescription="The city that never sleeps - skyscrapers, culture, and endless possibilities.",
        images=["/static/images/nyc1.jpg", "/static/images/nyc2.jpg", "/static/images/nyc3.jpg"]
    ),
    Destination(
        id="16",
        name="Paris",
        country="France",
        rating=4.9,
        shortDescription="City of lights with iconic landmarks, art, and romantic ambiance.",
        images=["/static/images/paris1.jpg", "/static/images/paris2.jpg", "/static/images/paris3.jpg"]
    ),
    Destination(
        id="17",
        name="Bangkok",
        country="Thailand",
        rating=4.5,
        shortDescription="Bustling street life, ornate temples, and amazing street food.",
        images=["/static/images/bangkok1.jpg", "/static/images/bangkok2.jpg", "/static/images/bangkok3.jpg"]
    ),
    Destination(
        id="18",
        name="Rio de Janeiro",
        country="Brazil",
        rating=4.7,
        shortDescription="Carnival spirit, Christ the Redeemer, and famous beaches.",
        images=["/static/images/rio1.jpg", "/static/images/rio2.jpg", "/static/images/rio3.jpg"]
    ),
    Destination(
        id="19",
        name="Singapore",
        country="Singapore",
        rating=4.6,
        shortDescription="Modern city-state with futuristic gardens and diverse cultures.",
        images=["/static/images/singapore1.jpg", "/static/images/singapore2.jpg", "/static/images/singapore3.jpg"]
    ),
    Destination(
        id="20",
        name="Vienna",
        country="Austria",
        rating=4.7,
        shortDescription="Imperial palaces, classical music heritage, and coffeehouse culture.",
        images=["/static/images/vienna1.jpg", "/static/images/vienna2.jpg", "/static/images/vienna3.jpg"]
    )
]


def get_destinations(
    q: Optional[str] = None,
    country: Optional[str] = None,
    sort: Optional[str] = None,
    page: int = 1,
    limit: int = 8
) -> dict:
    filtered = destinations_data
    
    # Filter by search query
    if q:
        q_lower = q.lower()
        filtered = [d for d in filtered if q_lower in d.name.lower() or q_lower in d.country.lower()]
    
    # Filter by country
    if country:
        filtered = [d for d in filtered if d.country.lower() == country.lower()]
    
    # Sort
    if sort == "name":
        filtered = sorted(filtered, key=lambda x: x.name)
    elif sort == "rating":
        filtered = sorted(filtered, key=lambda x: x.rating, reverse=True)
    
    # Pagination
    total = len(filtered)
    total_pages = (total + limit - 1) // limit
    start = (page - 1) * limit
    end = start + limit
    items = filtered[start:end]
    
    return {
        "items": items,
        "page": page,
        "totalPages": total_pages,
        "total": total
    }


def get_destination_by_id(destination_id: str) -> Optional[Destination]:
    for dest in destinations_data:
        if dest.id == destination_id:
            return dest
    return None