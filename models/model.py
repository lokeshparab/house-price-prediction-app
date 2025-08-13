from pydantic import BaseModel

class HouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    living_room_sqft: float
    lot_sqft: float
    floors: float
    waterfront: int  # 0 or 1
    view: int  # 0-4 scale
    condition: int  # 1-5 scale
    sqft_above: float
    sqft_basement: float
    year_built: int
    year_renovated: int
