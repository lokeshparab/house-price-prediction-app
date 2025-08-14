from pydantic import BaseModel

class HouseFeatures(BaseModel):
    bedRoom: int
    bathroom: int
    area: float
    price_per_sqft: float
    floorNum: int
