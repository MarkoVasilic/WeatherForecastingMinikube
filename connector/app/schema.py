from pydantic import BaseModel
from typing import Optional

class DataBase(BaseModel):
    id: Optional[int]
    date_time: str
    p: float
    T: float
    rh: float
    Vpact: float
    wv: float
    rho: float

    class Config:
        from_attributes = True


class DataCreate(BaseModel):
    date_time: str
    p: float
    T: float
    rh: float
    Vpact: float
    wv: float
    rho: float
