from sqlalchemy import Column, Double, Integer, DateTime
from .database import Base

class Data(Base):
    __tablename__ = "weather_data"

    id = Column(Integer, primary_key=True, index=True)
    date_time = Column(DateTime, nullable=False)
    p = Column(Double)
    T = Column(Double)
    rh = Column(Double)
    Vpact = Column(Double)
    wv = Column(Double)
    rho = Column(Double)