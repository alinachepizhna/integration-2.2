from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, func
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True, server_default="true")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    client_name = Column(String(120), nullable=False)
    gr_liv_area = Column(Float, nullable=False)
    total_bsmt_sf = Column(Float, nullable=False)
    garage_area = Column(Float, nullable=False)
    year_built = Column(Integer, nullable=False)
    overall_qual = Column(Integer, nullable=False)
    ms_zoning = Column(String(10), nullable=False)
    predicted_price = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
