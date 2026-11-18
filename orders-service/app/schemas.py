from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=8, max_length=72)

    @field_validator("password")
    @classmethod
    def validate_password_bytes(cls, v: str) -> str:
        if not (8 <= len(v.encode("utf-8")) <= 72):
            raise ValueError("Password must be between 8 and 72 bytes")
        return v


class UserOut(BaseModel):
    id: int
    username: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class OrderCreate(BaseModel):
    client_name: str = Field(..., min_length=2, max_length=120)
    gr_liv_area: float = Field(..., gt=0, le=10000)
    total_bsmt_sf: float = Field(..., ge=0, le=10000)
    garage_area: float = Field(..., ge=0, le=2000)
    year_built: int = Field(..., ge=1872, le=2010)
    overall_qual: int = Field(..., ge=1, le=10)
    ms_zoning: str = Field(..., min_length=1, max_length=10)


class OrderUpdate(BaseModel):
    client_name: Optional[str] = Field(default=None, min_length=2, max_length=120)
    gr_liv_area: Optional[float] = Field(default=None, gt=0, le=10000)
    total_bsmt_sf: Optional[float] = Field(default=None, ge=0, le=10000)
    garage_area: Optional[float] = Field(default=None, ge=0, le=2000)
    year_built: Optional[int] = Field(default=None, ge=1872, le=2010)
    overall_qual: Optional[int] = Field(default=None, ge=1, le=10)
    ms_zoning: Optional[str] = Field(default=None, min_length=1, max_length=10)


class OrderRead(BaseModel):
    id: int
    client_name: str
    gr_liv_area: float
    total_bsmt_sf: float
    garage_area: float
    year_built: int
    overall_qual: int
    ms_zoning: str
    predicted_price: Optional[float] = None
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}
