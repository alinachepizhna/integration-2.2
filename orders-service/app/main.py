import logging
import os
from typing import List

from fastapi import Depends, FastAPI, HTTPException, Query, Response, status
from fastapi.security import OAuth2PasswordRequestForm
import httpx
from sqlalchemy.orm import Session

from . import models, schemas
from .auth import create_access_token, get_current_user, hash_password, verify_password
from .database import Base, engine, get_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orders_api")

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Orders API")

FORECAST_URL = os.getenv("FORECAST_URL", "http://forecast-api:8001/predict")
FORECAST_TIMEOUT = float(os.getenv("FORECAST_TIMEOUT", "3.0"))

FEATURE_FIELDS = {
    "gr_liv_area",
    "total_bsmt_sf",
    "garage_area",
    "year_built",
    "overall_qual",
    "ms_zoning",
}


def fetch_predicted_price(payload: dict) -> float | None:
    try:
        with httpx.Client(timeout=FORECAST_TIMEOUT) as client:
            resp = client.post(FORECAST_URL, json=payload)
            resp.raise_for_status()
            return float(resp.json()["predicted_price"])
    except Exception as exc:
        logger.warning(f"Forecast API unavailable, graceful degradation to None: {exc}")
        return None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/register", response_model=schemas.UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.username == payload.username).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already registered")
    user = models.User(
        username=payload.username,
        hashed_password=hash_password(payload.password),
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/auth/login", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"access_token": create_access_token(subject=user.username), "token_type": "bearer"}


@app.get("/auth/me", response_model=schemas.UserOut)
def read_me(current_user: models.User = Depends(get_current_user)):
    return current_user


@app.post("/orders", response_model=schemas.OrderRead, status_code=status.HTTP_201_CREATED)
def create_order(
    payload: schemas.OrderCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    forecast_payload = {
        "gr_liv_area": payload.gr_liv_area,
        "total_bsmt_sf": payload.total_bsmt_sf,
        "garage_area": payload.garage_area,
        "year_built": payload.year_built,
        "overall_qual": payload.overall_qual,
        "ms_zoning": payload.ms_zoning,
    }
    predicted_price = fetch_predicted_price(forecast_payload)
    order = models.Order(**payload.model_dump(), predicted_price=predicted_price)
    db.add(order)
    db.commit()
    db.refresh(order)
    return order


@app.get("/orders", response_model=List[schemas.OrderRead])
def list_orders(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    return db.query(models.Order).order_by(models.Order.id).offset(offset).limit(limit).all()


@app.get("/orders/{oid}", response_model=schemas.OrderRead)
def get_order(
    oid: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    order = db.query(models.Order).filter(models.Order.id == oid).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order


@app.patch("/orders/{oid}", response_model=schemas.OrderRead)
def update_order(
    oid: int,
    payload: schemas.OrderUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    order = db.query(models.Order).filter(models.Order.id == oid).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    update_data = payload.model_dump(exclude_unset=True)
    features_changed = bool(FEATURE_FIELDS.intersection(update_data.keys()))

    for field, value in update_data.items():
        setattr(order, field, value)

    if features_changed:
        forecast_payload = {
            "gr_liv_area": order.gr_liv_area,
            "total_bsmt_sf": order.total_bsmt_sf,
            "garage_area": order.garage_area,
            "year_built": order.year_built,
            "overall_qual": order.overall_qual,
            "ms_zoning": order.ms_zoning,
        }
        order.predicted_price = fetch_predicted_price(forecast_payload)

    db.commit()
    db.refresh(order)
    return order


@app.delete("/orders/{oid}", status_code=status.HTTP_204_NO_CONTENT)
def delete_order(
    oid: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    order = db.query(models.Order).filter(models.Order.id == oid).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    db.delete(order)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
