import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import os
import time
from typing import Annotated, List

from fastapi import BackgroundTasks, Body, Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
import httpx
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sqlalchemy.orm import Session

from . import models, schemas
from .auth import create_access_token, get_current_user, hash_password, verify_password
from .database import Base, SessionLocal, engine, get_db

Base.metadata.create_all(bind=engine)

FORECAST_URL = os.getenv("FORECAST_URL", "http://forecast-api:8001/predict")
FORECAST_TIMEOUT = float(os.getenv("FORECAST_TIMEOUT", "3.0"))
BATCH_MAX_ITEMS = int(os.getenv("BATCH_MAX_ITEMS", "50"))
FEATURE_FIELDS = {"gr_liv_area", "total_bsmt_sf", "garage_area", "year_built", "overall_qual", "ms_zoning"}

ORDERS_TOTAL = Counter("orders_total", "Total orders by terminal pricing status", ["status"])
PRICING_SECONDS = Histogram(
    "pricing_seconds",
    "Wall time of one pricing operation in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
ORDERS_TOTAL.labels(status="priced").inc(0)
ORDERS_TOTAL.labels(status="failed").inc(0)


def emit(event: str, level: str = "info", **fields) -> None:
    record = {"ts": datetime.now(timezone.utc).isoformat(), "level": level, "event": event, **fields}
    print(json.dumps(record), flush=True)


def run_pricing_task(order_id: int) -> None:
    emit("pricing_started", order_id=order_id)
    t0 = time.perf_counter()
    db = SessionLocal()
    try:
        order = db.query(models.Order).filter(models.Order.id == order_id).first()
        if not order:
            return
        payload = {
            "gr_liv_area": order.gr_liv_area,
            "total_bsmt_sf": order.total_bsmt_sf,
            "garage_area": order.garage_area,
            "year_built": order.year_built,
            "overall_qual": order.overall_qual,
            "ms_zoning": order.ms_zoning,
        }
        last_err = ""
        for attempt in range(2):
            try:
                with httpx.Client(timeout=FORECAST_TIMEOUT) as client:
                    resp = client.post(FORECAST_URL, json=payload, headers={"X-Order-Id": str(order_id)})
                    resp.raise_for_status()
                    data = resp.json()
                elapsed = time.perf_counter() - t0
                order.predicted_price = float(data["predicted_price"])
                order.model_version = str(data["model_version"])
                order.status = "priced"
                db.commit()
                ORDERS_TOTAL.labels(status="priced").inc()
                PRICING_SECONDS.observe(elapsed)
                emit("pricing_succeeded", order_id=order_id, duration_ms=round(elapsed * 1000, 1))
                return
            except Exception as exc:
                last_err = str(exc)
                if attempt == 0:
                    time.sleep(1.0)

        elapsed = time.perf_counter() - t0
        order.predicted_price = None
        order.status = "failed"
        db.commit()
        ORDERS_TOTAL.labels(status="failed").inc()
        PRICING_SECONDS.observe(elapsed)
        emit("pricing_failed", level="error", order_id=order_id, error=last_err)
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(timeout=FORECAST_TIMEOUT)
    yield
    await app.state.http_client.aclose()


app = FastAPI(title="Orders API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/auth/register", response_model=schemas.UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: schemas.UserCreate, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.username == payload.username).first():
        raise HTTPException(status_code=409, detail="Username already registered")
    user = models.User(username=payload.username, hashed_password=hash_password(payload.password), is_active=True)
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


@app.post("/orders", response_model=schemas.OrderRead, status_code=status.HTTP_202_ACCEPTED)
def create_order(
    payload: schemas.OrderCreate,
    background_tasks: BackgroundTasks,
    response: Response,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    order = models.Order(**payload.model_dump(), predicted_price=None, status="pending", model_version=None)
    db.add(order)
    db.commit()
    db.refresh(order)
    response.headers["Location"] = f"/orders/{order.id}"
    background_tasks.add_task(run_pricing_task, order.id)
    return order


@app.post("/orders/batch", response_model=schemas.BatchOrderResponse)
async def create_orders_batch(
    items: Annotated[List[schemas.OrderCreate], Body(min_length=1, max_length=BATCH_MAX_ITEMS)],
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    created_orders: List[models.Order] = []
    for item in items:
        order = models.Order(**item.model_dump(), predicted_price=None, status="pending", model_version=None)
        db.add(order)
        created_orders.append(order)
    db.commit()
    for order in created_orders:
        db.refresh(order)

    http_client: httpx.AsyncClient = request.app.state.http_client

    async def fetch_one(o: models.Order):
        payload = {
            "gr_liv_area": o.gr_liv_area,
            "total_bsmt_sf": o.total_bsmt_sf,
            "garage_area": o.garage_area,
            "year_built": o.year_built,
            "overall_qual": o.overall_qual,
            "ms_zoning": o.ms_zoning,
        }
        r = await http_client.post(FORECAST_URL, json=payload, headers={"X-Order-Id": str(o.id)}, timeout=FORECAST_TIMEOUT)
        r.raise_for_status()
        return r.json()

    raw_results = await asyncio.gather(*(fetch_one(o) for o in created_orders), return_exceptions=True)

    results: List[schemas.BatchOrderItemResult] = []
    for idx, (order, res) in enumerate(zip(created_orders, raw_results)):
        if isinstance(res, Exception):
            order.status = "failed"
            order.predicted_price = None
            ORDERS_TOTAL.labels(status="failed").inc()
            results.append(schemas.BatchOrderItemResult(index=idx, order_id=order.id, status="failed", error="ForecastUnavailable"))
        else:
            order.status = "priced"
            order.predicted_price = float(res["predicted_price"])
            order.model_version = str(res["model_version"])
            ORDERS_TOTAL.labels(status="priced").inc()
            results.append(
                schemas.BatchOrderItemResult(
                    index=idx,
                    order_id=order.id,
                    status="priced",
                    predicted_price=order.predicted_price,
                    model_version=order.model_version,
                )
            )
    db.commit()
    return schemas.BatchOrderResponse(total=len(results), results=results)


@app.get("/orders", response_model=List[schemas.OrderRead])
def list_orders(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    return db.query(models.Order).order_by(models.Order.id).offset(offset).limit(limit).all()


@app.get("/orders/{oid}", response_model=schemas.OrderRead)
def get_order(oid: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    order = db.query(models.Order).filter(models.Order.id == oid).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order


@app.patch("/orders/{oid}", response_model=schemas.OrderRead)
def update_order(
    oid: int,
    payload: schemas.OrderUpdate,
    background_tasks: BackgroundTasks,
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
        order.status = "pending"
        order.predicted_price = None
        order.model_version = None
        db.commit()
        db.refresh(order)
        background_tasks.add_task(run_pricing_task, order.id)
    else:
        db.commit()
        db.refresh(order)

    return order


@app.delete("/orders/{oid}", status_code=status.HTTP_204_NO_CONTENT)
def delete_order(oid: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    order = db.query(models.Order).filter(models.Order.id == oid).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    db.delete(order)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
