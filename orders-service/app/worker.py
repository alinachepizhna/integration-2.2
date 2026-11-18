from datetime import datetime, timezone
import logging
import os
import time
import httpx
import redis
from . import models
from .database import Base, SessionLocal, engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orders_worker")

REDIS_URL = os.getenv("REDIS_URL", "redis://broker:6379/0")
QUEUE_NAME = "pricing_queue"
FORECAST_URL = os.getenv("FORECAST_URL", "http://forecast-api:8001/predict")
FORECAST_TIMEOUT = float(os.getenv("FORECAST_TIMEOUT", "3.0"))


def process_order(order_id: int) -> None:
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
        for _ in range(2):
            try:
                with httpx.Client(timeout=FORECAST_TIMEOUT) as client:
                    resp = client.post(FORECAST_URL, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                db.refresh(order)
                order.predicted_price = float(data["predicted_price"])
                order.model_version = str(data["model_version"])
                order.status = "priced"
                db.commit()
                logger.info(f"order_id={order_id} status=priced price={order.predicted_price} model_version={order.model_version}")
                return
            except Exception:
                time.sleep(0.5)
        order.predicted_price = None
        order.status = "failed"
        db.commit()
    finally:
        db.close()


def main() -> None:
    Base.metadata.create_all(bind=engine)
    r = redis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Orders worker started...")
    while True:
        item = r.blpop(QUEUE_NAME, timeout=5)
        if item:
            _, raw_id = item
            process_order(int(raw_id))


if __name__ == "__main__":
    main()
