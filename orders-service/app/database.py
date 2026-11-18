import os
from pathlib import Path
from sqlalchemy import URL, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


def read_secret(name: str) -> str:
    secret_path = Path(f"/run/secrets/{name}")
    if not secret_path.is_file():
        raise RuntimeError(f"FAIL-FAST: Secret '{name}' not found at {secret_path}")
    return secret_path.read_text(encoding="utf-8").strip()


DB_PASSWORD = read_secret("db_password")

database_url = URL.create(
    drivername="postgresql+psycopg",
    username=os.getenv("POSTGRES_USER", "orders"),
    password=DB_PASSWORD,
    host=os.getenv("POSTGRES_HOST", "postgres"),
    port=int(os.getenv("POSTGRES_PORT", "5432")),
    database=os.getenv("POSTGRES_DB", "orders"),
)

engine = create_engine(database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
