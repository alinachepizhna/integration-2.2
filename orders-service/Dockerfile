FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic sqlalchemy "psycopg[binary]" httpx pyjwt "bcrypt==4.0.1" python-multipart prometheus-client
COPY orders-service/app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
