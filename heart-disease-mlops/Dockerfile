FROM python:3.14.6-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements-docker.txt ./
RUN python -m pip install --no-cache-dir -r requirements-docker.txt

COPY src/ ./src/
COPY artifacts/final_heart_disease_pipeline.joblib ./artifacts/final_heart_disease_pipeline.joblib
COPY artifacts/final_heart_disease_metadata.json ./artifacts/final_heart_disease_metadata.json

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "heart_disease.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
