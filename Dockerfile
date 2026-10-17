FROM python:3.11-slim

# xgboost needs OpenMP at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY sql/ sql/
COPY config/ config/
COPY tests/ tests/
COPY pytest.ini .

ENTRYPOINT ["python", "-m", "leadscore"]
CMD ["all"]
