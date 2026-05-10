# ---- Multi-stage build for slim deployment ----
FROM python:3.12-slim AS base

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ src/
COPY data/processed/ data/processed/

# ---- FastAPI service ----
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- Streamlit dashboard ----
FROM base AS dashboard
EXPOSE 8501
CMD ["streamlit", "run", "src/api/dashboard.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]
