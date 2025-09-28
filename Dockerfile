FROM python:3.11-slim

# Prevent .pyc files, get instant logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV AWS_DEFAULT_REGION=us-east-2

ENV MPLBACKEND=Agg

# XGBoost needs libgomp on slim variants
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first to maximize Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bring in the rest of the repo (includes data/ and analysis.py)
COPY . .

# Default entrypoint: run the analysis
CMD ["python", "analysis.py"]