# BASE
FROM python:3.10-slim

WORKDIR /app

# Upgrade pip (important)
RUN pip install --no-cache-dir --upgrade pip

# Copy only requirements first (for caching)
COPY requirements.txt .

# Install deps (split for better layer caching)
RUN pip install --no-cache-dir numpy \
    && pip install --no-cache-dir fastapi uvicorn pydantic \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy code
COPY . .

EXPOSE 8000

CMD ["python", "app.py"] 