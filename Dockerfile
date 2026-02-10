FROM python:3.11-slim

# System deps for OpenCV headless
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . /app
WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
