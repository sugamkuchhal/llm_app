FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use gunicorn for Cloud Run; bind to injected $PORT.
CMD ["sh", "-c", "exec gunicorn -b 0.0.0.0:${PORT:-8080} app:app"]
