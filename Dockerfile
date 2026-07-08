FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Menggunakan index:app karena entry point ada di index.py
CMD ["uvicorn", "index:app", "--host", "0.0.0.0", "--port", "8000"]
