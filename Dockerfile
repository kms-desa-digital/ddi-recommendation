FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Menggunakan uvicorn dengan 4 workers agar bisa memanfaatkan multi-core vCPU
CMD ["uvicorn", "index:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
