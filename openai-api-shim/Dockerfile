FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc libffi-dev \
    && pip install --no-cache-dir python-multipart fastapi uvicorn httpx \
    && apt-get remove -y gcc libffi-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY openai-api-shim.py /app/app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
