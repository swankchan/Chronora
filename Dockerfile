FROM python:3.11-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# copy only requirements first for better caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy app
COPY . /app

ENV PYTHONUNBUFFERED=1
ENV REDIS_URL=redis://redis:6379/0

EXPOSE 8001

CMD ["uvicorn", "sdxl_direct:app", "--host", "0.0.0.0", "--port", "8001"]
