FROM python:3.11-slim

# Små, nyttige defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Installér wget til healthcheck (og certs), og ryd op efter apt
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates wget \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Afhængigheder først for bedre layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App-kode ind
COPY . .

EXPOSE 8000

CMD ["uvicorn","bot.live_connector.runner:app","--host","0.0.0.0","--port","8000","--workers","1"]
