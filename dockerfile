FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git curl \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100
WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD python scripts/download.py \
 && python scripts/bert_embeddings.py \
 && streamlit run app.py --server.port 8501 --server.address 0.0.0.0