FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download model SBERT dan IndoBERT saat build
RUN python -c "from sentence_transformers import SentenceTransformer; \
    print('Downloading SBERT model...'); \
    SentenceTransformer('paraphrase-multilingual-mpnet-base-v2'); \
    print('SBERT model downloaded successfully')"
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    print('Downloading IndoBERT model...'); \
    AutoTokenizer.from_pretrained('indobenchmark/indobert-base-p1'); \
    AutoModel.from_pretrained('indobenchmark/indobert-base-p1'); \
    print('IndoBERT model downloaded successfully')"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
