FROM python:3.12-slim

WORKDIR /app

# System deps needed by tokenizers (sentencepiece) and numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Ports: 8501 (Streamlit chat UI), 8000 (FastAPI REST)
EXPOSE 8501 8000

# Ollama runs outside the container; point at it with OLLAMA_HOST.
# Default CMD launches the Streamlit UI.  Override with:
#   docker run ... uvicorn api:app --host 0.0.0.0 --port 8000
ENV OLLAMA_HOST=http://host.docker.internal:11434

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
