FROM python:3.12-slim

# System deps for PyMuPDF, Pillow, open_clip
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (prevents pip pulling CUDA build)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python deps
COPY pipeline/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy pipeline source
COPY pipeline/ /app/pipeline/

# Copy omnisuite packages (deps already covered by main requirements.txt)
COPY omnisuitef1/ /app/omnisuitef1/

# Copy AgentML module (imported by chat_server)
COPY onmichine/ /app/onmichine/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/health')" || exit 1

CMD ["python", "/app/pipeline/chat_server.py"]
