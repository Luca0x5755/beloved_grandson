FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY . .
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    rubberband-cli \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt
CMD ["python", "-u", "worker/main.py"]
