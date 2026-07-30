# Beeline API. Serves POST /api/path and the concept graph; the cut clips
# themselves are served from S3/CloudFront in a deployed environment, which is
# why ffmpeg and yt-dlp are optional here (see BEELINE_CANNED below).
FROM python:3.12-slim

# ffmpeg is only needed when gap filling runs live. A deployed demo runs with
# BEELINE_CANNED=1 and pre-filled clips, because yt-dlp from a datacentre IP is
# slow and routinely throttled -- but keeping ffmpeg available means the image
# can still do it if you turn canned mode off.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Only what the API needs at runtime. The frontend is served as static files
# from S3, and data/media (the 440MB of source video) never belongs in an image.
COPY beeline/shared/ ./beeline/shared/
COPY beeline/graph/ ./beeline/graph/
COPY beeline/integration/ ./beeline/integration/
COPY beeline/ingestion/graph_payload.json ./beeline/ingestion/graph_payload.json
COPY beeline/data/cache/ ./beeline/data/cache/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BEELINE_STORE=neo4j \
    BEELINE_CANNED=1 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -fsS http://localhost:${PORT}/api/health || exit 1

# App Runner sets PORT; honour it rather than hardcoding.
CMD ["sh", "-c", "uvicorn app:app --app-dir beeline/integration --host 0.0.0.0 --port ${PORT}"]
