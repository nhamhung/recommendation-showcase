# Build from the project root:
#   docker build -t recommendation-showcase-app -f app/Dockerfile .
# Run:
#   docker run -p 8501:8501 recommendation-showcase-app

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY models/ models/
# Both sides read their own catalog directly (track metadata for the
# content-based side; songs/members/song_extra_info for the
# collaborative side) — download them first, see the README's "Get the
# data" section. Neither dataset may be freely redistributed publicly
# (Kaggle competition/dataset terms) — the same caution as every other
# project in this portfolio applies here, doubled.
COPY data/content_based/raw/dataset.csv data/content_based/raw/dataset.csv
COPY data/collaborative/raw/train.csv data/collaborative/raw/train.csv
COPY data/collaborative/raw/songs.csv data/collaborative/raw/songs.csv
COPY data/collaborative/raw/members.csv data/collaborative/raw/members.csv
COPY data/collaborative/raw/song_extra_info.csv data/collaborative/raw/song_extra_info.csv

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
