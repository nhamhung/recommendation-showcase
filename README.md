# Recommendation Systems Showcase

A single project presenting **two genuinely different ways to solve
"recommend a song"**, built end to end and compared head to head:

- **Content-based** (`src/recommendation_showcase/content_based/`) — no
  user data at all. Given a track's own Spotify audio features
  (danceability, energy, acousticness, ...), recommend similar-sounding
  tracks. Built on Kaggle's [Spotify Tracks
  Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset).
- **Collaborative filtering** (`src/recommendation_showcase/collaborative/`)
  — no item content at all. Given a real member's listening history and
  a specific song, predict whether they'll replay it within a month.
  Built on Kaggle's real [WSDM — KKBox's Music Recommendation
  Challenge](https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge)
  (2018), scored by AUC on a real leaderboard.

This project began as two separate portfolio entries
(`music_recommendation` and `kkbox_music_recommendation`) and was merged
here — two clean sub-packages sharing one project, one unified report,
one app — once each half was independently verified working.

## What's here

| Deliverable | Where |
|---|---|
| Two well-documented notebooks, one per approach | `notebooks/01_content_based.ipynb`, `notebooks/02_collaborative_filtering.ipynb` |
| A multi-page Streamlit app — both recommenders, grouped by approach in the sidebar | `app/streamlit_app.py` + `app/pages_src/` (+ `app/Dockerfile`) |
| One research-style writeup: a section per approach plus a head-to-head comparison | `report/report.qmd` |
| Scripts proving each approach works end-to-end | `scripts/train_content_based.py` + `scripts/recommend.py`; `scripts/train_collaborative.py` + `scripts/make_submission.py` |

Each sub-package (`content_based/`, `collaborative/`) is a complete,
independent `config`/`data`/`features`/`model`/`interpretability` set —
the notebook, the app, and the scripts for that approach all load the
same trained model artifact (`models/content_based_model.joblib` /
`models/collaborative_model.joblib`), so nothing can quietly drift
between them.

## Why merge two working, independent projects into one?

Comparing the two techniques honestly needed them to actually run in
the same place — the app's whole point is trying both recommenders in
one session, and the report's head-to-head section reuses numbers
directly from both halves' own methodology sections. Everything
approach-specific (data loading, feature engineering, evaluation) stays
fully separate in its own sub-package; only the presentation layer
(app, unified report) is genuinely shared.

## Project layout

```
data/
  content_based/raw/         # Spotify dataset CSV (gitignored — see below)
  collaborative/raw/         # KKBox competition CSVs (gitignored)
notebooks/
  01_content_based.ipynb
  02_collaborative_filtering.ipynb
src/recommendation_showcase/
  content_based/             # config, data, features, model, interpretability, genre_families
  collaborative/             # config, data, features, model, interpretability
models/
  content_based_model.joblib
  collaborative_model.joblib   # ~48MB
app/                          # Streamlit app (multi-page, grouped by approach) + Dockerfile
scripts/
  train_content_based.py, recommend.py
  train_collaborative.py, make_submission.py
report/                       # one Quarto research writeup, both approaches
tests/                        # pytest tests for both sub-packages (synthetic data — no download needed)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data

Neither dataset is committed (redistribution isn't permitted for
either). Download both:

```bash
# Spotify (a Kaggle dataset, not a competition)
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset -p data/content_based/raw
unzip -o data/content_based/raw/spotify-tracks-dataset.zip -d data/content_based/raw

# KKBox (a real competition — accept its rules on kaggle.com first, in a browser)
kaggle competitions download -c kkbox-music-recommendation-challenge -p data/collaborative/raw
unzip -o data/collaborative/raw/kkbox-music-recommendation-challenge.zip -d data/collaborative/raw
cd data/collaborative/raw && for f in *.csv.7z; do 7z x -y "$f"; done && cd ../../..
```

## Run the notebooks

```bash
jupyter notebook notebooks/01_content_based.ipynb
jupyter notebook notebooks/02_collaborative_filtering.ipynb
```

## Train from the command line

```bash
python scripts/train_content_based.py
python scripts/train_collaborative.py
```

## Try each approach directly

```bash
python scripts/recommend.py "a track name"                          # content-based
python scripts/make_submission.py                                    # collaborative — real Kaggle submission
```

## Run the app

Eight pages total, grouped into two sidebar sections — **Content-Based
(Spotify)**: Recommend, Overview, Feature Engineering, Model Insights;
**Collaborative Filtering (KKBox)**: Predict, Overview, Feature
Engineering, Model Insights.

```bash
streamlit run app/streamlit_app.py
```

Or in Docker (from the project root, after training both models and
downloading both datasets):

```bash
docker build -t recommendation-showcase-app -f app/Dockerfile .
docker run -p 8501:8501 recommendation-showcase-app
```

## Render the research writeup

```bash
quarto render report/report.qmd
```

## Run the tests

```bash
pytest tests/
```

Tests for both sub-packages run against small synthetic data — no
download needed, and they already pass without any real data.

## Deploy

The Streamlit app fetches its source data through the Kaggle API at runtime. Configure either KAGGLE_API_TOKEN or a [kaggle] secrets section containing username and key. The KKBox competition also requires accepted competition terms.

- Repository: <https://github.com/nhamhhung/recommendation-showcase>
- Report: <https://nhamhung.github.io/recommendation-showcase/>
- Streamlit: <https://recommendation-showcase.streamlit.app>
- Fork setup: [docs/SETUP_AND_DEPLOYMENT.md](docs/SETUP_AND_DEPLOYMENT.md)
