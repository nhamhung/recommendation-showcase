# Verifying `recommendation_showcase`

Run from the project root (`cd projects/recommendation_showcase` first). Each
step is independent — skip any you've already confirmed.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Tests (no data download needed — synthetic data)

```bash
export PYTHONPATH="src:."
pytest tests/ -v
```
Expect: **30 passed**.

## 3. Data (skip if already downloaded — see README for full instructions)

```bash
# Spotify
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset -p data/content_based/raw
unzip -o data/content_based/raw/spotify-tracks-dataset.zip -d data/content_based/raw

# KKBox — accept the competition rules on kaggle.com first, in a browser
kaggle competitions download -c kkbox-music-recommendation-challenge -p data/collaborative/raw
unzip -o data/collaborative/raw/kkbox-music-recommendation-challenge.zip -d data/collaborative/raw
cd data/collaborative/raw && for f in *.csv.7z; do 7z x -y "$f"; done && cd ../../..
```

## 4. Train both models (skip if `models/*.joblib` already present)

```bash
python scripts/train_content_based.py       # ~1 minute
python scripts/train_collaborative.py       # ~90 seconds, full 7.4M rows
```

## 5. Exercise each approach directly

```bash
python scripts/recommend.py "a track name" -k 5      # content-based
python scripts/make_submission.py                     # collaborative — writes submission.csv
```

## 6. Notebooks (executes end-to-end against real data)

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_content_based.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_collaborative_filtering.ipynb
```
Or open interactively: `jupyter notebook notebooks/`

## 7. Report

```bash
quarto render report/report.qmd
open report/report.html    # or report/report.pdf
```

## 8. App — manual check

```bash
streamlit run app/streamlit_app.py
```
Open http://localhost:8501. Check both sidebar sections load, and that
clicking "Recommend" (content-based) and "Predict replay probability"
(collaborative) both produce results without errors.

## 9. App — automated check (no browser needed)

```bash
python -c "
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('app/streamlit_app.py', default_timeout=120)
at.run(timeout=120)
assert not at.exception, at.exception
print('App loads cleanly:', at.title[0].value)
"
```

## 10. Docker (optional)

```bash
docker build -t recommendation-showcase-app -f app/Dockerfile .
docker run -p 8501:8501 recommendation-showcase-app
```

---

**Full clean-room check** (everything above, in sequence, from a fresh
clone/extract with data already downloaded):

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
export PYTHONPATH="src:."
pytest tests/ -v
python scripts/train_content_based.py
python scripts/train_collaborative.py
python scripts/recommend.py "Bohemian Rhapsody" -k 5
python scripts/make_submission.py
quarto render report/report.qmd
streamlit run app/streamlit_app.py   # manual: Ctrl+C when done checking
```
