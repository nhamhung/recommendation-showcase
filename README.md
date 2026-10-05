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

## Prerequisites

Install once, before Setup below:

| Dependency | Why | Install |
|---|---|---|
| **Python 3.12** | This project's `.venv` is built against 3.12 — a different version may resolve incompatible package versions from `requirements.txt`. | [python.org/downloads](https://www.python.org/downloads/) or a version manager (e.g. `pyenv install 3.12`) |
| **Quarto** | Renders `report/report.qmd` — a standalone binary, not a Python package, so `pip install` never gets it. | [quarto.org/docs/get-started](https://quarto.org/docs/get-started/) |
| **Kaggle account** | Needed only for the complete datasets; the default Streamlit app uses bundled samples, and `pytest` uses synthetic data. The KKBox side is a competition — accept its rules on kaggle.com in a browser first, or its API download will 403. | Kaggle account → **Account → Create New API Token** → save as `~/.kaggle/kaggle.json`. See the [Kaggle API docs](https://www.kaggle.com/docs/api). |
| **`p7zip`** (`7z`) | KKBox's competition files are `.7z`-compressed, not plain-zipped — needed to extract them. | `brew install p7zip` (macOS), `apt install p7zip-full` (Debian/Ubuntu), or see [7-zip.org](https://www.7-zip.org/) |
| **Docker** (optional) | Only if you want to run the app in its pre-baked container instead of `streamlit run`. | [docker.com/get-started](https://www.docker.com/get-started/) |

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

## Making changes

Each sub-package is its own single source of truth, with no code shared
between them beyond the app/report presentation layer:

- `src/recommendation_showcase/content_based/` — `config.py`, `data.py`,
  `features.py`, `model.py`, `interpretability.py`, `genre_families.py`.
- `src/recommendation_showcase/collaborative/` — the same module set,
  independent implementation.

The edit loop (same shape for either sub-package):

```bash
# 1. Edit src/recommendation_showcase/<sub_package>/*.py

# 2. Check it against that sub-package's tests (fast, synthetic data, no download needed)
pytest tests/

# 3. Retrain, so models/<sub_package>_model.joblib reflects your change
python scripts/train_content_based.py     # or:
python scripts/train_collaborative.py
```

The relevant `models/*.joblib` is what that sub-package's notebook, app
pages, and report section all load — retraining is the one step that
makes a model-code change visible everywhere else.

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
python3.12 -m venv .venv          # use the 3.12 interpreter specifically
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data

Neither dataset is committed (redistribution isn't permitted for
either). Download both:

```bash
# Spotify (a Kaggle dataset, not a competition)
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset -p data/content_based/raw
unzip -o data/content_based/raw/-spotify-tracks-dataset.zip -d data/content_based/raw

# KKBox (a real competition — accept its rules on kaggle.com first, in a browser)
kaggle competitions download -c kkbox-music-recommendation-challenge -p data/collaborative/raw
unzip -o data/collaborative/raw/kkbox-music-recommendation-challenge.zip -d data/collaborative/raw
cd data/collaborative/raw && for f in *.csv.7z; do 7z x -y "$f"; done && cd ../../..
```

## Run the notebooks

Both notebooks (and the Quarto report below) run on a named Jupyter
kernel, `recommendation-showcase`. Register it once, from the active
venv:

```bash
python -m ipykernel install --user --name recommendation-showcase
```

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

Requires the `recommendation-showcase` kernel registered above (`report.qmd`
declares it via `jupyter: recommendation-showcase`):

```bash
quarto render report/report.qmd
```

This regenerates both `report/report.html` and `report/report.pdf` (PDF
needs a LaTeX distribution — if you don't have one, run
`quarto install tinytex` once). Render just one format when you don't need
both:

```bash
quarto render report/report.qmd --to html
quarto render report/report.qmd --to pdf
```

Live-preview while editing (auto-rerenders on save):

```bash
quarto preview report/report.qmd
```

A `.qmd` file is Markdown prose plus fenced Python code chunks
(` ```{python} `/` ``` `), executed top to bottom by the kernel above, same
as a notebook cell. Common per-chunk options (a `#|` comment, first line of
the chunk): `#| echo: false` (hide this chunk's source code),
`#| output: false` (suppress its output, e.g. a setup/import cell),
`#| label: fig-foo` + `#| fig-cap: "..."` (name and caption a figure for
cross-referencing). The [Quarto VS Code
extension](https://marketplace.visualstudio.com/items?itemName=quarto.quarto)
adds syntax highlighting and a one-click Render button if you're doing more
than a one-line edit.

Troubleshooting:

| Symptom | Likely cause |
|---|---|
| `Jupyter engine failed ... kernel not found` | The `ipykernel install --name recommendation-showcase` step above hasn't been run yet. |
| `ModuleNotFoundError` inside a code chunk | `quarto render` runs with its working directory set to `report/`, not the project root — check the chunk's `sys.path.insert(0, "../src")` points at the right relative path. |
| Output looks stale after editing | Force a clean re-run: `quarto render report/report.qmd --execute-daemon-restart`. |
| PDF render fails, HTML succeeds | Missing LaTeX — run `quarto install tinytex` once, then retry. |

## Run the tests

```bash
pytest tests/
```

Tests for both sub-packages run against small synthetic data — no
download needed, and they already pass without any real data.

## Deploy

The Streamlit app starts immediately from compact Spotify and KKBox samples sourced from Kaggle. Set `USE_FULL_KAGGLE_DATA=true` to fetch and use the complete datasets through the Kaggle API; configure either `KAGGLE_API_TOKEN` or a `[kaggle]` secrets section containing username and key. The KKBox competition also requires accepted competition terms.

- Repository: <https://github.com/nhamhhung/recommendation-showcase>
- Report: <https://nhamhung.github.io/recommendation-showcase/>
- Streamlit: <https://recommendation-showcase.streamlit.app>
- Fork setup: [docs/SETUP_AND_DEPLOYMENT.md](docs/SETUP_AND_DEPLOYMENT.md)
