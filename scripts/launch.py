#!/usr/bin/env python
"""One-click launcher: open the notebooks (JupyterLab) or the Streamlit app in your browser.

You normally don't run this yourself. Double-click `start_windows.bat` (Windows)
or `start_mac.command` (macOS): they install uv if needed, let uv set up the
exact Python and packages from `uv.lock`, then run this script.

Usage (from the project folder):
    uv run python scripts/launch.py            # menu
    uv run python scripts/launch.py notebooks  # straight to JupyterLab
    uv run python scripts/launch.py app        # straight to the Streamlit app
    uv run python scripts/launch.py check      # only check the setup, then exit
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME = "Recommendation Showcase"
APP = PROJECT_ROOT / "app" / "streamlit_app.py"
NOTEBOOKS = PROJECT_ROOT / "notebooks"


# --- Project-specific: data the notebooks need ------------------------------
# The app runs on the small samples bundled in data/demo; only the notebooks
# need the full Kaggle data.

SPOTIFY_DIR = PROJECT_ROOT / "data" / "content_based" / "raw"
KKBOX_DIR = PROJECT_ROOT / "data" / "collaborative" / "raw"
KKBOX_FILES = ["train.csv", "test.csv", "songs.csv", "members.csv", "song_extra_info.csv",
               "sample_submission.csv"]
NOTEBOOK_DATA = {
    "01_content_based.ipynb": [SPOTIFY_DIR / "dataset.csv"],
    "02_collaborative_filtering.ipynb": [KKBOX_DIR / name for name in KKBOX_FILES],
}
# Spotify Tracks is a public Kaggle dataset: Kaggle serves it without an account.
SPOTIFY_DATASET = "maharshipandya/-spotify-tracks-dataset"
SPOTIFY_URL = f"https://www.kaggle.com/api/v1/datasets/download/{SPOTIFY_DATASET}"
# KKBox is a competition: it needs a Kaggle account that has accepted the rules.
KKBOX_COMPETITION = "kkbox-music-recommendation-challenge"
KKBOX_RULES_URL = f"https://www.kaggle.com/competitions/{KKBOX_COMPETITION}/rules"
KAGGLE_DIR = Path(os.environ.get("KAGGLE_CONFIG_DIR") or Path.home() / ".kaggle")


def _missing(notebook: str) -> list[Path]:
    return [p for p in NOTEBOOK_DATA[notebook] if not p.exists()]


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:  # no keyboard attached (e.g. started from another program)
        print()
        return ""


def _kaggle_token_found() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        return True
    # Notepad on Windows likes to add ".txt"; the Kaggle tool only reads kaggle.json.
    if (KAGGLE_DIR / "kaggle.json.txt").is_file() and not (KAGGLE_DIR / "kaggle.json").exists():
        (KAGGLE_DIR / "kaggle.json.txt").rename(KAGGLE_DIR / "kaggle.json")
    return any((KAGGLE_DIR / name).is_file() for name in ("access_token", "access_token.txt", "kaggle.json"))


def _save_pasted_token(text: str) -> None:
    """Save a token pasted into this window where the Kaggle tool looks for it."""
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    if text.startswith("{"):  # the contents of a legacy kaggle.json file
        json.loads(text)
        target = KAGGLE_DIR / "kaggle.json"
    else:
        target = KAGGLE_DIR / "access_token"
    target.write_text(text + "\n", encoding="utf-8")
    target.chmod(0o600)
    print(f"Saved your Kaggle token to {target}")


def _download_spotify() -> bool:
    """Notebook 1's data: Spotify Tracks (~9 MB zip, ~20 MB CSV). No account needed."""
    print("\nNotebook 1 needs the Spotify Tracks dataset from Kaggle (about 9 MB, no account needed).")
    print("Downloading it now...")
    SPOTIFY_DIR.mkdir(parents=True, exist_ok=True)
    target = SPOTIFY_DIR / "dataset.csv"
    archive = SPOTIFY_DIR / "spotify-tracks-dataset.zip.part"
    try:
        import certifi
        import ssl

        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(SPOTIFY_URL, timeout=60, context=context) as response, open(archive, "wb") as out:
            shutil.copyfileobj(response, out)
        with zipfile.ZipFile(archive) as zf, zf.open("dataset.csv") as src, open(f"{target}.part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        Path(f"{target}.part").replace(target)
        print(f"Done: {target.relative_to(PROJECT_ROOT)}")
        return True
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        print(f"The download didn't work ({error}).")
        print("You can download it by hand instead: open")
        print(f"  https://www.kaggle.com/datasets/{SPOTIFY_DATASET}")
        print("click Download, unzip it, and put dataset.csv in this folder:")
        print(f"  {SPOTIFY_DIR}")
        return False
    finally:
        archive.unlink(missing_ok=True)
        Path(f"{target}.part").unlink(missing_ok=True)


def _extract_kkbox(archive: Path) -> None:
    """Unpack Kaggle's competition ZIP: it holds one .7z file per CSV.

    Pure Python (zipfile + py7zr), so no 7-Zip install is needed. Each CSV is
    unpacked and moved into place one at a time, to keep the disk space low.
    """
    import py7zr

    with zipfile.ZipFile(archive) as zf, tempfile.TemporaryDirectory(dir=KKBOX_DIR) as tmp:
        for member in zf.namelist():
            name = Path(member).name
            csv_name = name.removesuffix(".7z")
            if csv_name not in KKBOX_FILES or (KKBOX_DIR / csv_name).exists():
                continue
            print(f"  unpacking {csv_name} ...")
            work = Path(tmp) / csv_name.removesuffix(".csv")
            work.mkdir()
            extracted = Path(zf.extract(member, work))
            if extracted.suffix == ".7z":
                with py7zr.SevenZipFile(extracted, mode="r") as seven:
                    seven.extractall(path=work)
                extracted.unlink()
            found = next(p for p in work.rglob(csv_name) if p.is_file())
            shutil.move(str(found), KKBOX_DIR / csv_name)
            shutil.rmtree(work)


def _download_kkbox() -> bool:
    """Notebook 2's data: the KKBox competition files, via the user's Kaggle token."""
    KKBOX_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
    except (Exception, SystemExit) as error:  # the Kaggle package exits on bad credentials
        detail = "" if isinstance(error, SystemExit) else f" ({error})"
        print(f"\nCould not sign in to Kaggle with your token{detail}.")
        print("Create a new token on kaggle.com (see START_HERE.md) and try again.")
        return False
    print("\nDownloading the KKBox competition files (about 400 MB)...")
    try:
        api.competition_download_files(KKBOX_COMPETITION, path=str(KKBOX_DIR), quiet=False)
    except Exception as error:
        text = str(error)
        print(f"\nThe download didn't work ({text[:300]}).")
        if "403" in text or "Forbidden" in text:
            print("Kaggle refused it: your account has not accepted the competition rules yet.")
            print(f"Open {KKBOX_RULES_URL} , sign in, click 'Late Submission' or 'I Understand and Accept',")
            print("then choose this option again.")
        elif "401" in text or "Unauthorized" in text:
            print("Kaggle did not accept your token. Create a new one (see START_HERE.md) and try again.")
        return False
    archive = next(KKBOX_DIR.glob(f"{KKBOX_COMPETITION}.*"), None)
    if archive is None:
        print("The download finished but the archive is missing; please try again.")
        return False
    print("Unpacking (about 1.8 GB of CSV files; a few minutes)...")
    _extract_kkbox(archive)
    if _missing("02_collaborative_filtering.ipynb"):
        print("Some files are still missing after unpacking:", _missing("02_collaborative_filtering.ipynb"))
        return False
    archive.unlink()
    print("Done: the KKBox files are in data/collaborative/raw.")
    return True


def _explain_kkbox() -> None:
    print("\nNotebook 2 needs the KKBox competition files from Kaggle. This part is optional:")
    print("notebook 1 and the app work without it.")
    print("  - You need a free Kaggle account, and you must accept the competition rules once in")
    print(f"    your browser: {KKBOX_RULES_URL}")
    print("  - Download: about 400 MB. Disk: about 1.8 GB once unpacked (2.5 GB free while unpacking).")
    print("  - Time: roughly 5-15 minutes, depending on your internet connection.")
    print("  - Notebook 2 itself needs about 4 GB of free memory (8 GB RAM computer or more).")
    print("Step-by-step guide: START_HERE.md, section 'Get the data for notebook 2'.")


def ensure_notebook_data() -> bool:
    """Make sure each notebook's data is present, fetching what can be fetched.

    Notebook 1 (Spotify, ~9 MB) is downloaded automatically without an account.
    Notebook 2 (KKBox, ~400 MB download, ~1.8 GB unpacked) needs a Kaggle token and
    accepted competition rules; with a token the launcher offers to download and
    unpack it. Returns whether every notebook's data is present.
    """
    if _missing("01_content_based.ipynb"):
        _download_spotify()
    if _missing("02_collaborative_filtering.ipynb"):
        _explain_kkbox()
        if not _kaggle_token_found():
            print("\nNo Kaggle token found on this computer.")
            pasted = _ask("If you have one, paste it here and press Enter (or just press Enter to skip): ")
            if pasted:
                try:
                    _save_pasted_token(pasted)
                except (OSError, ValueError) as error:
                    print(f"Could not save that token ({error}).")
        if _kaggle_token_found():
            if _ask("\nDownload the KKBox files now? [y/N] ").lower() == "y":
                _download_kkbox()
            else:
                print("Skipped.")
        else:
            print("Skipped: notebook 2 will stop at its first cell until the files are there.")
    for notebook in NOTEBOOK_DATA:
        print(f"  {notebook}: {'ready' if not _missing(notebook) else 'data missing'}")
    return not any(_missing(notebook) for notebook in NOTEBOOK_DATA)


# --- macOS: the OpenMP runtime LightGBM and XGBoost need ---------------------
# Their macOS wheels expect Homebrew's libomp (`brew install libomp`), which most
# Macs don't have. If they can't load, fetch libomp from conda-forge (~300 kB,
# no account, checksum-pinned) into .libomp/ and point the app and notebooks at it.

LIBOMP_DIR = PROJECT_ROOT / ".libomp"
LIBOMP_PACKAGES = {  # platform.machine() -> (conda-forge file, sha256)
    "arm64": ("osx-arm64/llvm-openmp-23.1.2-hdb3d66b_0.conda",
              "f9aea67117079c00b9663ee0bbca7e5409dbe94625c34e68881ce2c97c3d47c9"),
    "x86_64": ("osx-64/llvm-openmp-23.1.2-h8c1e6b9_0.conda",
               "3c17127e2a94d60a8babef57c66d643374c5b08c896ec5f0987a62ecfb81c8ee"),
}


def _openmp_loads() -> bool:
    probe = [sys.executable, "-c", "import lightgbm, xgboost"]
    return subprocess.run(probe, env=_env(), capture_output=True).returncode == 0


def ensure_openmp() -> bool:
    if sys.platform != "darwin" or _openmp_loads():
        return True
    import hashlib
    import io
    import platform
    import ssl
    import tarfile

    import certifi
    from backports import zstd

    print("\nOne-time fix: downloading the OpenMP runtime that LightGBM and XGBoost need on macOS (~300 kB)...")
    name, sha256 = LIBOMP_PACKAGES[platform.machine()]
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        url = f"https://conda.anaconda.org/conda-forge/{name}"
        with urllib.request.urlopen(url, timeout=60, context=context) as response:
            package = response.read()
        if hashlib.sha256(package).hexdigest() != sha256:
            raise OSError("checksum mismatch")
        with zipfile.ZipFile(io.BytesIO(package)) as conda:
            inner = next(n for n in conda.namelist() if n.startswith("pkg-") and n.endswith(".tar.zst"))
            tar_bytes = zstd.decompress(conda.read(inner))
        with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
            dylib = tar.extractfile("lib/libomp.dylib").read()
        LIBOMP_DIR.mkdir(exist_ok=True)
        (LIBOMP_DIR / "libomp.dylib").write_bytes(dylib)
    except (OSError, KeyError, StopIteration, zipfile.BadZipFile, tarfile.TarError) as error:
        print(f"That didn't work ({error}). The app's recommenders and the notebooks need it.")
        print("Fix: install Homebrew (https://brew.sh), run `brew install libomp`, then start again.")
        return False
    ok = _openmp_loads()
    print("Done." if ok else "LightGBM still can't load; see START_HERE.md, 'If something goes wrong'.")
    return ok


# --- Shared ------------------------------------------------------------------


def _free_port(start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free port found.")


def _open_when_ready(url: str, health: str, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=2) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except OSError:
            time.sleep(1)
    print(f"\nThe app is taking a while to start. Open {url} in your browser when it's ready.")


def _env() -> dict:
    env = os.environ.copy()
    # Keep Jupyter's kernels and settings inside the project, so an older
    # Jupyter install elsewhere on the computer can't take over the kernel.
    env["JUPYTER_DATA_DIR"] = str(PROJECT_ROOT / ".jupyter_data")
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    if sys.platform == "darwin" and (LIBOMP_DIR / "libomp.dylib").exists():
        # Used only if Homebrew's libomp isn't there (see ensure_openmp).
        env["DYLD_FALLBACK_LIBRARY_PATH"] = f"{LIBOMP_DIR}:/usr/local/lib:/usr/lib"
    return env


def run_notebooks() -> None:
    # Notebook 1 usually works even when notebook 2's optional data is missing,
    # so JupyterLab opens either way.
    ensure_openmp()
    ensure_notebook_data()
    print("\nOpening JupyterLab in your browser...")
    print("Open a notebook from the list on the left, then use Run > Run All Cells.")
    print("To stop: close the browser tab, then press Ctrl+C in this window.\n")
    port = _free_port(8888)
    subprocess.run([sys.executable, "-m", "jupyter", "lab", f"--notebook-dir={NOTEBOOKS}", f"--port={port}"],
                   cwd=PROJECT_ROOT, env=_env())


def run_app() -> None:
    ensure_openmp()
    port = _free_port(8501)
    url = f"http://localhost:{port}"
    print(f"\nStarting the app at {url} (your browser opens by itself)...")
    print("To stop: press Ctrl+C in this window.\n")
    threading.Thread(target=_open_when_ready, args=(url, f"{url}/_stcore/health"), daemon=True).start()
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP), "--server.headless=true",
                    f"--server.port={port}"], cwd=PROJECT_ROOT, env=_env())


def check() -> None:
    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    for name in ("streamlit", "jupyterlab", "ipykernel"):
        __import__(name)
        print(f"  {name}: ok")
    if sys.platform == "darwin":
        print("OpenMP for LightGBM/XGBoost:", "ok" if ensure_openmp() else "missing")
    demo = PROJECT_ROOT / "data" / "demo"
    print("App data (bundled samples):", "ok" if (demo / "content_tracks.csv").exists() else "missing")
    print("Notebook 1 data (Spotify):", "ok" if not _missing("01_content_based.ipynb")
          else "missing (the launcher downloads it, no account needed)")
    print("Notebook 2 data (KKBox):", "ok" if not _missing("02_collaborative_filtering.ipynb")
          else "missing (optional; needs a free Kaggle account, see START_HERE.md)")
    print("Kaggle token:", "found" if _kaggle_token_found() else "not found (only needed for notebook 2's data)")
    print("Setup looks good.")


def main() -> None:
    choice = sys.argv[1] if len(sys.argv) > 1 else None
    actions = {"notebooks": run_notebooks, "app": run_app, "check": check}
    if choice:
        actions[choice]()
        return
    while True:
        print(f"\n=== {PROJECT_NAME} ===")
        print("  1  Open the notebooks (JupyterLab)")
        print("  2  Open the app (Streamlit)")
        print("  3  Quit")
        picked = input("Type 1, 2 or 3 and press Enter: ").strip()
        try:
            if picked == "1":
                run_notebooks()
            elif picked == "2":
                run_app()
            elif picked == "3":
                return
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
