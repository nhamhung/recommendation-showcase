"""Asset-aware deployment smoke check for both recommendation models."""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recommendation_showcase.collaborative import model as collaborative_model  # noqa: E402
from recommendation_showcase.content_based import model as content_model  # noqa: E402


def check_models() -> None:
    if content_model.load_pipeline() is None:
        raise RuntimeError("The content-based model did not load.")
    if collaborative_model.load_pipeline() is None:
        raise RuntimeError("The collaborative model did not load.")


def check_streamlit() -> None:
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.headless=true", "--server.port=8501"],
        cwd=PROJECT_ROOT,
    )
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Streamlit exited before becoming healthy.")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError:
                time.sleep(1)
        raise RuntimeError("Streamlit health endpoint did not become ready within 90 seconds.")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    check_models()
    check_streamlit()
    print("recommendation-showcase deployment smoke check passed")
