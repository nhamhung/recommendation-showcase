# Start here: run the Recommendation Showcase notebooks and app on your computer

This guide is for anyone, including people who have never installed Python. You **don't** need
Python, Git, VS Code, 7-Zip or a Streamlit account. You double-click one file, and it sets
everything up. The app and notebook 1 also need **no Kaggle account**; only the optional data for
notebook 2 does.

**What you get**

- **Two notebooks**: step-by-step walk-throughs of two ways to recommend songs, which you can run,
  change and re-run in your web browser.
  - `01_content_based.ipynb`: recommend songs that *sound* similar, from Spotify audio features.
  - `02_collaborative_filtering.ipynb`: predict whether a KKBox listener will replay a song, from
    real listening history (optional data, see [Get the data for notebook 2](#get-the-data-for-notebook-2-optional)).
- **The app**: the interactive Recommendation Systems Showcase (both recommenders side by side)
  running on your own computer, on small sample data that ships with the project.

**What you need**

- Windows 10 or 11, or macOS 12 or newer (Apple silicon or Intel).
- An internet connection the first time (it downloads about 400 MB of Python and packages).
- About 2 GB of free disk space (another 2.5 GB if you get the notebook 2 data).
- For notebook 2: about 4 GB of free memory (a computer with 8 GB of RAM or more).

---

## Windows

### 1. Download the project

1. Go to <https://github.com/nhamhung/recommendation-showcase>.
2. Click the green **Code** button, then **Download ZIP**.
3. Open your **Downloads** folder, right-click `recommendation-showcase-main.zip` and choose
   **Extract All...**. Extract it to a simple place such as
   `C:\Users\<you>\Documents\recommendation-showcase-main`.

> Don't run it from inside the ZIP file, and avoid folders synced by OneDrive if you can
> (they work, but setup is slower).

### 2. Start it

1. Open the extracted folder and double-click **`start_windows.bat`**.
2. If Windows shows **"Windows protected your PC"**, click **More info**, then **Run anyway**.
   (This appears because the file was downloaded from the internet.)
3. A black window opens. **The first time only**, it installs *uv* (a small tool that sets up
   Python for this project) and downloads Python and the packages. This takes about 2–5 minutes.
   Later starts take a few seconds.
4. When it's ready, you see a menu:

   ```
   === Recommendation Showcase ===
     1  Open the notebooks (JupyterLab)
     2  Open the app (Streamlit)
     3  Quit
   Type 1, 2 or 3 and press Enter:
   ```

5. If Windows Firewall asks whether to allow Python, click **Allow** (it only talks to your own
   computer).

### 3. Stop it

Click the black window and press **Ctrl + C**. If it asks *"Terminate batch job (Y/N)?"*, type
**Y** and press Enter. Or simply close the black window.

---

## macOS

### 1. Download the project

1. Go to <https://github.com/nhamhung/recommendation-showcase>.
2. Click the green **Code** button, then **Download ZIP**.
3. Open your **Downloads** folder and double-click `recommendation-showcase-main.zip` to unzip it.
   Move the `recommendation-showcase-main` folder somewhere simple, such as your **Documents**
   folder.

### 2. Start it

1. Open the folder. **The first time**, right-click (or Control-click) **`start_mac.command`**,
   choose **Open**, then click **Open** again in the warning. (macOS asks because the file was
   downloaded from the internet. After this, a normal double-click works.)
   - If macOS says it *"cannot be opened because Apple cannot check it"* and there is no Open
     button: open **System Settings → Privacy & Security**, scroll down, click **Open Anyway**
     next to `start_mac.command`, then try again.
2. A Terminal window opens. **The first time only**, it installs *uv* (a small tool that sets up
   Python for this project) and downloads Python and the packages. This takes about 2–5 minutes.
   Later starts take a few seconds. On most Macs it also prints *"One-time fix: downloading the
   OpenMP runtime..."*: that is a small library (about 300 kB) the models need on macOS, and it is
   stored inside the project folder.
3. When it's ready, you see the same menu as on Windows: type **1** (notebooks) or **2** (app)
   and press Return.

### 3. Stop it

Click the Terminal window and press **Control + C**, then close the window.

---

## Using the notebooks (choice 1)

Before JupyterLab opens, the launcher checks each notebook's data:

- **Notebook 1** needs the Spotify Tracks dataset. If it isn't there yet, the launcher downloads it
  by itself (about 9 MB, a few seconds, no account needed).
- **Notebook 2** needs the KKBox competition files. If they are missing, the launcher explains how
  to get them, asks for a Kaggle token if it can't find one (just press Enter to skip), and then
  opens JupyterLab anyway. See [Get the data for notebook 2](#get-the-data-for-notebook-2-optional).

JupyterLab then opens in your web browser.

1. In the file list on the left, double-click **`01_content_based.ipynb`** (or
   **`02_collaborative_filtering.ipynb`**).
2. Run it from top to bottom with **Run → Run All Cells**. A `[*]` next to a cell means it is
   still running. Notebook 1 takes about 2 minutes; notebook 2 about 5 minutes. The last cell of
   each prints how long it took.
3. To run just one cell, click it and press **Shift + Enter**.
4. **Experiment freely.** Change a number or a song name in a cell and run it again (notebook 1
   ends with `recommend_for("Clair de lune", k=5)`: try your own favourite track).
5. If things get into a strange state, use **Kernel → Restart Kernel and Run All Cells**.

Notebook 2 is written so it never downloads data or overwrites the project's models. Some of its
leaderboard sections use large files that only exist after running the full leaderboard
pipeline (`scripts/make_leaderboard_submission.py`); without them those cells print a short
"skipping" message, and everything else still runs.

Your edits are saved in the notebook file. To keep the original, first use **File → Save
Notebook As...** and work on the copy (or download the ZIP again later).

## Using the app (choice 2)

The app opens in your browser by itself (if it doesn't, copy the `http://localhost:...` address
shown in the black/Terminal window into your browser). It runs on small samples of both datasets
that ship with the project, so it needs no download and no account. The sidebar has two groups:

- **Content-Based (Spotify)**
  - **Recommend**: search for a track or artist, pick a song, and get similar-sounding tracks
    with the audio features that make them similar.
  - **Overview**, **Feature Engineering**, **Model Insights**: the data, the features and what
    the model learned.
- **Collaborative Filtering (KKBox)**
  - **Predict**: pick a listener and a song and see how likely they are to replay it within a month.
  - **Overview**, **Feature Engineering**, **Model Insights**: the same tour for this approach.

Nothing you do in the app changes the project files.

## Next time

Just double-click `start_windows.bat` or `start_mac.command` again. No internet needed once the
first setup has finished (and the data is downloaded).

---

## Get the data for notebook 2 (optional)

Notebook 2 uses the real data from the *WSDM – KKBox's Music Recommendation Challenge* on Kaggle.
Kaggle doesn't allow these files to be shared, so each person downloads them with their own free
Kaggle account. You do steps 1–3 once.

**Sizes and time**: about 400 MB to download, about 1.8 GB on disk once unpacked (keep 2.5 GB free
while it unpacks), and roughly 5–15 minutes in total depending on your internet connection.

1. **Create a free Kaggle account** at <https://www.kaggle.com> (**Register**, top right).
   Kaggle may ask you to verify your phone number before you can join competitions.
2. **Accept the competition rules** (Kaggle refuses the download until you do): while signed in,
   open <https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge/rules>, scroll
   down and click **I Understand and Accept** (on some pages the button says **Late Submission**
   or **Join Competition**; any of them works).
3. **Create an API token**: click your profile picture (top right) → **Settings** → section
   **API** → **Generate New Token**. Kaggle shows a long code: click the copy button next to it.
   Keep it private, like a password.
4. **Run the launcher**: double-click the start file and choose **1** (notebooks). The launcher
   says it found no Kaggle token and asks you to paste one:
   - **Windows:** right-click inside the black window to paste (or press **Ctrl + V**), then
     press Enter.
   - **macOS:** press **Command + V** in the Terminal window, then press Return.

   It saves the token for next time, asks *"Download the KKBox files now? [y/N]"*: type **y** and
   press Enter. It downloads and unpacks everything by itself (no 7-Zip needed), then opens
   JupyterLab.

**Where the token is saved** (if you'd rather put it there yourself, or want to delete it later):

- **Windows:** `C:\Users\<you>\.kaggle\access_token`
- **macOS:** `~/.kaggle/access_token` (the `.kaggle` folder in your home folder; in Finder press
  **Command + Shift + .** to see hidden folders).

If you downloaded an older-style `kaggle.json` file from Kaggle instead, put it in that same
`.kaggle` folder (as `kaggle.json`); the launcher finds it too.

If the download says Kaggle *refused* it (error 403), step 2 hasn't been done with the same
account: accept the rules and choose 1 again. If it was interrupted, just choose 1 again.

---

## Optional: use VS Code instead of the browser

If you prefer VS Code for the notebooks:

1. Run the start file once (above), so the project's Python environment (the `.venv` folder)
   exists. You can quit from the menu.
2. Install VS Code from <https://code.visualstudio.com>.
3. In VS Code: **File → Open Folder...** and choose the project folder. Accept the prompt to
   install the recommended **Python** and **Jupyter** extensions.
4. Open `notebooks/01_content_based.ipynb`. Click **Select Kernel** (top right) →
   **Python Environments** → the one marked **`.venv`** (Recommended).

That is the only kernel step, and you do it once. (Use the launcher once first to fetch the data.)
On a Mac, VS Code doesn't get the launcher's OpenMP fix: if a cell fails with *"Library not
loaded: @rpath/libomp.dylib"*, either use the launcher's JupyterLab instead, or install
[Homebrew](https://brew.sh) and run `brew install libomp` in Terminal once.

---

## If something goes wrong

| Problem | What to do |
|---|---|
| Installing uv fails (often on work computers that block scripts) | Install uv another way, then double-click the start file again. **Windows:** open *Command Prompt* and run `winget install --id=astral-sh.uv -e`. **macOS:** if you use Homebrew, run `brew install uv` in Terminal. Or follow <https://docs.astral.sh/uv/getting-started/installation/>. |
| macOS: *"permission denied"* or the file opens in a text editor | Open **Terminal**, type `bash ` (with a space), drag `start_mac.command` into the Terminal window, and press Return. |
| The browser didn't open | Copy the `http://localhost:...` address from the black/Terminal window into your browser. |
| *"Address already in use"* | Another copy is probably running. Close the other black/Terminal window, or just pick the option again: the launcher picks a free port. |
| macOS: *"Library not loaded: @rpath/libomp.dylib"* or *"brew install libomp"* | The launcher normally fixes this by itself (it needs the internet once). If it says the fix didn't work, install [Homebrew](https://brew.sh), run `brew install libomp` in Terminal, and start again. |
| Notebook 1 says `dataset.csv` is missing | Choose 1 in the launcher again while connected to the internet. Or download it by hand from <https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset> (**Download**), unzip it and put `dataset.csv` in the project's `data/content_based/raw` folder. |
| Notebook 2 stops at its first cell with *"Missing [...]"* | Its data isn't there yet: see [Get the data for notebook 2](#get-the-data-for-notebook-2-optional). |
| Notebook 2 is very slow or the kernel dies | It needs about 4 GB of free memory. Close other programs (especially browser tabs) and use **Kernel → Restart Kernel and Run All Cells**. |
| Errors after editing a notebook | **Kernel → Restart Kernel and Run All Cells**. Still broken? Download the ZIP again for a fresh copy. |
| Something else is broken | Delete the `.venv` folder inside the project and double-click the start file again: it rebuilds a clean setup. |

## Uninstall

1. Delete the project folder (this also deletes any downloaded data).
2. Optional, to remove uv and the downloaded Python too:
   - **Windows:** delete `uv.exe` and `uvx.exe` from `C:\Users\<you>\.local\bin`, the folder
     `%APPDATA%\uv`, and the folder `%LOCALAPPDATA%\uv`.
   - **macOS:** in Terminal, run `rm -rf ~/.local/bin/uv ~/.local/bin/uvx ~/.local/share/uv ~/.cache/uv`.
3. Optional, to remove your Kaggle token: delete the `.kaggle` folder in your home folder
   (`C:\Users\<you>\.kaggle` or `~/.kaggle`). You can also revoke the token on kaggle.com under
   **Settings → API**.

---

## For developers

The project is a standard [uv](https://docs.astral.sh/uv/) project: `pyproject.toml` lists the
direct dependencies at the exact versions it was tested with, and `uv.lock` pins everything else.

```bash
uv sync                                   # create .venv (add --no-dev to skip pytest/ruff)
uv run pytest                             # tests
uv run jupyter lab notebooks              # notebooks
uv run streamlit run app/streamlit_app.py # app
uv sync --group leaderboard-nn            # adds PyTorch, only for the experimental NN leaderboard member
```

On Intel Macs the lock uses NumPy 2.3 instead of 2.5, because numba (needed by SHAP) has no
newer Intel-Mac builds. LightGBM's and XGBoost's macOS wheels need an OpenMP runtime
(`brew install libomp`); on Macs without it, the launcher downloads conda-forge's `llvm-openmp`
(checksum-pinned) into `.libomp/` and sets `DYLD_FALLBACK_LIBRARY_PATH` for the app and notebooks. `scripts/launch.py` is what the start files run. `requirements*.txt` and
`app/requirements.txt` stay for Streamlit Community Cloud, Docker and CI. To publish your own
changes to GitHub, see the setup manual at
<https://github.com/nhamhung/nhamhung.github.io/blob/main/README.md> (Git, VS Code, clone, push).
