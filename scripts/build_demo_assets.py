"""Build the small, redistributable data bundle used by the hosted app.

The Spotify sample comes from the public source dataset. KKBox identities are
drawn from the fitted model's frequency maps, while all displayed member/song
metadata is synthetic because the competition files are not redistributed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recommendation_showcase.content_based import config as cb_config  # noqa: E402
from recommendation_showcase.content_based import data as cb_data  # noqa: E402
from recommendation_showcase.content_based import model as cb_model  # noqa: E402
from recommendation_showcase.collaborative import config as coll_config  # noqa: E402
from recommendation_showcase.collaborative import model as coll_model  # noqa: E402


def build_content_sample(spotify_csv: Path, output_dir: Path) -> None:
    tracks = pd.read_csv(spotify_csv)
    tracks = tracks.drop(
        columns=[column for column in tracks if column.startswith("Unnamed")],
        errors="ignore",
    ).drop_duplicates(subset=cb_config.ID_COL, keep="first")
    tracks = tracks.set_index(cb_config.ID_COL)

    recommender = cb_model.load_pipeline()
    model_ids = pd.Index(recommender.train_ids_)
    tracks = tracks.loc[tracks.index.intersection(model_ids)]

    # Keep recognizable popular tracks plus enough seeds from every genre for
    # the evaluation/interpretability pages to remain meaningful.
    ranked = tracks.sort_values(cb_config.POPULARITY_COL, ascending=False)
    seed_ids = ranked.head(250).index.union(
        ranked.groupby(cb_config.GENRE_COL, group_keys=False).head(4).index
    )
    seed_rows = tracks.loc[seed_ids]
    X_seed, _ = cb_data.split_features_target(seed_rows)
    recommendations = recommender.recommend(X_seed, k=20, exclude_own_id=True)
    selected_ids = seed_ids.union(pd.Index(recommendations["recommended_id"].unique()))

    sample = tracks.loc[tracks.index.intersection(selected_ids)].copy()
    sample.index.name = cb_config.ID_COL
    sample["is_demo_seed"] = sample.index.isin(seed_ids)
    sample.reset_index().to_csv(output_dir / "content_tracks.csv", index=False)


def build_collaborative_sample(output_dir: Path) -> None:
    pipeline = coll_model.load_pipeline()
    counts = pipeline.named_steps["interaction_counts"]
    users = counts.user_counts_.head(200).index.to_list()
    songs = counts.song_counts_.head(400).index.to_list()

    cities = [1, 3, 4, 5, 6, 13, 15, 22]
    registration_methods = [3, 4, 7, 9, 13]
    members = pd.DataFrame(
        {
            coll_config.USER_COL: users,
            "city": [cities[i % len(cities)] for i in range(len(users))],
            "bd": [18 + (i % 48) for i in range(len(users))],
            "gender": ["female" if i % 2 else "male" for i in range(len(users))],
            "registered_via": [registration_methods[i % len(registration_methods)] for i in range(len(users))],
            "registration_init_time": [20120101 + (i % 5) * 10000 for i in range(len(users))],
            "expiration_date": [20180101 + (i % 9) * 100 for i in range(len(users))],
        }
    )

    genres = sorted(pipeline.named_steps["group_rare"].kept_categories_["primary_genre"] - {"unknown"})
    artists = counts.artist_counts_.head(50).index.to_list()
    languages = [-1.0, 3.0, 10.0, 17.0, 24.0, 31.0, 52.0]
    songs_frame = pd.DataFrame(
        {
            coll_config.SONG_COL: songs,
            "song_length": [150_000 + (i % 180) * 1_000 for i in range(len(songs))],
            "genre_ids": [genres[i % len(genres)] for i in range(len(songs))],
            "artist_name": [artists[i % len(artists)] for i in range(len(songs))],
            "composer": [f"Demo composer {i % 30:02d}" if i % 3 else np.nan for i in range(len(songs))],
            "lyricist": [f"Demo lyricist {i % 20:02d}" if i % 6 == 0 else np.nan for i in range(len(songs))],
            "language": [languages[i % len(languages)] for i in range(len(songs))],
        }
    )
    extra = pd.DataFrame(
        {
            coll_config.SONG_COL: songs,
            "name": [f"Demo track {i + 1:03d}" for i in range(len(songs))],
            "isrc": [f"TWAAA{10 + i % 8:02d}{i:05d}" for i in range(len(songs))],
        }
    )

    rng = np.random.default_rng(coll_config.RANDOM_SEED)
    row_count = 6_000
    contexts = {
        "source_system_tab": ["discover", "explore", "my library", "radio", "search"],
        "source_screen_name": ["Discover Feature", "Explore", "Local playlist more", "My library", "Radio"],
        "source_type": ["local-library", "online-playlist", "radio", "song", "song-based-playlist"],
    }
    interactions = pd.DataFrame(
        {
            coll_config.USER_COL: rng.choice(users, row_count),
            coll_config.SONG_COL: rng.choice(songs, row_count),
            **{column: rng.choice(values, row_count) for column, values in contexts.items()},
        }
    )
    panel = (
        interactions.merge(songs_frame, on=coll_config.SONG_COL, how="left")
        .merge(members, on=coll_config.USER_COL, how="left")
        .merge(extra, on=coll_config.SONG_COL, how="left")
    )
    probabilities = pipeline.predict_proba(panel)[:, 1]
    interactions[coll_config.TARGET_COL] = rng.binomial(1, probabilities)

    interactions.to_csv(output_dir / "collaborative_train.csv", index=False)
    songs_frame.to_csv(output_dir / "collaborative_songs.csv", index=False)
    members.to_csv(output_dir / "collaborative_members.csv", index=False)
    extra.to_csv(output_dir / "collaborative_song_extra_info.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spotify-csv", type=Path, required=True)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "data" / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    build_content_sample(args.spotify_csv, output_dir)
    build_collaborative_sample(output_dir)
    print(f"Wrote demo assets to {output_dir}")


if __name__ == "__main__":
    main()
