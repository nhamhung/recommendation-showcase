"""CLI: look up a track by name and print its top-k recommendations.

The parallel deliverable to the other two projects' make_submission.py —
this dataset has no Kaggle leaderboard to submit to, so the equivalent
"prove it works end-to-end" artifact is a working recommendation, not a
scored CSV.
"""

import argparse

from recommendation_showcase.content_based import config, data, model


def find_track(df, query: str):
    """Case-insensitive substring match on track_name; returns the first
    hit. Good enough for a CLI demo — the Streamlit app's Predict page
    uses a proper searchable dropdown instead.
    """
    matches = df[df["track_name"].str.contains(query, case=False, na=False)]
    if matches.empty:
        raise SystemExit(f"No track found matching {query!r}")
    return matches.iloc[[0]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track_name", help="Substring to search for in track_name")
    parser.add_argument("-k", type=int, default=config.DEFAULT_TOP_K)
    args = parser.parse_args()

    df = data.load_tracks()
    recommender = model.load_pipeline()

    seed_row = find_track(df, args.track_name)
    print(f"Seed track: {seed_row['track_name'].iloc[0]} — {seed_row['artists'].iloc[0]}")

    X_query, _ = data.split_features_target(seed_row)
    recs = recommender.recommend(X_query, k=args.k, exclude_own_id=True)

    print(f"\nTop {args.k} recommendations:")
    for _, row in recs.iterrows():
        track = df.loc[row["recommended_id"]]
        print(
            f"  {int(row['rank']):>2}. {track['track_name']} — {track['artists']} "
            f"({track['track_genre']}, distance={row['distance']:.3f})"
        )


if __name__ == "__main__":
    main()
