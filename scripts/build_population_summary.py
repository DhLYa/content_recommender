"""
Summarise the Kaggle "Anime Dataset 2023" user-filtered.csv (~1.5 GB) into two small files
that the notebook uses for the sample vs population comparison:

- data/external/population_rating_counts.csv     (rating, count)
- data/external/population_user_mean_ratings.csv (user_id, n_ratings, mean_rating)

A rating of 0 in the source means "watched but not scored" on MAL, so those rows are excluded.

Usage:
    python scripts/build_population_summary.py path/to/user-filtered.csv
"""
import sys
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"


def main(source):
    rating_counts = pd.Series(dtype="int64")
    user_totals = []

    reader = pd.read_csv(source, usecols=["user_id", "rating"], dtype={"user_id": "int32", "rating": "int8"}, chunksize=10_000_000)
    for chunk in reader:
        chunk = chunk[chunk["rating"] > 0]
        rating_counts = rating_counts.add(chunk["rating"].value_counts(), fill_value=0)
        user_totals.append(chunk.groupby("user_id")["rating"].agg(["sum", "count"]))

    # A user's rows can span chunks, so aggregate the partial sums again
    users = pd.concat(user_totals).groupby(level=0).sum()
    user_means = pd.DataFrame({
        "user_id": users.index,
        "n_ratings": users["count"].to_numpy(),
        "mean_rating": (users["sum"] / users["count"]).to_numpy(),
    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rating_counts = rating_counts.astype("int64").sort_index().rename_axis("rating").rename("count")
    rating_counts.to_csv(OUT_DIR / "population_rating_counts.csv")
    user_means.to_csv(OUT_DIR / "population_user_mean_ratings.csv", index=False)

    print(f"{int(rating_counts.sum())} scored ratings from {len(user_means)} users written to {OUT_DIR}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python scripts/build_population_summary.py path/to/user-filtered.csv")
    main(Path(sys.argv[1]))
