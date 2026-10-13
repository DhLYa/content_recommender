"""
Replace MAL usernames with random user IDs in the raw user files.

Reads the original files from data/private/ (gitignored) and writes anonymised copies to data/raw/:
    data/private/user_anime_ratings_wide.csv  ->  data/raw/user_anime_ratings_wide.csv.gz
    data/private/user_profiles.csv            ->  data/raw/user_profiles.csv

IDs are drawn at random rather than hashed, since a hash of a public username can be reversed by
hashing candidate usernames. The username -> ID map is saved to data/private/user_id_map.csv and
reused on later runs, so IDs stay stable.

Usage:
    python scripts/anonymise_users.py
"""
import csv
import gzip
import io
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private"
RAW = ROOT / "data" / "raw"
MAP_PATH = PRIVATE / "user_id_map.csv"


def read_usernames(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        return {row[0] for row in reader}


def load_or_create_map(usernames):
    if MAP_PATH.exists():
        with open(MAP_PATH, newline="", encoding="utf-8") as f:
            user_map = {row["username"]: row["user_id"] for row in csv.DictReader(f)}
        missing = usernames - user_map.keys()
        if missing:
            raise ValueError(f"{len(missing)} usernames are not in {MAP_PATH}; delete it to rebuild all IDs")
        return user_map

    # Random order from the OS, so IDs carry no information about the usernames
    shuffled = sorted(usernames)
    random.SystemRandom().shuffle(shuffled)
    user_map = {name: f"u{i:04d}" for i, name in enumerate(shuffled, start=1)}

    with open(MAP_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["username", "user_id"])
        writer.writerows(user_map.items())
    return user_map


def anonymise(src, dst_file, user_map):
    """Copy a CSV whose first column is the username, replacing it with the user ID."""
    with open(src, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        writer = csv.writer(dst_file, lineterminator="\n")
        header = next(reader)
        writer.writerow(["user"] + header[1:])
        for row in reader:
            writer.writerow([user_map[row[0]]] + row[1:])


def main():
    ratings_src = PRIVATE / "user_anime_ratings_wide.csv"
    profiles_src = PRIVATE / "user_profiles.csv"

    user_map = load_or_create_map(read_usernames(ratings_src) | read_usernames(profiles_src))

    # mtime=0 keeps the gzip output byte-identical across runs
    with open(RAW / "user_anime_ratings_wide.csv.gz", "wb") as raw_file:
        with gzip.GzipFile(fileobj=raw_file, mode="wb", mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                anonymise(ratings_src, text, user_map)

    with open(RAW / "user_profiles.csv", "w", newline="", encoding="utf-8") as f:
        anonymise(profiles_src, f, user_map)

    print(f"{len(user_map)} users anonymised into {RAW}")


if __name__ == "__main__":
    main()
