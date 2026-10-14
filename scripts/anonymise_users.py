"""
Anonymise the AniList collection for committing.

Reads the output of anilist_ingestion.ipynb (gitignored, contains usernames and AniList user IDs) and
writes anonymised copies to SRC_anonymised/ (by default data/raw/anilist_anonymised/):
    ratings.csv.gz, user_profiles.csv, users.csv, anime_metadata.csv (without synopsis text) and
    anime_synopses.csv (the synopsis text, gitignored)

Seed users get random IDs and lose their username and AniList user ID. Manual users (added by name
in the notebook as people of interest) keep their AniList username as their user ID.

The ratings file also loses its date columns and anime titles. The dates are the main way an anonymised
list could be matched back to a public profile, and without them the file stays under GitHub's 100 MB
limit (10,000 users: ~38 MB instead of ~200 MB). Titles are in anime_metadata.csv.

Synopses are copyrighted text (mostly MyAnimeList's), so they go to anime_synopses.csv, which is gitignored.
Only the embeddings built from them (artifacts/synopsis_embeddings.npy) are committed.

IDs are drawn at random rather than hashed, since a hash of a public username or ID can be reversed
by hashing candidates. The map is saved to data/private/<src name>_user_id_map.csv (gitignored) and
reused on later runs, so IDs stay stable; new users get new IDs as the collection grows.

Usage:
    python scripts/anonymise_users.py [SRC_DIR [DST_DIR]]
"""
import csv
import gzip
import io
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private"
RAW = ROOT / "data" / "raw"


def load_or_extend_map(map_path, seed_ids):
    """AniList user ID -> random ID for seed users. New users get new IDs; existing IDs never change."""
    user_map = {}
    if map_path.exists():
        with open(map_path, newline="", encoding="utf-8") as f:
            user_map = {row["anilist_user_id"]: row["user_id"] for row in csv.DictReader(f)}

    new = sorted(seed_ids - user_map.keys())
    if new:
        # The hyphen can't appear in an AniList username, so random IDs never clash with manual users' names
        random.SystemRandom().shuffle(new)
        start = max((int(v.split("-")[1]) for v in user_map.values()), default=0)
        user_map.update({uid: f"anon-{i:05d}" for i, uid in enumerate(new, start=start + 1)})
        map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(map_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["anilist_user_id", "user_id"])
            writer.writerows(user_map.items())
    return user_map, len(new)


def anonymise_csv(src_file, dst_file, id_for, id_column, drop):
    """Copy a CSV, putting the user's ID in id_column (first) and dropping the identifying columns."""
    reader = csv.DictReader(src_file)
    kept = [c for c in reader.fieldnames if c not in drop and c != id_column]
    writer = csv.writer(dst_file, lineterminator="\n")
    writer.writerow([id_column] + kept)
    for row in reader:
        writer.writerow([id_for(row)] + [row[c] for c in kept])


SYNOPSIS_COLUMNS = ["Synopsis", "synopsis_anilist"]


def split_metadata(src_path, dst):
    """anime_metadata.csv without the synopsis text, and anime_synopses.csv with it."""
    with open(src_path, newline="", encoding="utf-8") as f, \
            open(dst / "anime_metadata.csv", "w", newline="", encoding="utf-8") as meta_out, \
            open(dst / "anime_synopses.csv", "w", newline="", encoding="utf-8") as synopsis_out:
        reader = csv.DictReader(f)
        meta_columns = [c for c in reader.fieldnames if c not in SYNOPSIS_COLUMNS]
        synopsis_columns = ["anime_ID", "Synopsis"] + [c for c in ("synopsis_source",) if c in reader.fieldnames]
        meta_writer = csv.writer(meta_out, lineterminator="\n")
        synopsis_writer = csv.writer(synopsis_out, lineterminator="\n")
        meta_writer.writerow(meta_columns)
        synopsis_writer.writerow(synopsis_columns)
        for row in reader:
            meta_writer.writerow([row[c] for c in meta_columns])
            synopsis_writer.writerow([row[c] for c in synopsis_columns])


def main(src, dst):
    with open(src / "users.csv", newline="", encoding="utf-8") as f:
        users = list(csv.DictReader(f))
    seed_ids = {u["anilist_user_id"] for u in users if u["source"] == "seed" and u["anilist_user_id"]}
    user_map, n_new = load_or_extend_map(PRIVATE / f"{src.name}_user_id_map.csv", seed_ids)

    # Manual users keep their username; everyone else gets their random ID
    manual_names = {u["anilist_user_id"]: u["username"] for u in users if u["source"] == "manual" and u["anilist_user_id"]}

    def id_for(row):
        uid = row["anilist_user_id"]
        if row["source"] == "manual":
            return manual_names.get(uid) or row.get("username") or row.get("user")
        return user_map[uid] if uid else "anon-unknown"

    dst.mkdir(parents=True, exist_ok=True)
    identifying = {"anilist_user_id", "username", "user"}
    dates_and_titles = {"anime_title", "started_at", "completed_at", "updated_at", "created_at"}

    # mtime=0 keeps the gzip output byte-identical across runs
    with gzip.open(src / "ratings.csv.gz", "rt", newline="", encoding="utf-8") as f, \
            open(dst / "ratings.csv.gz", "wb") as raw_file, \
            gzip.GzipFile(fileobj=raw_file, mode="wb", mtime=0) as gz, \
            io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
        anonymise_csv(f, text, id_for, "user_id", identifying | dates_and_titles)

    with open(src / "user_profiles.csv", newline="", encoding="utf-8") as f, \
            open(dst / "user_profiles.csv", "w", newline="", encoding="utf-8") as out:
        anonymise_csv(f, out, id_for, "user", identifying)

    with open(src / "users.csv", newline="", encoding="utf-8") as f, \
            open(dst / "users.csv", "w", newline="", encoding="utf-8") as out:
        anonymise_csv(f, out, id_for, "user_id", identifying | {"error"})

    split_metadata(src / "anime_metadata.csv", dst)

    n_manual = sum(u["source"] == "manual" for u in users)
    print(f"{len(seed_ids)} seed users anonymised ({n_new} new IDs), {n_manual} manual users kept by name, into {dst}")


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW / "anilist"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(f"{src.name}_anonymised")
    main(src, dst)
