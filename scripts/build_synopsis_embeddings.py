"""
Encode anime synopses with sentence-transformers (all-MiniLM-L6-v2) and save them to
artifacts/synopsis_embeddings.npy, with the matching anime IDs in artifacts/synopsis_embedding_ids.npy.

The synopsis text comes from data/raw/anilist_anonymised/anime_synopses.csv, which is gitignored (it is
mostly MyAnimeList's copyrighted text) and written by scripts/anonymise_users.py. Only the embeddings are
committed, so a fresh clone can run the analysis without the text.

This runs as its own process because on Windows pip's torch and conda's MKL cannot both be
loaded into one Python process, which crashes the notebook kernel.

Usage:
    python scripts/build_synopsis_embeddings.py
"""
from sentence_transformers import SentenceTransformer  # import torch before anything loads MKL

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SYNOPSES_PATH = ROOT / "data" / "raw" / "anilist_anonymised" / "anime_synopses.csv"
OUT_PATH = ROOT / "artifacts" / "synopsis_embeddings.npy"
IDS_PATH = ROOT / "artifacts" / "synopsis_embedding_ids.npy"


def main():
    if not SYNOPSES_PATH.exists():
        sys.exit(f"{SYNOPSES_PATH} not found. The synopsis text isn't in the repo: collect the data with "
                 "anilist_ingestion.ipynb and run scripts/anonymise_users.py first.")
    synopses = pd.read_csv(SYNOPSES_PATH).drop_duplicates(subset="anime_ID", keep="first")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(synopses["Synopsis"].fillna("").tolist(), show_progress_bar=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH, embeddings)
    np.save(IDS_PATH, synopses["anime_ID"].to_numpy())
    print(f"{embeddings.shape[0]} synopsis embeddings saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
