"""
Encode anime synopses with sentence-transformers (all-MiniLM-L6-v2) and save them to
artifacts/synopsis_embeddings.npy, one row per de-duplicated title in data/raw/anime_metadata.csv.

This runs as its own process because on Windows pip's torch and conda's MKL cannot both be
loaded into one Python process, which crashes the notebook kernel.

Usage:
    python scripts/build_synopsis_embeddings.py
"""
from sentence_transformers import SentenceTransformer  # import torch before anything loads MKL

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "artifacts" / "synopsis_embeddings.npy"


def main():
    # Same de-duplication as the notebook so rows line up
    anime_df = pd.read_csv(ROOT / "data" / "raw" / "anime_metadata.csv").drop_duplicates(subset="Title", keep="first")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(anime_df["Synopsis"].fillna("").tolist(), show_progress_bar=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH, embeddings)
    print(f"{embeddings.shape[0]} synopsis embeddings saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
