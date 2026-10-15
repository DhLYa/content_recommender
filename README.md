# Anime Recommender

User-based collaborative filtering, content-based filtering, a weighted hybrid and Surprise SVD on [AniList](https://anilist.co) ratings, scored against simple baselines on the same held-out test set, with recommendations from each model.

## Background

This started as university coursework built on a different dataset from MyAnimeList (MAL). A later review of the code found problems in the original methods, so the dataset and methods were rewritten and updated.

The project now uses AniList data only. The AniList data is collected by [`anilist_ingestion.ipynb`](anilist_ingestion.ipynb), so the dataset can be rebuilt from scratch.

## Issues in the original

- **CF similarity was wrong.** Unrated anime were filled with 0 and masked with `> 0` *after* mean-centring, so only anime both users rated *above their own mean* counted.
- **CF neighbours didn't have to have rated the anime.** They were the top-n most similar users overall, so ~90% of predictions silently fell back to the user's mean. Now neighbours are picked per anime from the users who rated it.
- **The hybrid mixed rating scales.** CF predicted on 1–10 while content-based used `mean(similarity × rating)`, so CF always won and the "hybrid" was just the CF list. Now both components predict ratings and are blended.
- **The hybrid evaluation leaked test data** through similarities computed from all ratings. Now evaluation uses training ratings only.
- **Scores of 0 (signifying watched, not scored) counted as a rating**, which lowered the population mean to 4.25 and wrongly suggested the sample waws non-representative.
## Data

### Collection

[`anilist_ingestion.ipynb`](anilist_ingestion.ipynb) collects from the AniList GraphQL API in stages:

1. **Seed anime**: the top 500 anime by score.
2. **Users**: for each seed anime, up to 50 users who scored it, until `MAX_USERS` unique users are found.
3. **Manual users**: specific users added by name (people of interest), always fetched and marked `source = "manual"`.
4. **User lists**: every entry on each user's anime list.
5. **Anime metadata**: the `N_TOP_COMMON` anime rated most often in the collection (10,500 in the current data) plus the top 100 seed anime, keeping the MAL ID for cross-referencing purposes.
6. **MAL synopses**: AniList's descriptions are often short, so each anime's synopsis is looked up on MyAnimeList by its MAL ID. This uses the official MAL API when a `MAL_CLIENT_ID` environment variable is set (a free client ID from [myanimelist.net/apiconfig](https://myanimelist.net/apiconfig)), and otherwise [Jikan](https://jikan.moe), which needs no key.

Requests are throttled to AniList's current limit (30/min) with retries and backoff. Every stage checkpoints to disk, so a collection can run over several sessions and be interrupted safely, and each stage is validated before the next.

The current dataset has:
- 10,001 Total Users
    - 9,483 Total Users
    - 518 Private Lists (unusable for ranking)
- 20,700 retrieved anime
- 7.49M list entries
- 4.56M ratings


### Transformations

**Ratings** (`ratings.csv.gz`, one row per user and anime)
- Scores are fetched on AniList's 1–100 scale, and are rescaled to `rating` = 0.1–10. A score of 0 means unrated: `is_rated` is false and `rating` is empty. The analysis uses rated entries only.
- Custom lists are skipped, since they repeat entries from the status lists. Duplicate entries for the same anime are merged, keeping the rated entry, then the most recently updated.
- Each entry keeps its `status` (completed, planning, dropped, …), `progress` (episodes watched) and `repeat` (rewatch count).

**User profiles** (`user_profiles.csv`) are computed from each user's list: status counts, total entries, rewatches, episodes watched, mean rating, and days watched (from AniList's own statistic).

**Anime metadata** (`anime_metadata.csv`)
- Title(s), format, episodes, season, genres, studios, scores and the MAL ID.
- **Themes**: AniList has no separate themes, so these are its tags with a relevance of at least 60/100, excluding spoiler tags. The full tag list is kept in `tags`.
- **Synopsis**: MyAnimeList's where there is one (14,353 anime), otherwise AniList's description. HTML and credit lines such as `[Written by MAL Rewrite]` or `(Source: …)` are removed.
- `score_distribution` holds AniList-wide score counts, used to compare the sample with AniList as a whole.

<!-- **The committed copy** drops list dates and anime titles from the ratings file to stay under GitHub's 100 MB limit (~38 MB instead of ~200 MB). It also keeps synopsis text out of git: only the embeddings built from it are committed. -->

### Sampling bias

The sample is not representative of AniList users:

- **Users come from top-rated anime**, so they lean towards active, engaged users with long lists (median 343 ratings).
- **Early accounts dominate.** For a given anime, AniList's API only returns the first 5,000 list entries, oldest first, and ignores sorting and date filters on that query. Users are taken from random pages within that window, but it can't reach past it. The median sampled account ID is about 236,000, and 63% are below 500,000, out of roughly 7.9 million accounts.
- **The sample rates lower than AniList as a whole**: on the same anime, a mean of 7.1 against 7.6 (median 7 vs 8), with a similar spread (`figures/sample_vs_anilist_ratings.png`).
- **Private lists are missing** (5% of users found), and only users who scored a seed anime are taken.
- **Manual users are hand-picked**, so they're excluded from the sample statistics. They are kept in the models and exempt from the user filters, so they always get recommendations.

### Privacy

Usernames are replaced with random identifiers (`anon-00001`, …) by [`scripts/anonymise_users.py`](scripts/anonymise_users.py). Manual users keep their AniList username.

## Method

- **Filtering**:
  - users with fewer than 85 ratings and anime with fewer than 25 are dropped;
  - users whose mean rating is above 9 are removed, since rating nearly everything 9–10 says little about taste;
  - manual users are exempt from both.

  This leaves **8,276 users, 10,704 anime and 4.35M ratings**, and sparsity falls from 97.7% to 95.1%.
- **Evaluation**: every model is scored on one shared test set (20% of each user's ratings). Neighbourhood sizes, the hybrid weight and SVD's settings are tuned on a validation split carved out of the training data, so the test set is only used for final scores.
- **Baselines**: global mean, user mean, and user mean + item bias. The item bias is how far above their own mean users rate an anime on average, shrunk towards 0 for rarely rated anime.
- **kNN models build on that baseline** (the kNN-baseline approach): a prediction is the baseline plus the neighbours' average deviation from *their* baselines, so neighbours only explain what the baseline doesn't.
  - **User-based CF**: neighbours are the most similar users who rated the anime. Each neighbour's deviation is scaled by their own rating spread, since AniList users score on different scales.
  - **Content-based**: neighbours are the most similar anime the user has rated.
- **Content features** (weighted, then compared by cosine similarity):
  - episode count (log-scaled and standardised), weight 1;
  - format, weight 0.25;
  - genres, weight 2.5;
  - themes, weight 1.75;
  - a 384-dimension `all-MiniLM-L6-v2` synopsis embedding, weight 3.
- **Hybrid**: a weighted average of the CF and content-based predictions for every anime, with the weight tuned on validation.
- **SVD** (Surprise, which includes user and item bias terms): 36 configurations are trained on the tuning split and scored on validation, and the best is retrained on the full training set.

## Results

| Model | MAE | RMSE | Coverage |
|---|---|---|---|
| SVD (tuned: 200 factors, 40 epochs, lr 0.005, reg 0.05) | 0.845 | 1.164 | 100% |
| Weighted hybrid (0.7 CF / 0.3 content) | 0.873 | 1.200 | 100% |
| SVD (default: 100 factors, 20 epochs, lr 0.005, reg 0.02) | 0.872 | 1.203 | 100% |
| User-based CF (k = 50) | 0.880 | 1.211 | 100% |
| Content-based (k = 20) | 0.913 | 1.249 | 100% |
| *Baseline: user mean + item bias* | 0.997 | 1.347 | 100% |
| *Baseline: user mean* | 1.194 | 1.585 | 100% |
| *Baseline: global mean* | 1.414 | 1.859 | 100% |

Ratings are on a 1–10 scale. Coverage is the share of test ratings a model predicted itself rather than handing to the baseline.

- **SVD is the most accurate model**
- **The hybrid beats both of its components**, though most of its accuracy comes from CF (weight 0.7).
- **Every model beats the strongest baseline.** For the kNN models, that comes from building on the user + item baseline: content-based filtering, which only sees a user's own ratings, would otherwise ignore how each anime is rated overall.

## Layout

```
anime-recommender/
├── anilist_ingestion.ipynb        # collects the AniList data (resumable, rate-limited)
├── anime_recommender.ipynb        # EDA, models, evaluation, recommendations
├── environment.yml                # conda environment
├── data/
│   ├── raw/
│   │   ├── anilist/               # not in git: raw collection with usernames + checkpoints
│   │   └── anilist_anonymised/
│   │       ├── ratings.csv.gz     # one row per list entry
│   │       ├── anime_metadata.csv # one row per anime (no synopsis text)
│   │       ├── anime_synopses.csv # not in git: synopsis text
│   │       ├── user_profiles.csv  # per-user list statistics
│   │       └── users.csv          # every user found, with source and status
│   ├── private/                   # not in git: AniList ID -> anonymised ID map
│   └── processed/                 # not in git: filtered ratings (generated)
├── artifacts/                     # synopsis embeddings + their anime IDs, top-15 similar anime
├── figures/                       # EDA plots (generated)
└── scripts/
    ├── anonymise_users.py         # anonymised copy of the collection (manual users keep their names)
    └── build_synopsis_embeddings.py
```

## Running

```
conda env create -f environment.yml
conda activate anime-recommender
```

1. **Collect** (optional, since the anonymised data and embeddings are included): open `anilist_ingestion.ipynb`, set `TEST_MODE = False` and the sample size in the config cell, and run all cells. Rerunning continues where it stopped. To use the official MAL API for synopses, set `MAL_CLIENT_ID` in the environment before starting Jupyter.
2. **Anonymise**: `python scripts/anonymise_users.py`
3. **Analyse**: run `anime_recommender.ipynb` from the repo root with the `anime-recommender` environment as the kernel.

The synopsis embeddings are cached in `artifacts/synopsis_embeddings.npy`, with their anime IDs in `artifacts/synopsis_embedding_ids.npy`. If either file is missing, the notebook rebuilds them from `anime_synopses.csv` with `scripts/build_synopsis_embeddings.py`, downloading the `all-MiniLM-L6-v2` model on first use. Delete `synopsis_embeddings.npy` after the synopses change, since changed text isn't detected automatically.

The embeddings are built in a separate process because on Windows, pip's `torch` and conda's MKL crash when loaded into the same Python process, so don't import `torch` in the notebook kernel.

### Run times

Measured on the 10,000-user collection on Windows desktop using a Ryzen 9800x3d CPU:

| Step | Time | Notes |
|---|---|---|
| Seed anime + user discovery | ~1 h | 282 seed anime searched, 1–3 requests each |
| User lists | ~6.4 h | One request per user, ~2.3 s each |
| Anime metadata | ~15 min | 50 anime per request |
| MAL synopses | ~3.5 h | ~10,500 lookups at 50/min |
| Assembly, anonymisation, embeddings | ~5 min | A rerun of a finished collection makes no requests |
| Analysis notebook | ~7 min | About 5 min of it is the SVD grid search |
| **Total from scratch** | **~13 h** | About 45 minutes per 1,000 users, plus the MAL synopses |

Keep AniList at `REQUESTS_PER_MINUTE = 30`. A run at 45 triggered hundreds of rate-limit responses and ended up slower overall (~26 requests/min).
