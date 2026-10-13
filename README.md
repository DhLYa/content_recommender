# Anime Recommender

User-based collaborative filtering, content-based filtering, a weighted hybrid and Surprise SVD on MyAnimeList ratings, all scored against simple baselines on the same held-out test set.

MyAnimeList's built-in recommendations are based on popularity and on recent user-written recommendations, not on what an individual user has rated. This project tests how much personalised models improve on that, using ratings collected from MAL.

Originally built as a university data analytics coursework project. The code was reviewed and fixed afterwards (see [Changes](#changes-from-the-original-coursework-notebook)), so the results here differ from the original coursework report, which is not included.

## Data

**Collection.** The MAL API returned the 500 most popular anime. Their stats pages were scraped with Beautiful Soup for usernames, keeping only users who had scored the title. Collection was capped at 10,000 users by API limits. Each user's scored anime list was then fetched through the API, giving a 5,610 × 20,896 user-by-anime matrix. Metadata for each anime came from the API, and themes were scraped from each anime's page.

**Population check.** The Kaggle [Anime Dataset 2023](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset) user ratings (313,670 users once MAL's `0` = "not scored" entries are removed) are used to check that the sample is representative. The rating distributions are close: the mean rating is 7.38 in the sample and 7.43 in the population, and the median is 8 in both.

**Preprocessing.**
- Usernames are replaced with random user IDs by `scripts/anonymise_users.py`. The original files with real usernames are not in this repo.
- Duplicate users are removed: 5,610 rows become 5,034 users, 4,915 of whom have at least one rating.
- The wide matrix is melted into ~1.39M `(user_id, anime_title, rating)` rows.
- Users with fewer than 85 ratings and anime with fewer than 25 are dropped. This leaves 3,506 users, 5,535 anime and 1.26M ratings.
- Users whose mean rating is above 9 are removed, since many users rate everything 10 as a checklist. This leaves **3,370 users** for modelling.
- Sparsity falls from 98.68% to 93.45%.
- 72% of the remaining anime (3,977 of 5,535) have metadata. The content-based model falls back to the baseline for the rest.

## Results

All models are scored on the same test set: 20% of each user's ratings. k and the hybrid weight were tuned on a validation split carved out of the training data.

| Model | MAE | RMSE | Coverage |
|---|---|---|---|
| SVD (tuned: 150 factors, lr 0.005, reg 0.1) | 0.871 | 1.176 | 100% |
| Weighted hybrid (0.8 CF / 0.2 content) | 0.889 | 1.200 | 100% |
| User-based CF (k = 50) | 0.904 | 1.218 | 100% |
| *Baseline: user mean + item bias* | 0.948 | 1.272 | 100% |
| Content-based (k = 20) | 1.027 | 1.372 | 91.8% |
| *Baseline: user mean* | 1.113 | 1.469 | 100% |
| *Baseline: global mean* | 1.296 | 1.678 | 100% |

Coverage is the share of test ratings a model predicted itself rather than handing to the baseline.

The hybrid beats both of its components, but most of its accuracy comes from CF. Content-based filtering does worse than the user mean + item bias baseline, even on the pairs it covers. It adjusts the user's mean using only their own ratings of similar anime and ignores how each anime is rated overall. Basing both kNN models on the user + item baseline (Koren's kNN-baseline) is the obvious next step.

## Layout

```
content_recommender/
├── anime_recommender.ipynb              # EDA, preprocessing, models, evaluation
├── environment.yml                      # conda environment
├── data/
│   ├── raw/
│   │   ├── user_anime_ratings_wide.csv.gz  # one row per MAL user, one column per anime
│   │   ├── anime_metadata.csv           # type, genres, themes, episodes, synopsis per anime
│   │   └── user_profiles.csv            # MAL profile stats (gender, days watched, ...)
│   ├── private/                         # not in git: original files with usernames, username -> ID map
│   ├── external/
│   │   ├── population_rating_counts.csv       # Kaggle "Anime Dataset 2023" rating histogram
│   │   └── population_user_mean_ratings.csv   # ... and per-user mean ratings
│   └── processed/
│       └── ratings_long_filtered.csv    # generated: filtered (user_id, anime_title, rating)
├── artifacts/                           # generated: synopsis embeddings, top-15 similar anime
├── figures/                             # generated: EDA plots
└── scripts/
    ├── anonymise_users.py               # writes data/raw user files from data/private with random IDs
    ├── build_population_summary.py      # rebuilds data/external from the 1.5 GB Kaggle file
    └── build_synopsis_embeddings.py     # encodes synopses into artifacts/ (run by the notebook if missing)
```

## Running

```
conda env create -f environment.yml
conda activate anime-recommender
```

Open `anime_recommender.ipynb` with the `anime-recommender` environment as the kernel and run all cells from this folder. A full run takes a few minutes with the synopsis embeddings cached.

The synopsis embeddings are cached in `artifacts/synopsis_embeddings.npy`. If the file is missing, the notebook runs `scripts/build_synopsis_embeddings.py` to rebuild it, downloading the `all-MiniLM-L6-v2` model on first use. That step runs in a separate process because on Windows, pip's `torch` and conda's MKL crash when loaded into the same Python process. Don't import `torch` in the notebook kernel.

## Changes from the original coursework notebook

**Collaborative filtering**
- The similarity is computed over anime both users rated. The original filled unrated anime with 0 and kept only anime both users rated above their own mean, so every similarity was ≥ 0 and many were exactly 1 from one or two shared anime.
- Similarities are shrunk when users share few anime, and pairs sharing fewer than 5 anime are ignored.
- Neighbours are picked per anime from the users who rated it. Before, they were the top-n users overall, so about 90% of predictions silently fell back to the user's mean.

**Content-based**
- Predictions are on the 1–10 rating scale: the user's mean plus the similarity-weighted average of their mean-centred ratings of the most similar anime they rated. Before, the score was the average of similarity × rating.
- Pairs that cannot be predicted fall back to the baseline instead of being dropped, and coverage is reported.

**Hybrid**
- It blends two rating-scale predictions for every anime. Before, it combined scores on different scales, so the CF list always won.
- Evaluation uses training-only CF similarities. Before, the similarities were built from all ratings, including the test ratings.

**Evaluation**
- All models are scored on one shared train/test split.
- Hyperparameters (k, hybrid weight) are tuned on a validation split taken from the training data instead of on the test set.
- Global-mean, user-mean and user-mean + item-bias baselines are added.

**Smaller fixes**
- The population comparison excludes MAL's `0` ("not scored") ratings.
- A data path that no longer existed is fixed, and a dead `StandardScaler` is removed.
- An empty-string genre/theme label is no longer created for missing values.
- `eval` on stringified dicts is replaced by a NumPy similarity matrix, and duplicate metadata titles are de-duplicated.
