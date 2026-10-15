# Anime Recommender

User-based collaborative filtering, content-based filtering, a weighted hybrid and Surprise SVD on [AniList](https://anilist.co) ratings, all scored against simple baselines on the same held-out test set.

## Background

This started as a university data analytics coursework project built on a different dataset: ratings scraped from MyAnimeList (MAL). A later review of the code found problems in the original methods, some of them fundamental enough to change the results and the conclusions drawn from them, so the methods were rewritten (see [Issues in the original](#issues-in-the-original)).

The project now uses AniList data only. Unlike the MAL data, which was collected with code outside this repo, the AniList data is collected by a notebook in this repo ([`anilist_ingestion.ipynb`](anilist_ingestion.ipynb)), so the dataset can be rebuilt from scratch. The original coursework report and the MAL data are not included.

## Issues in the original

Problems found in the coursework version, and how they were fixed:

- **Collaborative filtering similarity was wrong.** Unrated anime were filled with 0 and the ratings masked with `> 0` *after* mean-centring, so only anime both users rated *above their own mean* counted. Every similarity was ≥ 0 and most users had several "perfect" (1.0) neighbours based on one or two shared anime. Now: mean-centred cosine over all co-rated anime (similarities can be negative), shrunk by `n_shared / (n_shared + 25)`, and ignored below 5 shared anime.
- **CF neighbours didn't have to have rated the anime.** Neighbours were the top-n most similar users overall, so about 90% of predictions silently fell back to the user's mean. Now neighbours are picked per anime from the users who rated it.
- **The hybrid mixed rating scales.** CF scores were on the 1–10 scale but content-based scores were `mean(similarity × rating)`, so CF always won and the "hybrid" output was just the CF list. Now both components predict ratings and are blended for every anime.
- **The hybrid evaluation leaked test data.** It used user similarities computed from all ratings, test ratings included. Now evaluation only uses training ratings.
- **Content-based errors were flattering.** Pairs it couldn't predict were dropped from the evaluation instead of counted. Now they fall back to a baseline and coverage is reported.
- **Hyperparameters were tuned on the test set**, and models were scored on different splits with no baselines. Now every model is scored on one shared test set, neighbourhood sizes and the hybrid weight are tuned on a separate validation split, and three baselines are reported. Against those baselines, the original CF did *worse* than predicting each user's mean rating.
- **The population comparison counted MAL's 0 ("watched, not scored") as a rating**, which put the population mean at 4.25 and led to the wrong conclusion that the sample wasn't representative.
- Smaller fixes: a `StandardScaler` whose output was discarded, a file path that no longer existed, a "top 10" loop that read 9, uncapped predictions, `eval()` on stringified dicts, and an empty-string genre label created for missing values.

## Data

### Collection

[`anilist_ingestion.ipynb`](anilist_ingestion.ipynb) collects the data from the AniList GraphQL API:

1. **Seed anime**: the top 500 anime by score.
2. **Users**: for each seed anime, up to 50 users who have scored it, until `MAX_USERS` unique users are found.
3. **Manual users**: specific users added by name (people of interest). They are always fetched and marked `source = "manual"`.
4. **User lists**: every entry on each user's anime list, with scores on AniList's 1–100 scale (rescaled to 1–10 for the models).
5. **Anime metadata**: the 15,000 anime rated most often in the collection plus the top 100 seed anime. This covers format, episodes, genres, tags, studios, scores and description, and keeps the MAL ID for cross-referencing.
6. **MAL synopses**: AniList's descriptions are often short, so each anime's synopsis is looked up on MyAnimeList by its MAL ID. The source is the official MAL API when a `MAL_CLIENT_ID` environment variable is set (a free client ID from [myanimelist.net/apiconfig](https://myanimelist.net/apiconfig)), otherwise [Jikan](https://jikan.moe), an unofficial MAL API that needs no key. Some VPNs are blocked by Jikan; the stage detects this and stops with a message.

Requests are throttled to AniList's current limit (30/min) with retries and backoff. Every stage checkpoints to disk, so a collection can run over several sessions and be interrupted safely, and each stage is validated before the next.

The current dataset has 10,000 seed users plus 1 manual user. The seed users came from 241 of the 282 highest-rated anime, the ones searched before `MAX_USERS` was reached. 518 seed users had private lists, which leaves **9,483 lists, 7,485,213 list entries and 4,557,327 ratings** across 20,700 anime.

### How the collected data is transformed

**List entries → `ratings.csv.gz`** (one row per user and anime)
- **Scores** are requested on AniList's 1–100 scale (`score`), whatever scale each user rates in, and rescaled to `rating = score / 10` (0.1–10). A score of 0 means *unrated*: `is_rated` is false and `rating` is empty. The analysis only uses rated entries.
- **Custom lists are skipped**, since they repeat entries from the status lists (Watching, Completed, …).
- **Duplicate entries are merged.** AniList occasionally holds two entries for the same anime in one list, mostly on accounts from 2014–2015. The rated entry is kept, then the most recently updated, then the one with more progress.
- **Kept per entry:** `status` (`COMPLETED`, `PLANNING`, `DROPPED`, `CURRENT`, `PAUSED`, `REPEATING`), `progress` (episodes watched) and `repeat` (rewatch count).
- **Dates:** AniList's partial dates become `YYYY-MM-DD` (or `YYYY-MM` / `YYYY` when incomplete), and Unix timestamps become ISO UTC. The committed copy drops these dates and the anime titles (titles are in `anime_metadata.csv`) to stay under GitHub's 100 MB file limit.

**User lists → `user_profiles.csv`** (one row per user, computed from their list)
- Status counts: `Watching` (current + rewatching), `Completed`, `On-Hold` (paused), `Dropped`, `Plan to Watch` (planning), and `Total Entries`.
- `Rewatched` is the sum of rewatch counts, `Episodes` the sum of episodes watched, and `Mean Score` the mean rating of rated entries.
- `Days` comes from AniList's own watch-time statistic, and is empty when AniList has none for the account.

**Anime metadata → `anime_metadata.csv`** (one row per anime)
- **Which anime:** the 15,000 anime rated most often in the collection plus the top 100 seed anime, fetched 50 per request.
- `Title` is the romaji title. English and native titles, the MAL ID, studios (main studios only), source material, duration and airing status are kept as extra columns.
- `Type` is AniList's format in lower case (`tv`, `tv_short`, `movie`, `special`, `ova`, `ona`, `music`).
- **Genres** (`Genre(s)`) come from AniList's fixed list of 19 genres, comma-separated.
- **Themes** (`Theme(s)`): AniList has no separate themes, so these are its tags with a relevance of at least 60/100, excluding spoiler tags, most relevant first. The full tag list with relevance scores is kept in `tags` (`name:rank; …`).
- **Synopsis:** MyAnimeList's synopsis where there is one, with its credit line (`[Written by MAL Rewrite]`, `(Source: …)`) removed. Anime without a MAL synopsis fall back to AniList's description, with HTML removed, entities decoded, and the `(Source: …)` credit cut along with any broadcast notes after it. `synopsis_source` records which was used, and AniList's text is kept in `synopsis_anilist` for comparison. The synopsis text is kept out of git in `anime_synopses.csv`; only the embeddings built from it are committed.
- **Scores:** `Mean Rating` is AniList's mean score / 10. `Rank` and `Popularity` are AniList's all-time rankings, which only exist for top anime, so `Rank` is mostly empty. `score_distribution` holds AniList-wide score counts in 10-point buckets as JSON.
- **Dates:** `Year` and `Season` (e.g. `spring 2023`) come from the release season, and `Start Date` / `End Date` are `dd/mm/yyyy` when the full date is known.

**In the analysis** ([`anime_recommender.ipynb`](anime_recommender.ipynb))
- **Content-based features** for each anime, scaled by a weight before being combined:
  - episode count: log-scaled and standardised, weight 1;
  - `Type`: one-hot encoded, weight 0.25;
  - genres: multi-hot encoded, weight 2.5;
  - themes: multi-hot encoded, weight 1.75;
  - synopsis: a 384-dimension `all-MiniLM-L6-v2` sentence embedding, weight 3.
- Content similarity is the cosine similarity of the weighted features.
- `score_distribution` is summed over the sampled anime to compare the sample's ratings with AniList as a whole.
- Ratings are matched to metadata by AniList anime ID.
- **Both kNN models build on a baseline** (the kNN-baseline approach): a prediction is the user mean + item bias, plus the neighbours' average deviation from *their* baselines. Neighbours then only explain what the baseline doesn't.
- **Per-user normalisation:** in user-based CF, each neighbour's deviation is divided by their own rating spread and multiplied by the target user's, since AniList users score on different scales (all converted to 1–100). In content-based filtering the neighbours are the user's own ratings, so this cancels out.
- **SVD is tuned like the other models:** 36 configurations are trained on the tuning split, scored on the validation split, and the best is retrained on the full training set.

### Sampling bias

The sample is not representative of AniList users:

- **Users come from top-rated anime**, so they lean towards active, engaged users with long lists (median 343 ratings per user).
- **The sample is dominated by early accounts.** For a given anime, AniList's API only returns the first 5,000 list entries, oldest first, and ignores sorting and date filters on that query. Users are taken from random pages within that window, but it can't reach past it. The median sampled account has an AniList ID of about 236,000, and 63% have IDs below 500,000, against roughly 7.9 million accounts in total.
- **Users rate lower than AniList as a whole.** On the same anime, the sample's mean rating is about 7.1 against 7.6 for all AniList users (median 7 vs 8), with a similar spread (`figures/sample_vs_anilist_ratings.png`).
- **The seed anime shape the sample.** Every seed user has scored at least one of the 282 highest-rated anime (e.g. *Sousou no Frieren*, *Gintama*, *Fullmetal Alchemist: Brotherhood*).
- **Private lists are missing** (5% of the users found), and only users who scored the seed anime are taken.
- **Manual users are hand-picked**, so they are excluded from the sample statistics. They are kept in the models and exempt from the user filters, so they always get recommendations.

### Privacy

Usernames are replaced with random identifiers (`anon-00001`, …) by [`scripts/anonymise_users.py`](scripts/anonymise_users.py). Manual users keep their AniList username.

### Preprocessing

- Unscored list entries (planned, dropped or watched without a score) are not ratings and are dropped.
- Users with fewer than 85 ratings and anime with fewer than 25 are dropped (manual users are kept). This leaves 8,503 users, 10,704 anime and 4,427,738 ratings.
- Users whose mean rating is above 9 are removed, since rating nearly everything 9–10 says little about taste. This leaves **8,276 users** for modelling.
- Sparsity falls from 97.67% to 95.09%.
- All 10,704 remaining anime have metadata (the metadata covers the 15,000 anime rated most often).

## Results

All models are scored on the same test set: 20% of each user's ratings. k and the hybrid weight were tuned on a validation split carved out of the training data. Ratings are on a 1–10 scale.

| Model | MAE | RMSE | Coverage | Recall@10 | NDCG@10 |
|---|---|---|---|---|---|
| SVD (tuned: 200 factors, 40 epochs, lr 0.005, reg 0.05) | 0.845 | 1.164 | 100% | 0.024 | 0.121 |
| Weighted hybrid (0.7 CF / 0.3 content) | 0.873 | 1.200 | 100% | 0.020 | 0.109 |
| SVD (default: 100 factors, 20 epochs, lr 0.005, reg 0.02) | 0.872 | 1.203 | 100% | 0.024 | 0.110 |
| User-based CF (k = 50) | 0.880 | 1.211 | 100% | 0.016 | 0.091 |
| Content-based (k = 20) | 0.913 | 1.249 | 100% | 0.018 | 0.096 |
| *Baseline: user mean + item bias* | 0.997 | 1.347 | 100% | 0.016 | 0.070 |
| *Baseline: user mean* | 1.194 | 1.585 | 100% | – | – |
| *Baseline: global mean* | 1.414 | 1.859 | 100% | – | – |
| *Baseline: most popular* | – | – | – | **0.063** | **0.312** |

Coverage is the share of test ratings a model predicted itself rather than handing to the baseline.

**Ranking metrics.** For each of 1,000 sampled users, every anime they haven't rated in training is scored, and the top 10 are compared with their test ratings. An anime counts as relevant if the user rated it at or above their own mean rating. Recall@10 is the share of relevant anime that made the top 10. NDCG@10 also rewards placing them higher. The user and global means give every anime the same score for a user, so they can't rank.

**Rating accuracy.** Building the kNN models on the user + item baseline made the biggest difference. Content-based filtering went from worse than the baseline (RMSE 1.433) to clearly better (1.249), and user-based CF improved from 1.271 to 1.211. SVD tuned on the validation split is now the most accurate model, ahead of its defaults; the earlier cross-validated grid search had picked a worse configuration. The hybrid beats both of its components.

**Ranking.** Recommending the most popular anime wins easily on ranking, with NDCG@10 of 0.31 against 0.12 for the best model. This is a known gap between rating accuracy and top-N recommendation, not a bug:
- The models rank by predicted *rating*, so their top 10 is full of highly rated but less watched anime.
- The test set only contains anime users actually watched, which are mostly popular ones. A good but obscure suggestion the user hasn't seen can never count as a hit.

Ranking on predicted rating weighted by popularity, or training on what users *watch* (every list entry, including unscored ones) rather than how they score it, would close the gap. Recall@10 is low for every model because users have about 59 relevant test anime each, so 10 recommendations can cover at most about 17% of them.

## Layout

```
anime-recommender/
├── anilist_ingestion.ipynb              # collects the AniList data (resumable, rate-limited)
├── anime_recommender.ipynb              # EDA, preprocessing, models, evaluation
├── environment.yml                      # conda environment
├── data/
│   ├── raw/
│   │   ├── anilist/                     # not in git: raw collection with usernames + checkpoints
│   │   └── anilist_anonymised/
│   │       ├── ratings.csv.gz           # one row per list entry (user_id, anime_id, score, rating, status, ...)
│   │       ├── anime_metadata.csv       # format, genres, themes (tags), episodes, scores per anime (no synopsis text)
│   │       ├── anime_synopses.csv       # not in git: synopsis text (MAL, or AniList as fallback)
│   │       ├── user_profiles.csv        # per-user list statistics (status counts, episodes, mean score)
│   │       └── users.csv                # every user found, with source and status
│   ├── private/                         # not in git: AniList ID -> anonymised ID map
│   └── processed/                       # not in git: generated filtered ratings
├── artifacts/                           # synopsis embeddings and their anime IDs (committed), top-15 similar anime
├── figures/                             # generated: EDA plots
└── scripts/
    ├── anonymise_users.py               # anonymised copy of the collection (manual users keep their names)
    └── build_synopsis_embeddings.py     # encodes synopses into artifacts/ (run by the notebook if missing)
```

## Running

```
conda env create -f environment.yml
conda activate anime-recommender
```

1. **Collect** (optional, the anonymised data and the synopsis embeddings are included): open `anilist_ingestion.ipynb`, set `TEST_MODE = False` and the sample size in the config cell, and run all cells. Rerunning continues where it stopped. Allow about 45 minutes per 1,000 users plus the MAL synopses (see [Run times](#run-times)). To use the official MAL API for synopses, set `MAL_CLIENT_ID` in the environment before starting Jupyter.
2. **Anonymise**: `python scripts/anonymise_users.py`
3. **Analyse**: open `anime_recommender.ipynb` with the `anime-recommender` environment as the kernel and run all cells from the repo root. A full run takes a couple of minutes once the synopsis embeddings are cached.

The synopsis embeddings are cached in `artifacts/synopsis_embeddings.npy`, with the anime IDs they belong to in `artifacts/synopsis_embedding_ids.npy`. If either file is missing, the notebook runs `scripts/build_synopsis_embeddings.py` to rebuild them from `anime_synopses.csv`, downloading the `all-MiniLM-L6-v2` model on first use. Delete `synopsis_embeddings.npy` after the synopses change (for example after fetching MAL synopses), since changed text isn't detected automatically. That step runs in a separate process because on Windows, pip's `torch` and conda's MKL crash when loaded into the same Python process. Don't import `torch` in the notebook kernel.

### Run times

Measured on the 10,000-user collection (Windows desktop, CPU only):

| Step | Time | Notes |
|---|---|---|
| Seed anime + user discovery | 64 min | 282 seed anime searched, 1–3 requests each |
| User lists | 6.4 h | One request per user, about 2.3 s each |
| Anime metadata | ~5 min | ~110 requests of 50 anime (measured for 5,000; 15,000 takes about 3× as long) |
| MAL synopses | ~5 h (estimate) | One request per anime with a MAL ID (~15,000) at 50/min |
| Validation and assembly | ~3 min | Also the time for a rerun of a finished collection, which makes no requests |
| Anonymisation | ~1 min | |
| Synopsis embeddings | ~1.5 min | 15,000 synopses; cached afterwards |
| Analysis notebook | ~10 min | SVD grid search ~5 min, ranking metrics ~3 min |
| **Total from scratch** | **~13 h** | Mostly user lists (about 45 minutes per 1,000 users) and MAL synopses |

The collection ran with `REQUESTS_PER_MINUTE = 45`. AniList answered that with 361 rate-limit responses, and the notebook waited out each one. The effective rate came out at about 26 requests/minute, *slower* than the 30/min AniList currently allows, so leave it at 30. At a steady 30/min, user lists take 2 s each, about 5.5 h for 10,000 users.
