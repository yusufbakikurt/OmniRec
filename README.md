# OmniRec — Movie & Book Recommendation System

A Netflix-style recommender that suggests **movies** and **books** in one app, built with Python, scikit-learn and Streamlit.

- **Movies → content-based filtering:** finds films similar to one you like, using TF-IDF over director, genres, top cast, keywords and plot overview, compared with cosine similarity.
- **Books → collaborative filtering:** predicts ratings from what similar readers liked, using matrix factorization (Truncated SVD) on a mean-centered user–item rating matrix.
- **Cold start:** new users get popular titles filtered by genre (IMDb-style weighted rating for movies).

> Team project by [Aysu Çaluk](https://github.com/Aysu61) and [Yusuf Baki Kurt](https://github.com/yusufbakikurt). Forked from [Aysu61/OMN-REC](https://github.com/Aysu61/OMN-REC).

## App screens

| Screen | What it does |
| --- | --- |
| Discover (new user) | Pick movies and/or books and a genre, get the most popular titles |
| Smart recommender (existing user) | Search a movie title for similar films, or enter a user ID for personalised book picks |
| System performance (admin) | Model comparison and evaluation metrics |

## How it works

### Movies (TMDB 5000)
1. Parse genres, keywords, top-3 cast and director from the raw JSON-like columns.
2. Build a weighted "soup" of text per film: director ×3, genres ×2, cast ×2, keywords, overview.
3. Vectorise with `TfidfVectorizer` (English stop words, 5,000 features).
4. Recommend the films with the highest cosine similarity to the selected one.

### Books (Book-Crossing)
1. Keep explicit ratings (1–10), then iteratively keep only users and books with at least 5 ratings; derive a genre from the book title.
2. Build the user × book rating matrix and subtract each user's mean rating.
3. Factorize with `TruncatedSVD`, reconstruct the matrix to predict missing ratings.
4. Recommend the highest-predicted books the user has not read yet.

### Evaluation
`model_analiz.py` splits ratings 80/20 and compares **SVD, NMF and user-based CF** with RMSE, MAE, Precision@20 and Recall@20. It also searches the number of SVD components for the best F1 score:

![SVD components vs. Precision/Recall/F1](data/svd_sweet_spot.png)

## Project structure

```
OmniRec/
├── data/               # CSV datasets go here (not in the repo) + charts
├── veri_hazirlama.py   # 1) data cleaning + feature engineering → *_cleaned.csv
├── model_analiz.py     # 2) model comparison and metrics (optional)
├── search_engine.py    # recommendation engine (FastSearchEngine class)
├── app.py              # 3) Streamlit interface
└── requirements.txt
```

## Run it locally

1. Download the datasets into `data/`:
   - [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) → `tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`
   - [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) → `Books.csv`, `Ratings.csv`
2. Install and run:
   ```bash
   python -m venv venv
   venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
   pip install -r requirements.txt
   python veri_hazirlama.py       # creates the cleaned CSVs (once)
   streamlit run app.py           # opens http://localhost:8501
   ```

## Tech stack

Python · pandas · NumPy · scikit-learn · Streamlit · Matplotlib · Seaborn
