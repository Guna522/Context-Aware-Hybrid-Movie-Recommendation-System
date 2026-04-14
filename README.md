# Context-Aware Hybrid Movie Recommendation System

A machine learning-based movie recommendation system that combines collaborative filtering and supervised learning to deliver personalized recommendations.

---

## Features
- Hybrid recommendation model (SVD + XGBoost)
- Personalized top-N movie recommendations
- Real-time predictions using Streamlit
- Movie poster integration using TMDB API

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- Surprise (SVD)
- Streamlit

---

## Dataset
- MovieLens 100K dataset
- 100,000+ user–movie interactions
- 900+ movies

---

## How It Works
1. SVD predicts user–movie ratings
2. Features are engineered (genre, ratings, etc.)
3. XGBoost refines predictions
4. Top-N recommendations are generated

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
