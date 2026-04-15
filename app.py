import streamlit as st
import pandas as pd
import numpy as np
import os
import requests

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- LOAD DATA (ROBUST) ----------------
@st.cache_data
def load_data():
    github_url = "https://raw.githubusercontent.com/Guna522/Context-Aware-Hybrid-Movie-Recommendation-System/main/data/final_df.csv"
    local_path = os.path.join(BASE_DIR, "data", "final_df.csv")

    try:
        df = pd.read_csv(github_url)
        return df
    except:
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return pd.read_csv(local_path)
        else:
            st.error("Dataset not found or empty. Please check GitHub or local file.")
            st.stop()

df = load_data()

# ---------------- CLEAN DATA ----------------
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['num_ratings'] = pd.to_numeric(df['num_ratings'], errors='coerce')

df = df.dropna(subset=['title', 'rating'])

# ---------------- USER SYSTEM ----------------
unique_users = sorted(df['userId'].unique())

user_names = {uid: f"User {uid}" for uid in unique_users}

# ---------------- SELECT USER ----------------
st.markdown("### Select User")
selected_name = st.selectbox(
    "Choose a user",
    list(user_names.values()),
    label_visibility="collapsed"
)

user_id = [k for k, v in user_names.items() if v == selected_name][0]

st.markdown(f"#### Welcome, {selected_name}")

# ---------------- POSTER FUNCTION ----------------
def get_poster(title):
    try:
        api_key = "YOUR_TMDB_API_KEY"  # Replace if needed
        url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={title}"
        response = requests.get(url).json()

        if response['results']:
            poster_path = response['results'][0]['poster_path']
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass

    return "https://via.placeholder.com/300x450?text=No+Image"

# ---------------- WATCH HISTORY ----------------
st.markdown("## Previously Watched")

user_history = df[df['userId'] == user_id].sort_values(by='rating', ascending=False).head(10)

cols = st.columns(5)

for i, (_, row) in enumerate(user_history.iterrows()):
    with cols[i % 5]:
        poster = get_poster(row['title'])
        st.image(poster, width='stretch')
        st.caption(row['title'])

# ---------------- RECOMMENDATION FUNCTION ----------------
def recommend_movies(df, user_id, n=10):
    user_movies = df[df['userId'] == user_id]['title'].unique()

    recs = df[~df['title'].isin(user_movies)]

    recs = recs.sort_values(by=['rating', 'num_ratings'], ascending=False)

    return recs.drop_duplicates('title').head(n)

# ---------------- BUTTON ----------------
if st.button("Get Recommendations"):
    st.success("Recommendations ready!")

    recommendations = recommend_movies(df, user_id)

    st.markdown("## Top Picks For You")

    cols = st.columns(5)

    for i, (_, row) in enumerate(recommendations.iterrows()):
        with cols[i % 5]:
            poster = get_poster(row['title'])
            st.image(poster, width='stretch')

            st.markdown(f"**{row['title']}**")
            st.write(f"⭐ {round(row['rating'], 1)}")

            st.caption("Highly rated")
