import streamlit as st
import pandas as pd
import pickle
import os
import requests

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide")
st.title("Movie Recommender")

st.markdown(" ")

API_KEY = "3322cd254e599cf7a4ccf5207465a101"

# ---------------- LOAD ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'final_df.csv'), low_memory=False)
movies = pd.read_csv(os.path.join(BASE_DIR, 'data', 'movies.csv'), low_memory=False)
links = pd.read_csv(os.path.join(BASE_DIR, 'data', 'links.csv'))

with open(os.path.join(BASE_DIR, 'models', 'svd_model.pkl'), 'rb') as f:
    svd_model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'models', 'hybrid_model.pkl'), 'rb') as f:
    hybrid_model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'models', 'features.pkl'), 'rb') as f:
    features = pickle.load(f)

# ---------------- TMDB MAP ----------------
movie_tmdb_map = links.set_index('movieId')['tmdbId'].to_dict()

@st.cache_data
def get_poster(movie_id):
    try:
        tmdb_id = movie_tmdb_map.get(movie_id)

        if pd.isna(tmdb_id):
            return "https://i.imgur.com/6M513jN.png"

        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={API_KEY}"
        res = requests.get(url).json()

        if res.get("poster_path"):
            return f"https://image.tmdb.org/t/p/w500{res['poster_path']}"
    except:
        pass

    return "https://i.imgur.com/6M513jN.png"

# ---------------- USER NAMES ----------------
unique_users = sorted(df['userId'].unique())

names = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun",
    "Sai", "Reyansh", "Krishna", "Ishaan", "Shaurya",
    "Ananya", "Diya", "Aadhya", "Kiara", "Ira",
    "Saanvi", "Riya", "Priya", "Meera", "Anika"
]

user_names = {
    uid: names[i % len(names)] + f" ({uid})"
    for i, uid in enumerate(unique_users)
}

# ---------------- HEADER ----------------
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("User")

    selected_user = st.selectbox(
        "Select User",
        list(user_names.values()),
        label_visibility="collapsed"
    )

    user_id = [uid for uid, name in user_names.items() if name == selected_user][0]
    user_name = selected_user.split(" (")[0]

with col2:
    st.write("")
    st.caption(f"Signed in as {user_name}")

st.markdown(" ")
st.divider()

# ---------------- FILTERS ----------------
st.sidebar.header("Filters")
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.0)
min_popularity = st.sidebar.slider("Minimum Popularity", 0, 200, 20)
num_recs = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

# ---------------- HISTORY ----------------
st.subheader("Previously Watched")
st.markdown(" ")

user_history_ids = df[df['userId'] == user_id]['movieId'].unique()

num_cols = 5
for i in range(0, min(len(user_history_ids), 10), num_cols):
    cols = st.columns(num_cols, gap="medium")

    for col, movie_id in zip(cols, user_history_ids[i:i+num_cols]):
        title = movies[movies['movieId'] == movie_id]['title'].values[0]
        poster = get_poster(movie_id)

        with col:
            st.image(poster, width='stretch')
            st.caption(title[:40])

# ---------------- MODEL ----------------
def hybrid_recommend(user_id):

    candidate_df = df[
        (df['avg_rating'] >= min_rating) &
        (df['num_ratings'] >= min_popularity)
    ]

    candidate_movies = candidate_df['movieId'].unique()

    movie_lookup = df.drop_duplicates('movieId').set_index('movieId')
    watched = set(df[df['userId'] == user_id]['movieId'])

    preds = []

    for movie_id in candidate_movies:

        if movie_id in watched:
            continue

        try:
            svd_pred = svd_model.predict(user_id, movie_id).est

            row = movie_lookup.loc[movie_id].copy()
            row['svd_pred'] = svd_pred

            input_data = pd.DataFrame([row])

            for col in features:
                if col not in input_data.columns:
                    input_data[col] = 0

            input_data = input_data[features]

            pred = hybrid_model.predict(input_data)[0]

            preds.append((movie_id, pred))

        except:
            continue

    preds.sort(key=lambda x: x[1], reverse=True)
    return preds[:num_recs]

# ---------------- BUTTON ----------------
if st.button("Get Recommendations"):

    with st.spinner("Finding recommendations..."):
        recs = hybrid_recommend(user_id)

    if not recs:
        st.error("No recommendations found.")
    else:
        st.subheader("Recommended For You")
        st.markdown(" ")

        num_cols = 5

        for i in range(0, len(recs), num_cols):
            cols = st.columns(num_cols, gap="medium")

            for col, (movie_id, score) in zip(cols, recs[i:i+num_cols]):
                title = movies[movies['movieId'] == movie_id]['title'].values[0]
                poster = get_poster(movie_id)

                with col:
                    st.image(poster, width='stretch')
                    st.caption(title[:40])
                    st.caption(f"{score:.1f}")

        # ---------------- EXPLANATION ----------------
        st.subheader("Why these recommendations")
        st.markdown(" ")

        if len(user_history_ids) > 0:
            sample_movies = movies[
                movies['movieId'].isin(user_history_ids[:3])
            ]['title'].tolist()

            st.caption(", ".join(sample_movies))