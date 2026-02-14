import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Spotify Recommendation App")

st.title("🎵 Spotify Recommendation System")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("spotify.xls")
    return df

df = load_data()

st.write("Dataset Preview")
st.dataframe(df.head())

# Prepare data
users = df.iloc[:, 0]     # user column
features = df.iloc[:, 1:] # song features

# Train model
model = NearestNeighbors(metric="cosine", algorithm="brute")
model.fit(features)

# User selection
selected_user = st.selectbox("Select a User", users)

if st.button("Recommend Similar Users"):
    user_index = users[users == selected_user].index[0]
    distances, indices = model.kneighbors(
        [features.iloc[user_index]], 
        n_neighbors=5
    )

    st.subheader("Recommended Similar Users:")
    for i in indices[0][1:]:
        st.write(users.iloc[i])
