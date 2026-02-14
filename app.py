import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Spotify Dashboard", layout="wide")

st.title("🎵 Spotify User Dashboard")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("spotify.xls")
    return df

df = load_data()

# Rename user column
df.rename(columns={"Unnamed: 0": "User"}, inplace=True)

# Sidebar
st.sidebar.header("Navigation")
option = st.sidebar.radio(
    "Select Section",
    ["Dataset Overview", "User Analysis", "Recommendation System"]
)

# SECTION 1: DATASET OVERVIEW
if option == "Dataset Overview":
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

# SECTION 2: USER ANALYSIS
elif option == "User Analysis":
    st.subheader("User Listening Summary")

    selected_user = st.selectbox("Select User", df["User"])

    user_data = df[df["User"] == selected_user].iloc[:, 1:]

    top_songs = user_data.T.sort_values(by=user_data.index[0], ascending=False).head(10)

    fig = plt.figure()
    top_songs.plot(kind="bar")
    plt.title("Top 10 Songs Played")
    plt.ylabel("Play Count")
    st.pyplot(fig)

# SECTION 3: RECOMMENDATION SYSTEM
elif option == "Recommendation System":
    st.subheader("Find Similar Users")

    users = df["User"]
    features = df.iloc[:, 1:]

    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(features)

    selected_user = st.selectbox("Choose User", users)

    if st.button("Recommend"):
        user_index = users[users == selected_user].index[0]
        distances, indices = model.kneighbors(
            [features.iloc[user_index]],
            n_neighbors=6
        )

        st.write("### Similar Users")
        for i in indices[0][1:]:
            st.write(users.iloc[i])
