import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Page Config
st.set_page_config(page_title="Spotify Analytics Dashboard", layout="wide")

st.title("🎵 Spotify Analytics Dashboard")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("spotify.xls")
    df.rename(columns={"Unnamed: 0": "User"}, inplace=True)
    return df

df = load_data()

# =========================
# KPI CARDS
# =========================
total_users = df.shape[0]
total_songs = df.shape[1] - 1
total_streams = df.iloc[:, 1:].sum().sum()

col1, col2, col3 = st.columns(3)

col1.metric("Total Users", total_users)
col2.metric("Total Songs", total_songs)
col3.metric("Total Streams", int(total_streams))

st.divider()

# =========================
# TOP LISTENERS RANKING
# =========================
st.subheader("🏆 Top Listeners")

df["Total Plays"] = df.iloc[:, 1:].sum(axis=1)
top_users = df.sort_values("Total Plays", ascending=False).head(10)

st.dataframe(top_users[["User", "Total Plays"]])

st.divider()

# =========================
# INTERACTIVE CHART
# =========================
st.subheader("📊 User Listening Chart")

selected_user = st.selectbox("Select User", df["User"])

user_data = df[df["User"] == selected_user].iloc[:, 1:-1].T
user_data.columns = ["Play Count"]
user_data = user_data.sort_values("Play Count", ascending=False).head(10)

fig = plt.figure()
user_data.plot(kind="bar")
plt.title("Top 10 Songs Played")
plt.ylabel("Play Count")
st.pyplot(fig)

st.divider()

# =========================
# RECOMMENDATION SYSTEM
# =========================
st.subheader("🤝 Similar User Recommendation")

users = df["User"]
features = df.iloc[:, 1:-1]

model = NearestNeighbors(metric="cosine", algorithm="brute")
model.fit(features)

selected_user_rec = st.selectbox("Choose User for Recommendation", users)

if st.button("Recommend Similar Users"):
    user_index = users[users == selected_user_rec].index[0]
    distances, indices = model.kneighbors(
        [features.iloc[user_index]], 
        n_neighbors=6
    )

    st.write("### Similar Users:")
    for i in indices[0][1:]:
        st.write(users.iloc[i])
