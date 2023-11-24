import streamlit as st
import pickle
import pandas as pd
import requests
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re

def fetch_poster(movie_id):
    response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=0926738758361a53c79b1db0c2322d98&language=en-US")
    data = response.json()
    return "https://image.tmdb.org/t/p/original/"+data["poster_path"]

def Hybrid_recommender(Movie):
    movie_title = Movie
    movie_index = [index for index, x in enumerate(keywords_taglines_genres["original_title"] == movie_title) if x == True][0]
    top_100_movies = sorted(list(enumerate(cos_results[movie_index])), key=lambda x: x[1], reverse=True)[:100]
    top_movies = pd.DataFrame(np.squeeze(keywords_taglines_genres.values[[np.array(top_100_movies)[:, 0].astype(int)]]), columns=keywords_taglines_genres.columns)
    second_filters = top_movies.merge(datasets, on="id", how="inner")
    weight_averaged_method = second_filters[["id", "original_title_x", "vote_average", "vote_count"]]
    weight_averaged_method['R'] = weight_averaged_method.vote_average
    C = weight_averaged_method.R.mean()
    weight_averaged_method['v'] = weight_averaged_method.vote_count
    PERCENTAGE = .95 # set to 95%
    m = weight_averaged_method.v.quantile(PERCENTAGE)
    def WR(query):
        R = query.R
        v = query.v
        return (v / (v + m)) * R + (m / (v + m)) * C
    weight_averaged_method["WR"] = weight_averaged_method.apply(WR, axis=1)
    weight_averaged_method = weight_averaged_method.sort_values("WR", ascending=False)
    weight_averaged_method.reset_index(drop=True, inplace=True)
    top_50_movies = weight_averaged_method.head(50)
    contentbased = top_50_movies.merge(datasets[["id", "overview"]], on="id", how="inner")[["id", "original_title_x", "overview"]]
    contentbased.columns = ["id", "original_title", "overview"]
    check_movie_exist = True in list(contentbased.original_title == movie_title) 
    add_title = datasets[["id", "original_title", "overview"]][datasets[["id", "original_title", "overview"]].original_title == movie_title]
    def func(row):
        text = row.overview.lower()
        text = ' ' . join(re.findall(r"[a-z]+", text))
        return text

    contentbased["overview_cleaned"] = contentbased.apply(func, axis=1)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_features = tfidf.fit_transform(contentbased.overview_cleaned)
    tfidf_features.toarray()
    cosine_similarity_tbls = cosine_similarity(tfidf_features)
    movie_index = [index for index, x in enumerate(contentbased["original_title"] == movie_title) if x == True][0]
    top_movies = pd.DataFrame(list(sorted(enumerate(cosine_similarity_tbls[movie_index]), key=lambda x: x[1], reverse=True)))
    top_movies.columns = ["top_index", "cosine_similarity"]
    top_5_movies = pd.DataFrame(contentbased[["id", "original_title"]].values[top_movies.top_index[:6]], columns=["id", "original_title"]) 
    recommended_movies_posters = []
    recommended_movies = []
    #print(f"The movies related to the {movie_title} movie that the user just watched are ...\n")
    for i in top_5_movies.values[1:]:
        movie_id = i[0]
        recommended_movies.append(i[1])
        recommended_movies_posters.append(fetch_poster(movie_id))
    return recommended_movies,recommended_movies_posters

movies_dict = pickle.load(open("movies_dict.pkl", "rb"))
keywords_taglines_genres = pd.DataFrame(movies_dict)

datasets_dict = pickle.load(open("datasets_dict.pkl", "rb"))
datasets = pd.DataFrame(datasets_dict)

cos_results = pickle.load(open("similarity.pkl", "rb"))

st.title("Movie Recommender System")

selected_movie_name = st.selectbox(
    "Get ready to experience mind blowing movies....",
    keywords_taglines_genres["original_title"].values)

if st.button("Recommend"):
    names,posters = Hybrid_recommender(selected_movie_name)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])
