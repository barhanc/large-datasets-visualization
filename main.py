from typing import List, Tuple

import json
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from random import sample
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

from umap import UMAP
from sklearn.manifold import TSNE
import pacmap

def load_dataset(path, count :int = -1) -> pd.DataFrame:

    f = open(path, "r")
    data = json.loads(f.read())

    if count != -1:
        data = sample(data, count) 

    return pd.DataFrame(data)

def vectorize(texts: List[str]) -> Tuple[np.ndarray, TfidfVectorizer]:
    tfidf = TfidfVectorizer(
        max_features=10_000, ngram_range=(1, 2), stop_words="english"
    )
    vecs = tfidf.fit_transform(texts)
    return vecs.toarray(), tfidf

def lda_topics(
    tokenized_texts: List[List[str]], num_topics: int = 10
) -> Tuple[List[int], LdaModel]:
    dictionary = Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    lda = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=8, random_state=42)

    doc_topics: List[int] = []
    for bow in corpus:
        topic_distribution = lda.get_document_topics(bow)
        dominant = max(topic_distribution, key=lambda x: x[1])[0]
        doc_topics.append(dominant)
    return doc_topics, lda

def reduce_dimensions(X: np.ndarray, method: str = "umap") -> np.ndarray:
    
    reducer_dict = {
        "umap": UMAP(n_components=2, random_state=42),
        "tsne": TSNE(n_components=2, random_state=42),
        "pacmap": pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
    }

    return reducer_dict[method].fit_transform(X)

def scatter_plots(df: pd.DataFrame):
    df["topic"] = pd.Categorical(df["topic"])
    topic_order = sorted(df["topic"].unique())
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        fig_topic = px.scatter(
            df,
            x="x",
            y="y",
            color="topic",
            category_orders={"topic": topic_order},
            hover_data=["headline", "short_description", "date", "category"],
            height=700,
        )
        st.plotly_chart(fig_topic, use_container_width=True)

    with col2:
        fig_cat = px.scatter(
            df,
            x="x",
            y="y",
            color="category",
            hover_data=["headline", "short_description", "date", "topic"],
            height=700,
        )
        st.plotly_chart(fig_cat, use_container_width=True)

def topic_evolution_chart(df: pd.DataFrame):
    date_series = pd.to_datetime(
        df["date"].apply(lambda d: datetime(d["year"], d["month"], d["day"]))
    )

    df = df.copy()
    df["month"] = date_series.dt.to_period("M").astype(str)

    monthly = df.groupby(["month", "topic"]).size().reset_index(name="count")
    total = df["month"].value_counts().rename("total").reset_index().rename(
        columns={"index": "month"}
    )
    monthly = monthly.merge(total, on="month")
    monthly["share"] = monthly["count"] / monthly["total"]

    fig = px.line(
        monthly,
        x="month",
        y="share",
        color="topic",
        markers=True,
        height=500,
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Topic Share")
    st.plotly_chart(fig, use_container_width=True)

def sidebar_configuration() -> dict:
    st.sidebar.header("Configuration")

    sample_size = st.sidebar.number_input(
        "Sample size (0 = all)", min_value=0, max_value=200_000, value=20_000, step=1_000
    )

    dr_method = st.sidebar.selectbox(
        "Dimensionality-reduction algorithm", ["umap", "tsne", "pacmap", "trimap"], index=0
    )

    num_topics = st.sidebar.slider("Number of LDA topics", 5, 40, 15)
    run_button = st.sidebar.button("Run analysis ▸")

    return {
        "sample_size": None if sample_size == 0 else sample_size,
        "dr_method": dr_method,
        "num_topics": num_topics,
        "run": run_button,
    }

def main():
    st.set_page_config(page_title="High-Dimensional News Visualization", layout="wide")
    st.title("High-Dimensional Text Visualization of HuffPost News")

    cfg = sidebar_configuration()

    if not cfg["run"]:
        st.info("Configure options in the sidebar and press **Run analysis ▸**")
        st.stop()

    # --------------------------- Data loading --------------------------------
    with st.spinner("Loading dataset …"):
        df = load_dataset('processed_articles.json', count=cfg["sample_size"])

    st.success(f"Loaded {len(df):,} articles.")

    # ------------------------- Vectorizing ---------------------------------
    with st.spinner("Vectorizing …"):
        tokenized = df["processed_headline"] + df["processed_short_description"]
        X, _ = vectorize([" ".join(toks) for toks in tokenized])

    st.success(f"Vectorized articles.")

    # ------------------------- Topic modelling ----------------------
    with st.spinner("Running LDA topic model …"):
        doc_topics, lda_model = lda_topics(tokenized, cfg["num_topics"])

    df["topic"] = doc_topics

    with st.expander("Show topic keywords"):
        for t in range(cfg["num_topics"]):
            words = lda_model.show_topic(t, topn=10)
            st.markdown(f"**Topic {t}:** " + ", ".join(w for w, _ in words))

    # ------------------- Dimensionality reduction ----------------------------
    with st.spinner("Reducing dimensions for visualization …"):
        coords = reduce_dimensions(X, cfg["dr_method"])

    df["x"], df["y"] = coords[:, 0], coords[:, 1]

    st.header("2D Projection of Articles")
    scatter_plots(df)

    st.header("Topic Evolution Over Time")
    topic_evolution_chart(df)

if __name__ == "__main__":
    main()