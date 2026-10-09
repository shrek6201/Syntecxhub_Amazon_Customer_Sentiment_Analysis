import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS


def make_vectorizer():
    negations = {"not", "no", "nor", "never", "cannot", "nothing", "none", "neither", "nobody"}
    return TfidfVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS - negations),
        token_pattern=r"(?u)\b\w[\w']+\b",
        ngram_range=(1, 2),
        min_df=5,
        max_features=100_000,
        sublinear_tf=True,
    )


def load_clean_reviews(with_scores=True):
    """Load Reviews.csv with this project's cleaning applied, so every script and notebook agrees."""
    df = pd.read_csv("data/Reviews.csv")
    df = df.drop_duplicates(subset=["UserId", "ProfileName", "Time", "Text"], keep="first")
    df = df[df["HelpfulnessNumerator"] <= df["HelpfulnessDenominator"]].reset_index(drop=True)
    df["Summary"] = df["Summary"].fillna("")
    df["Date"] = pd.to_datetime(df["Time"], unit="s")
    df["TextClean"] = (
        df["Text"]
        .str.replace(r"<[^>]+>", " ", regex=True)
        .str.replace(r"http\S+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["StarSentiment"] = np.select(
        [df["Score"] <= 2, df["Score"] == 3], ["Negative", "Neutral"], default="Positive"
    )
    if with_scores:
        df = df.merge(pd.read_csv("data/vader_scores.csv"), on="Id", how="left")
        df["Sentiment"] = np.select(
            [df["Compound"] >= 0.05, df["Compound"] <= -0.05],
            ["Positive", "Negative"],
            default="Neutral",
        )
    return df
