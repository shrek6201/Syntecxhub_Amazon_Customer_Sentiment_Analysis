import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from reviews_common import load_clean_reviews, make_vectorizer

t0 = time.time()
df = load_clean_reviews()

vec = make_vectorizer()
vec.fit(df["TextClean"].sample(120_000, random_state=42))
X = vec.transform(df["TextClean"])
print(f"Vectorised {X.shape[0]:,} reviews ({time.time() - t0:.0f}s)", flush=True)

clf = LogisticRegression(max_iter=300, class_weight="balanced")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
df["PredSentiment"] = cross_val_predict(clf, X, df["StarSentiment"], cv=cv)
print(f"Out-of-fold predictions done ({time.time() - t0:.0f}s)\n", flush=True)

print(classification_report(df["StarSentiment"], df["PredSentiment"]))

df[["Id", "PredSentiment"]].to_csv("data/review_sentiment.csv", index=False)
print("Saved data/review_sentiment.csv")
