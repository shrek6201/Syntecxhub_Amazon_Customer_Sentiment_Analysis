import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from reviews_common import load_clean_reviews

t0 = time.time()
df = load_clean_reviews()
print(f"Loaded {len(df):,} reviews ({time.time() - t0:.0f}s)", flush=True)

# Stratified 80/20 split so every class keeps its share in both sets
train_idx, test_idx = train_test_split(
    df.index, test_size=0.2, random_state=42, stratify=df["StarSentiment"]
)

# Keep negation words: "not good" must not collapse into "good"
negations = {"not", "no", "nor", "never", "cannot", "nothing", "none", "neither", "nobody"}
vec = TfidfVectorizer(
    stop_words=list(ENGLISH_STOP_WORDS - negations),
    token_pattern=r"(?u)\b\w[\w']+\b",   # keeps "don't", "wasn't" as single tokens
    ngram_range=(1, 2),
    min_df=5,
    max_features=100_000,
    sublinear_tf=True,
)

# Build the vocabulary on a 120K sample (keeps memory low), then transform everything
print("Building vocabulary...", flush=True)
vec.fit(df.loc[train_idx, "TextClean"].sample(120_000, random_state=42))
print(f"Vocabulary: {len(vec.vocabulary_):,} terms ({time.time() - t0:.0f}s)", flush=True)

X_train = vec.transform(df.loc[train_idx, "TextClean"])
X_test = vec.transform(df.loc[test_idx, "TextClean"])
y_train = df.loc[train_idx, "StarSentiment"]
y_test = df.loc[test_idx, "StarSentiment"]
print(f"Vectorised ({time.time() - t0:.0f}s). Training...", flush=True)

# class_weight="balanced" stops the 78% Positive class from drowning out the rest
clf = LogisticRegression(max_iter=300, class_weight="balanced")
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(f"Trained ({time.time() - t0:.0f}s)\n", flush=True)

labels = ["Negative", "Neutral", "Positive"]
print("=== Trained model (held-out test set) ===")
print(classification_report(y_test, pred))
print("Confusion matrix (rows = star label, columns = predicted):")
print(pd.DataFrame(confusion_matrix(y_test, pred, labels=labels), index=labels, columns=labels))

print("\n=== VADER baseline on the same test set ===")
print(classification_report(y_test, df.loc[test_idx, "Sentiment"]))

# Words that push a review toward each class
terms = np.array(vec.get_feature_names_out())
rows = []
for i, cls in enumerate(clf.classes_):
    for rank, j in enumerate(np.argsort(clf.coef_[i])[::-1][:25], 1):
        rows.append({"Class": cls, "Rank": rank, "Term": terms[j], "Coef": round(clf.coef_[i][j], 3)})
top_terms = pd.DataFrame(rows)
top_terms.to_csv("data/top_terms_by_class.csv", index=False)
print("\nTop terms by class:")
print(top_terms.pivot(index="Rank", columns="Class", values="Term").to_string())

pd.DataFrame({
    "Id": df.loc[test_idx, "Id"], "StarSentiment": y_test,
    "Predicted": pred, "VADER": df.loc[test_idx, "Sentiment"],
}).to_csv("data/test_predictions.csv", index=False)
print("\nSaved data/top_terms_by_class.csv and data/test_predictions.csv")
