# %%
import pandas as pd
df = pd.read_csv("data/Reviews.csv")
print(df.shape)
print(df.columns.tolist())
print(df.isnull().sum())
df.head(3)

# %%
# Rating distribution (expecting heavy skew toward 5 stars)
print(df["Score"].value_counts(normalize=True).sort_index().round(3))

# Duplicates: same user, same time, same text, posted across product variants
dupe_cols = ["UserId", "ProfileName", "Time", "Text"]
print("\nDuplicate reviews:", df.duplicated(subset=dupe_cols).sum())

# Helpfulness sanity check: numerator should never exceed denominator
print("Bad helpfulness rows:", (df["HelpfulnessNumerator"] > df["HelpfulnessDenominator"]).sum())

# Convert the Unix timestamp to a real date
df["Date"] = pd.to_datetime(df["Time"], unit="s")
print("\nDate range:", df["Date"].min(), "to", df["Date"].max())

# Review length in words
df["TextLength"] = df["Text"].str.split().str.len()
print("\n", df["TextLength"].describe())

# %%
before = len(df)

# 1. Drop duplicate reviews
df = df.drop_duplicates(subset=["UserId", "ProfileName", "Time", "Text"], keep="first")

# 2. Drop impossible helpfulness rows
df = df[df["HelpfulnessNumerator"] <= df["HelpfulnessDenominator"]]

# 3. Fill missing summaries
df["Summary"] = df["Summary"].fillna("")

df = df.reset_index(drop=True)
print(f"{before:,} -> {len(df):,} rows ({before - len(df):,} removed)")

# 4. Check for HTML tags and URLs in review text
print("Reviews with HTML tags:", df["Text"].str.contains(r"<[^>]+>", regex=True).sum())
print("Reviews with URLs:", df["Text"].str.contains(r"http\S+", regex=True).sum())

# 5. Light cleaning only: strip tags/URLs, collapse whitespace (keep case and punctuation for VADER)
df["TextClean"] = (
    df["Text"]
    .str.replace(r"<[^>]+>", " ", regex=True)
    .str.replace(r"http\S+", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Spot-check one review that had HTML
sample = df[df["Text"].str.contains("<br />", regex=False)].iloc[0]
print("\nBEFORE:", sample["Text"][:200])
print("AFTER: ", sample["TextClean"][:200])

# %%
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report

sia = SentimentIntensityAnalyzer()

# Score every review (takes a couple of minutes on ~394K rows, the REPL will look idle)
df["Compound"] = [sia.polarity_scores(t)["compound"] for t in df["TextClean"]]

# Save scores so we never have to recompute if the kernel restarts
df[["Id", "Compound"]].to_csv("data/vader_scores.csv", index=False)

# Classify using the standard VADER thresholds
df["Sentiment"] = np.select(
    [df["Compound"] >= 0.05, df["Compound"] <= -0.05],
    ["Positive", "Negative"],
    default="Neutral",
)

# Reference label derived from the star rating
df["StarSentiment"] = np.select(
    [df["Score"] <= 2, df["Score"] == 3],
    ["Negative", "Neutral"],
    default="Positive",
)

print("VADER sentiment split:")
print(df["Sentiment"].value_counts(normalize=True).round(3))

print("\nStar-based split:")
print(df["StarSentiment"].value_counts(normalize=True).round(3))

print("\nAgreement (rows = star label, columns = VADER label):")
print(pd.crosstab(df["StarSentiment"], df["Sentiment"], normalize="index").round(3))

print("\n", classification_report(df["StarSentiment"], df["Sentiment"]))

# %%
# 1. Mean VADER score by star rating (should climb steadily from 1 to 5)
print(df.groupby("Score")["Compound"].agg(["mean", "median"]).round(3))

# 2. Does review length affect agreement with the star label?
df["LenBucket"] = pd.cut(
    df["Text"].str.split().str.len(),
    bins=[0, 25, 50, 100, 200, 5000],
    labels=["<=25", "26-50", "51-100", "101-200", "200+"],
)
print("\nAgreement with star label by length:")
print(df.assign(match=df["Sentiment"] == df["StarSentiment"])
        .groupby("LenBucket", observed=True)["match"].mean().round(3))

# 3. Share of reviews with a near-extreme score, by length
print("\nShare with |compound| > 0.9 by length:")
print(df.groupby("LenBucket", observed=True)["Compound"]
        .apply(lambda s: (s.abs() > 0.9).mean()).round(3))

# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reviews_common import load_clean_reviews

df = load_clean_reviews()
df = df.merge(pd.read_csv("data/review_sentiment.csv"), on="Id", how="left")
df["Year"] = df["Date"].dt.year
df["TextLength"] = df["Text"].str.split().str.len()
print(df.shape, "| missing labels:", df["PredSentiment"].isnull().sum())

sns.set_theme(style="whitegrid")
os.makedirs("charts", exist_ok=True)

# %%
# Chart 1
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

star_share = df["Score"].value_counts(normalize=True).sort_index() * 100
axes[0].bar(star_share.index, star_share.values, color="#4C78A8")
axes[0].set_title("Rating distribution")
axes[0].set_xlabel("Stars")
axes[0].set_ylabel("% of reviews")
for x, y in zip(star_share.index, star_share.values):
    axes[0].text(x, y + 0.8, f"{y:.1f}%", ha="center")

order = ["Positive", "Neutral", "Negative"]
methods = pd.DataFrame({
    "Star rating": df["StarSentiment"].value_counts(normalize=True),
    "VADER": df["Sentiment"].value_counts(normalize=True),
    "Trained model": df["PredSentiment"].value_counts(normalize=True),
}).loc[order] * 100
methods.T.plot(kind="bar", stacked=True, ax=axes[1], color=["#4C78A8", "#B0B0B0", "#F58518"])
axes[1].set_title("Sentiment split depends on the method")
axes[1].set_ylabel("% of reviews")
axes[1].set_xlabel("")
axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(title="")

plt.tight_layout()
plt.savefig("charts/01_distribution.png", dpi=150)
plt.show()
print(methods.round(1))

# %%
# Chart 2
yearly = df.groupby("Year").agg(
    Reviews=("Id", "count"),
    AvgRating=("Score", "mean"),
    NegByStars=("StarSentiment", lambda s: (s == "Negative").mean() * 100),
    NegByText=("PredSentiment", lambda s: (s == "Negative").mean() * 100),
).round(2)
print(yearly)

trend = yearly[yearly["Reviews"] >= 1000]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].bar(trend.index, trend["Reviews"], color="#4C78A8")
axes[0].set_title("Reviews per year")
axes[0].set_ylabel("Reviews")

axes[1].plot(trend.index, trend["NegByStars"], marker="o", color="#F58518", label="Negative by star rating")
axes[1].plot(trend.index, trend["NegByText"], marker="o", color="#4C78A8", linestyle="--", label="Negative by text model")
axes[1].set_title("Share of negative reviews over time")
axes[1].set_ylabel("% of reviews")
axes[1].legend()

plt.tight_layout()
plt.savefig("charts/02_trend.png", dpi=150)
plt.show()

# %%
# Chart 3
mismatch = (pd.crosstab(df["Score"], df["PredSentiment"], normalize="index") * 100)[
    ["Negative", "Neutral", "Positive"]
]
print(mismatch.round(1))

plt.figure(figsize=(6, 4))
sns.heatmap(mismatch, annot=True, fmt=".1f", cmap="Blues", cbar=False)
plt.title("What the text says, by star rating (% of reviews)")
plt.ylabel("Stars")
plt.xlabel("Text sentiment (model)")
plt.tight_layout()
plt.savefig("charts/03_rating_vs_text.png", dpi=150)
plt.show()

# %%
hidden_neg = df[(df["Score"] >= 4) & (df["PredSentiment"] == "Negative")]
print(f"4-5 star reviews with negative text: {len(hidden_neg):,} ({len(hidden_neg) / len(df):.1%} of all reviews)\n")
for t in hidden_neg.sample(8, random_state=1)["TextClean"]:
    print("-", t[:320], "\n")

hidden_pos = df[(df["Score"] <= 2) & (df["PredSentiment"] == "Positive")]
print(f"\n1-2 star reviews with positive text: {len(hidden_pos):,} ({len(hidden_pos) / len(df):.1%} of all reviews)\n")
for t in hidden_pos.sample(8, random_state=1)["TextClean"]:
    print("-", t[:320], "\n")

# %%
themes = {
    "Taste & flavor":       r"\b(?:taste|tastes|tasted|flavor|flavour|bland|bitter|tasteless|aftertaste|disgusting|gross|nasty)\b",
    "Freshness & expiry":   r"\b(?:stale|expired|expiration|rancid|moldy|mold|spoiled|rotten)\b",
    "Shipping & packaging": r"\b(?:shipping|shipped|arrived|package|packaging|damaged|broken|crushed|leaked|leaking|delivery)\b",
    "Price & value":        r"\b(?:price|expensive|overpriced|cost|costs|cheaper|rip-?off|worth)\b",
    "Ingredients & health": r"\b(?:ingredients|chemicals|preservatives|artificial|additives|msg|sodium)\b",
    "Not as described":     r"\b(?:described|advertised|misleading|mislabeled|labeled)\b",
    "Health & safety":      r"\b(?:sick|nausea|nauseous|vomit|vomiting|diarrhea|vet|allergic|allergy|illness|poisoning)\b",
}

lower = df["TextClean"].str.lower()
is_neg = df["PredSentiment"] == "Negative"
is_pos = df["PredSentiment"] == "Positive"

rows = []
for name, pattern in themes.items():
    hit = lower.str.contains(pattern, regex=True)
    rows.append({
        "Theme": name,
        "% of negative reviews": hit[is_neg].mean() * 100,
        "% of positive reviews": hit[is_pos].mean() * 100,
    })
themes_df = pd.DataFrame(rows).set_index("Theme")
themes_df["Lift"] = themes_df["% of negative reviews"] / themes_df["% of positive reviews"]
print(themes_df.round(1).sort_values("Lift", ascending=False))
themes_df.round(2).to_csv("data/complaint_themes.csv")

ax = (themes_df[["% of negative reviews", "% of positive reviews"]]
      .sort_values("% of negative reviews")
      .plot(kind="barh", figsize=(9, 4.5), color=["#F58518", "#4C78A8"]))
ax.set_xlabel("% of reviews mentioning the theme")
ax.set_title("Complaint themes: negative vs positive reviews")
plt.tight_layout()
plt.savefig("charts/04_themes.png", dpi=150)
plt.show()

# %%
terms = pd.read_csv("data/top_terms_by_class.csv")
drop = {"not", "highly", "stars", "don't wrong", "won't disappointed", "never did", "overall", "prefer"}
terms = terms[~terms["Term"].isin(drop)]

colors = {"Negative": "#F58518", "Neutral": "#B0B0B0", "Positive": "#4C78A8"}
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, cls in zip(axes, ["Negative", "Neutral", "Positive"]):
    top = terms[terms["Class"] == cls].nsmallest(12, "Rank").iloc[::-1]
    ax.barh(top["Term"], top["Coef"], color=colors[cls])
    ax.set_title(f"{cls}: strongest words and phrases")
    ax.set_xlabel("Model weight")
plt.tight_layout()
plt.savefig("charts/05_top_terms.png", dpi=150)
plt.show()

# %%
voted = df[df["HelpfulnessDenominator"] >= 5].copy()
voted["HelpfulRatio"] = voted["HelpfulnessNumerator"] / voted["HelpfulnessDenominator"]
print(f"{len(voted):,} reviews with 5+ votes ({len(voted) / len(df):.1%} of all)\n")

print("% of reviews that attract 5+ votes, by star rating:")
print((df["HelpfulnessDenominator"] >= 5).groupby(df["Score"]).mean().mul(100).round(1))

print("\nMedian helpful ratio among voted reviews, by star rating:")
print(voted.groupby("Score")["HelpfulRatio"].median().round(2))

print("\nMedian review length (words) by star-based sentiment:")
print(df.groupby("StarSentiment")["TextLength"].median())

# %%
from sklearn.metrics import f1_score

preds = pd.read_csv("data/test_predictions.csv")
classes = ["Negative", "Neutral", "Positive"]
f1 = pd.DataFrame({
    "VADER (baseline)": f1_score(preds["StarSentiment"], preds["VADER"], labels=classes, average=None),
    "TF-IDF + logistic regression": f1_score(preds["StarSentiment"], preds["Predicted"], labels=classes, average=None),
}, index=classes)
f1.loc["Macro average"] = f1.mean()
print(f1.round(2))

ax = f1.plot(kind="bar", figsize=(8, 4.5), color=["#B0B0B0", "#4C78A8"])
ax.set_ylabel("F1 score")
ax.set_title("Per-class F1 on the held-out test set")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=0)
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", padding=2)
plt.tight_layout()
plt.savefig("charts/06_model_comparison.png", dpi=150)
plt.show()
