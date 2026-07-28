import pandas as pd
import pickle
import os
import chardet

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==========================
# File Path
# ==========================
file_path = r"C:\Users\Armaan\Desktop\sentimentANALYSIS\data\datasets.csv"

# ==========================
# Detect Encoding
# ==========================
with open(file_path, "rb") as f:
    result = chardet.detect(f.read(100000))

encoding = result["encoding"]

print(f"Detected Encoding: {encoding}")

# ==========================
# Read CSV
# ==========================
encodings = [
    encoding,
    "utf-8",
    "utf-8-sig",
    "latin1",
    "cp1252",
    "ISO-8859-1"
]

df = None

for enc in encodings:
    if enc is None:
        continue

    try:
        print(f"Trying: {enc}")
        df = pd.read_csv(
            file_path,
            encoding=enc,
            engine="python",
            on_bad_lines="skip"
        )
        print(f"Loaded Successfully with {enc}")
        break

    except Exception as e:
        print(e)

if df is None:
    raise Exception("Could not read dataset.")

# ==========================
# Check Columns
# ==========================
columns = ["sentiment", "id", "date", "query", "user", "text"]

df = pd.read_csv(
    file_path,
    encoding="latin1",
    names=columns,
    header=None
)

print(df.head())
print(df.columns)


# Remove missing values
df = df[["text", "sentiment"]].dropna()

# Convert to string
df["text"] = df["text"].astype(str)
df["sentiment"] = df["sentiment"].astype(str)

X = df["text"]
y = df["sentiment"]

# ==========================
# TF-IDF
# ==========================
vectorizer = TfidfVectorizer(stop_words="english")

X_vectorized = vectorizer.fit_transform(X)

# ==========================
# Train Model
# ==========================
model = LogisticRegression(max_iter=1000)

model.fit(X_vectorized, y)

# ==========================
# Save Model
# ==========================
os.makedirs("model", exist_ok=True)

pickle.dump(model, open("model/sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Model Trained Successfully")

