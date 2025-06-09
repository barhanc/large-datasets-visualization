import os
import json

import nltk
import numpy as np

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
DATASET_NAME = "News_Category_Dataset_v3"

print("Load dataset...")
with open(f"{DATA_DIR}/{DATASET_NAME}.json") as file:
    articles = []
    for line in file.readlines():
        x = json.loads(line.strip("\n"))
        date = x["date"].split("-")
        x["date"] = {"year": int(date[0]), "month": int(date[1]), "day": int(date[2])}

        articles.append(x)

print("Preprocess dataset...")


def preprocess(text: str) -> list[str]:
    lem = WordNetLemmatizer()
    stpw = set(stopwords.words("english"))
    return [
        lem.lemmatize(token)
        for token in word_tokenize(text.lower())
        if token.isalpha() and token not in stpw and len(token) > 2
    ]


for article in tqdm(articles):
    article["processed_short_description"] = preprocess(article["short_description"])
    article["processed_headline"] = preprocess(article["headline"])

print("Save preprocessed dataset...")
with open(f"{DATA_DIR}/{DATASET_NAME}_processed.json", "w", encoding="utf-8") as out_f:
    json.dump(articles, out_f, ensure_ascii=True, indent=4)


print("Compute and save embeddings...")
texts = [" ".join(article["processed_headline"] + article["processed_short_description"]) for article in articles]
model = SentenceTransformer("distilbert-base-nli-mean-tokens")
embeddings = model.encode(texts, show_progress_bar=True)

np.save(f"{DATA_DIR}/{DATASET_NAME}_bert_embeddings.npy", embeddings)
