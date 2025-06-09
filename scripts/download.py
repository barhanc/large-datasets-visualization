import os
import zipfile
import requests


DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
DATASET_NAME = "news-category-dataset"
DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/rmisra/news-category-dataset"

if not os.path.exists(f"{DATA_DIR}/{DATASET_NAME}.zip"):
    print(f"Download '{DATASET_NAME}'...")
    response = requests.get(DATASET_URL)
    with open(f"{DATA_DIR}/{DATASET_NAME}.zip", "wb") as f:
        f.write(response.content)

print(f"Extract '{DATASET_NAME}.zip'...")
with zipfile.ZipFile(f"{DATA_DIR}/{DATASET_NAME}.zip", "r") as file:
    file.extractall(DATA_DIR)
