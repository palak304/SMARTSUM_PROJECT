
from datasets import load_dataset
import pandas as pd
import os

print("Loading CNN/DailyMail Dataset...")

dataset = load_dataset(
    "abisee/cnn_dailymail",
    "3.0.0",
    split="train",
    streaming=True
)

articles = []
summaries = []

for i, item in enumerate(dataset):
    if i == 500:
        break

    articles.append(item["article"])
    summaries.append(item["highlights"])

    if (i + 1) % 100 == 0:
        print(f"Collected {i + 1} articles")

df = pd.DataFrame({
    "article": articles,
    "summary": summaries
})

os.makedirs("datasets", exist_ok=True)

df.to_csv(
    "datasets/cnn_dailymail.csv",
    index=False
)

print("\nDataset downloaded successfully!")
print("Total records:", len(df))
print(df.head())
