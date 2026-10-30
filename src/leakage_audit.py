
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

SEED = 42
OUTPUT = Path("results")
OUTPUT.mkdir(exist_ok=True)

REPOS = {
    "safeguard": "xTRam1/safe-guard-prompt-injection",
    "deepset": "deepset/prompt-injections",
    "gandalf": "Lakera/gandalf_ignore_instructions",
}


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", text.strip().casefold())


def load_texts():
    frames = []

    for name, repo in REPOS.items():
        dataset = load_dataset(repo)

        for part, rows in dataset.items():
            texts = rows["text"]

            frames.append(pd.DataFrame({
                "row_id": [
                    f"{name}:{part}:{i}"
                    for i in range(len(texts))
                ],
                "text": texts,
            }))

    return pd.concat(frames, ignore_index=True)


def main():
    split = pd.read_csv("split.csv", dtype={"row_id": str})
    texts = load_texts()

    df = split.merge(
        texts,
        on="row_id",
        how="left",
        validate="one_to_one",
    )

    if df["text"].isna().any():
        raise ValueError("Some row IDs could not be matched to text.")

    df["normalized"] = df["text"].map(normalize)

    train = df[df["split"] == "train"].copy()
    evaluation = df[df["split"].isin(["validation", "test"])].copy()

    print("Training rows:", len(train))
    print("Evaluation rows:", len(evaluation))

    train_keys = set(train["normalized"])
    exact = evaluation["normalized"].isin(train_keys).sum()

    print("Exact normalized train/evaluation overlaps:", exact)

    # Character n-grams capture small wording changes,
    # punctuation differences, and similar prompt templates.
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100000,
        dtype=np.float32,
    )

    all_texts = pd.concat(
        [train["normalized"], evaluation["normalized"]],
        ignore_index=True,
    )

    matrix = vectorizer.fit_transform(all_texts)

    train_vectors = matrix[:len(train)]
    eval_vectors = matrix[len(train):]

    neighbors = NearestNeighbors(
        n_neighbors=1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    neighbors.fit(train_vectors)

    # Batch queries to limit temporary memory use.
    similarities = []
    nearest_ids = []
    batch_size = 100

    train_ids = train["row_id"].to_numpy()

    for start in range(0, eval_vectors.shape[0], batch_size):
        batch = eval_vectors[start:start + batch_size]
        distances, indices = neighbors.kneighbors(batch)

        similarities.extend((1.0 - distances[:, 0]).tolist())
        nearest_ids.extend(train_ids[indices[:, 0]].tolist())

    evaluation["nearest_train_id"] = nearest_ids
    evaluation["similarity"] = similarities

    report = evaluation[
        ["row_id", "dataset", "split", "label",
         "nearest_train_id", "similarity"]
    ].sort_values("similarity", ascending=False)

    report.to_csv(OUTPUT / "near_duplicate_audit.csv", index=False)

    print("\nNEAR-DUPLICATE AUDIT")
    print("--------------------")

    for threshold in [0.80, 0.90, 0.95, 0.99]:
        count = int((report["similarity"] >= threshold).sum())
        print(f"Similarity >= {threshold:.2f}: {count}")

    print("\nBy dataset and split:")
    for (dataset, split_name), group in report.groupby(["dataset", "split"]):
        print(
            f"{dataset:12} {split_name:12} "
            f"rows={len(group):5} "
            f">=0.90={int((group.similarity >= 0.90).sum()):5} "
            f">=0.95={int((group.similarity >= 0.95).sum()):5}"
        )

    print("\nTop 10 most similar evaluation rows:")
    print(report.head(10).to_string(index=False))

    print("\nSaved: results/near_duplicate_audit.csv")


if __name__ == "__main__":
    main()
