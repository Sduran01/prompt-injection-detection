
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

THRESHOLD = 0.14589301034960014

scores = pd.read_csv("scores.csv")
audit = pd.read_csv("results/near_duplicate_audit.csv")

df = scores.merge(
    audit[["row_id", "similarity"]],
    on="row_id",
    how="left",
    validate="one_to_one",
)

if df["similarity"].isna().any():
    raise ValueError("Missing similarity values for some scored rows.")

def evaluate(data):
    y = data["label"].to_numpy()
    s = data["score"].to_numpy()
    pred = (s >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return {
        "rows": len(data),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "precision": precision_score(
            y, pred, zero_division=0
        ),
        "recall": recall_score(
            y, pred, zero_division=0
        ),
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
        "AP": average_precision_score(y, s)
        if len(np.unique(y)) == 2 else np.nan,
        "AUROC": roc_auc_score(y, s)
        if len(np.unique(y)) == 2 else np.nan,
    }

rows = []

for test_set, group in df.groupby("test_set"):
    if test_set == "gandalf_all":
        continue

    for cutoff in [None, 0.90, 0.95, 0.99]:
        subset = (
            group if cutoff is None
            else group[group["similarity"] < cutoff]
        )

        result = evaluate(subset)
        result["test_set"] = test_set
        result["condition"] = (
            "original" if cutoff is None
            else f"similarity < {cutoff}"
        )
        rows.append(result)

report = pd.DataFrame(rows)

Path("results").mkdir(exist_ok=True)
report.to_csv(
    "results/sensitivity_analysis.csv",
    index=False,
)

print(report.to_string(index=False))
