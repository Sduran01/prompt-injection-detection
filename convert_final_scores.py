
import pandas as pd

# Load original midsemester predictions
scores = pd.read_csv("scores.csv")

# Convert dataset names into experimental values
dataset_names = {
    "safeguard_test": "safeguard",
    "deepset_all": "deepset",
    "gandalf_all": "gandalf",
}

scores["value"] = scores["test_set"].map(dataset_names)

if scores["value"].isna().any():
    unknown = scores.loc[scores["value"].isna(), "test_set"].unique()
    raise ValueError(f"Unknown test sets: {unknown}")

# Each column identifies one classifier
scores["column"] = "logistic_regression"

# Threshold chosen on the original validation set
scores["threshold"] = 0.14589301034960014

# Final checkpoint format
final_scores = scores[
    ["row_id", "value", "column", "label", "score", "threshold"]
]

final_scores.to_csv("scores_final.csv", index=False)

print(f"Exported {len(final_scores)} predictions")
print(final_scores.groupby(["value", "column"]).size())
