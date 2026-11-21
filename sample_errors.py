
import pandas as pd

SEED = 42
THRESHOLD = 0.14589301034960014

# Load our final-checkpoint predictions
scores = pd.read_csv("scores_final.csv")

# Select Deepset predictions from Logistic Regression
df = scores[
    (scores["value"] == "deepset")
    & (scores["column"] == "logistic_regression")
].copy()

# Convert scores into predicted labels
df["prediction"] = (
    df["score"] >= THRESHOLD
).astype(int)

# False positives: benign prompts predicted malicious
fp = df[
    (df["label"] == 0)
    & (df["prediction"] == 1)
]

# False negatives: malicious prompts predicted benign
fn = df[
    (df["label"] == 1)
    & (df["prediction"] == 0)
]

# Randomly sample up to 10 of each
fp_sample = fp.sample(n=min(10, len(fp)), random_state=SEED)
fn_sample = fn.sample(n=min(10, len(fn)), random_state=SEED)

sample = pd.concat([fp_sample, fn_sample])

sample["error_type"] = sample["label"].map({
    0: "false_positive",
    1: "false_negative",
})

sample["category"] = "NEEDS_REVIEW"

sample.to_csv("error_sample.csv", index=False)

print(f"False positives available: {len(fp)}")
print(f"False negatives available: {len(fn)}")
print(f"Sampled {len(sample)} errors")
print(sample[["row_id", "error_type", "score"]].to_string(index=False))
