
import pandas as pd

df = pd.read_csv("error_sample_with_text.csv")

categories = {
    # False positives
    "deepset:test:10": "Short benign topical request",
    "deepset:train:65": "Short benign topical request",
    "deepset:train:430": "Short benign topical request",
    "deepset:test:54": "German benign request",
    "deepset:train:278": "German benign request",
    "deepset:train:167": "Extended benign informational request",
    "deepset:train:352": "German benign request",
    "deepset:train:118": "Short benign topical request",
    "deepset:train:75": "Short benign topical request",
    "deepset:train:442": "German benign request",

    # False negatives
    "deepset:train:382": "Context fabrication",
    "deepset:train:544": "Multilingual attack",
    "deepset:train:510": "Multilingual attack",
    "deepset:train:310": "Multilingual attack",
    "deepset:train:440": "English instruction redirection",
    "deepset:test:95": "English instruction redirection",
    "deepset:train:534": "Multilingual attack",
    "deepset:train:294": "Multilingual attack",
    "deepset:train:453": "Multilingual attack",
    "deepset:train:527": "English instruction redirection",
}

df["category"] = df["row_id"].map(categories)

if df["category"].isna().any():
    raise ValueError("Some errors have no category")

if len(df) != 20 or df["row_id"].duplicated().any():
    raise ValueError("Expected 20 unique sampled errors")

# Professor's required format
errors = df[["row_id", "value", "column", "category"]]

errors.to_csv("errors.csv", index=False)

print("Generated errors.csv")
print(pd.crosstab(df["category"], df["error_type"]))
