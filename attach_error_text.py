
import pandas as pd
from main import load

# Read the randomly sampled errors
errors = pd.read_csv("error_sample.csv")

# Load original datasets without training the model
df, _ = load()

# Match error IDs to their original prompts
lookup = df[["row_id", "text"]]

merged = errors.merge(
    lookup,
    on="row_id",
    how="left",
    validate="one_to_one"
)

missing = merged["text"].isna().sum()
if missing:
    raise ValueError(f"Could not find text for {missing} rows")

merged.to_csv("error_sample_with_text.csv", index=False)

print(f"Saved {len(merged)} sampled errors with text")
