
from datasets import load_dataset

DATASETS = {
    "safeguard": "xTRam1/safe-guard-prompt-injection",
    "deepset": "deepset/prompt-injections",
    "gandalf": "Lakera/gandalf_ignore_instructions",
}


def load_all_datasets():
    loaded = {}

    for name, repository in DATASETS.items():
        print(f"\n{'=' * 60}")
        print(f"Loading {name}: {repository}")

        dataset = load_dataset(repository)
        loaded[name] = dataset

        print(f"Available splits: {list(dataset.keys())}")

        for split_name, split_data in dataset.items():
            print(f"\nSplit: {split_name}")
            print(f"Rows: {len(split_data)}")
            print(f"Columns: {split_data.column_names}")
            print(f"Features: {split_data.features}")

            if len(split_data) > 0:
                first = split_data[0]
                print("First record schema:")
                for key, value in first.items():
                    print(f"  {key}: {type(value).__name__}")

                # Print label distributions without exposing prompt text
                for column in split_data.column_names:
                    if column.lower() in {"label", "labels", "target"}:
                        from collections import Counter
                        print(
                            f"Label distribution ({column}):",
                            Counter(split_data[column])
                        )

    return loaded


if __name__ == "__main__":
    load_all_datasets()
