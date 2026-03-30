import pandas as pd
import os
 
 
# ── Paths ──────────────────────────────────────────────────────────────────────
 
RAW_DIR       = os.path.join("Datasets", "Raw")
PROCESSED_DIR = os.path.join("Datasets", "Processed")
 
DATASET_FILES = {
    "d1": "d1.csv",
    "d2": "d2.csv",
    "d3": "d3.csv",
}
 
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "final_merged_dataset.csv")
 
 
# ── Helper functions ───────────────────────────────────────────────────────────
 
def load_dataset(filename: str) -> pd.DataFrame:
    """Load a single CSV from the Raw directory."""
    path = os.path.join(RAW_DIR, filename)
    print(f"  Loading  →  {path}")
    return pd.read_csv(path)
 
 
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all pre-processing steps to a single dataset:
      1. Drop duplicates
      2. Strip leading 'Post:'  prefix from the post column
      3. Strip leading 'Question:' prefix from the question column
      4. Extract Yes / No stress label from the response column
      5. Extract the reasoning text that follows 'Reasoning:'
      6. Lowercase text columns
      7. Drop rows with NaN values
      8. Drop the original response column
      9. Label-encode stress_label  (yes → 1, no → 0)
    """
    # 1. Drop duplicates
    df = df.drop_duplicates()
 
    # 2. Strip 'Post:' prefix
    df["post"] = df["post"].str.replace(r"(?i)^post:\s*", "", regex=True)
 
    # 3. Strip 'Question:' prefix
    df["question"] = df["question"].str.replace(r"(?i)Question:\s*", "", regex=True)
 
    # 4. Extract Yes / No label
    df["stress_label"] = df["response"].str.extract(r"(?i)^(yes|no)")
 
    # 5. Extract reasoning text
    df["reasoning"] = df["response"].str.split("Reasoning:", expand=True)[1]
    df["reasoning"] = df["reasoning"].str.strip()
 
    # 6. Lowercase text columns
    for col in ["post", "question", "reasoning", "stress_label"]:
        df[col] = df[col].str.lower()
 
    # 7. Drop NaN rows
    df = df.dropna()
 
    # 8. Drop raw response column
    df = df.drop(columns=["response"])
 
    # 9. Label-encode
    df["stress_label"] = df["stress_label"].map({"yes": 1, "no": 0})
 
    return df
 
 
def preprocess_all(dataset_files: dict) -> list[pd.DataFrame]:
    """Load and clean every dataset; return a list of cleaned DataFrames."""
    cleaned = []
    for key, filename in dataset_files.items():
        print(f"\n[{key}] Processing '{filename}' ...")
        df = load_dataset(filename)
        df = clean_dataset(df)
        print(f"  ✔  {len(df)} rows retained after cleaning.")
        cleaned.append(df)
    return cleaned
 
 
def merge_and_save(dataframes: list[pd.DataFrame], output_path: str) -> pd.DataFrame:
    """Concatenate cleaned DataFrames, drop any remaining duplicates, and save."""
    merged = pd.concat(dataframes, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates()
    after  = len(merged)
 
    if before != after:
        print(f"\n  Removed {before - after} cross-dataset duplicate(s).")
 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"\n✅ Final merged dataset saved  →  {output_path}")
    print(f"   Total rows : {len(merged)}")
    print(f"   Columns    : {list(merged.columns)}")
    return merged
 
 
# ── Entry point ────────────────────────────────────────────────────────────────
 
def main():
    print("=" * 55)
    print("  Mental Health Risk Detection — Pre-processing")
    print("=" * 55)
 
    cleaned_dfs = preprocess_all(DATASET_FILES)
    final_df    = merge_and_save(cleaned_dfs, OUTPUT_FILE)
    return final_df
 
 
if __name__ == "__main__":
    main()