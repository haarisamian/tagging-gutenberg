import pandas as pd
import glob

if __name__ == "__main__":
    # Path to folder with CSVs
    path = "tagging_results_chapter/*.csv"

    # Read and concatenate
    df = pd.concat((pd.read_csv(f) for f in glob.glob(path)), ignore_index=True)

    # Save result
    df.to_csv("tagging_combined_chap.csv", index=False)