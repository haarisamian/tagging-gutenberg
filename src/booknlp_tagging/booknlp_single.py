#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from booknlp.booknlp import BookNLP


def to_pgid(x):
    return int(str(x).strip().upper().replace("PG", ""))


def search_meta(meta_path, query):
    meta = pd.read_csv(meta_path)
    hits = meta[meta["title"].astype(str).str.contains(query, case=False, na=False)]
    return hits[["id", "title", "author"]]


def main():
    if len(sys.argv) < 3:
        print('Usage: python run_booknlp.py "TITLE QUERY" "OUTPUT_FOLDER_NAME"')
        sys.exit(1)

    query = sys.argv[1]
    out_name = sys.argv[2].strip()

    META_PATH = os.path.join("data", "metadata.csv")
    TEXT_DIR = os.path.join("data", "text")
    OUT_BASE = ""

    hits = search_meta(META_PATH, query)
    if hits.empty:
        print("No matches.")
        sys.exit(1)

    for _, row in hits.iterrows():
        pgid = to_pgid(row["id"])
        print(f"PG{pgid}\t{row['title']} — {row['author']}")

    if len(hits) == 1:
        pgid = to_pgid(hits.iloc[0]["id"])
        print(f"\nOnly one match; selecting PG{pgid}.")
    else:
        pgid = int(input("\nType the PG id to run (e.g. 2199): ").strip())

    input_path = Path(TEXT_DIR) / f"PG{pgid}_text.txt"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing text file: {input_path}")

    output_dir = Path(OUT_BASE) / out_name
    output_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "pipeline": "entity,quote,supersense,event,coref",
        "model": "big",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    bnlp = BookNLP("en", params)

    book_id = f"PG{pgid}"
    print(f"\nRunning BookNLP on: {input_path}")
    print(f"Writing outputs to: {output_dir}")
    bnlp.process(str(input_path), str(output_dir), book_id)
    print("Done.")


if __name__ == "__main__":
    main()