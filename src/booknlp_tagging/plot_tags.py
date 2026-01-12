import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_totals(counts_df, path):
    plot_df = (
        counts_df.groupby(["character"])["count"]
        .sum()
        .reset_index()
    )
    char_order = (
        plot_df.sort_values("count", ascending=False)["character"]
        .tolist()
    )

    plt.figure(figsize=(10, max(6, len(char_order) * 0.3)))
    sns.barplot(
        data=plot_df,
        y="character", x="count",
        order=char_order,
        color="#5DA5DA",
        edgecolor="black",
        linewidth=0.4,
        dodge=True,
        errorbar=None
    )
    plt.xlabel("Total count (across tags and chapters)")
    plt.ylabel("")
    plt.title("BookNLP Total Component Scores for All Characters")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_tag(counts_df, tag, path):
    tag_df = counts_df[counts_df['tag'] == tag]
    plot_df = (
        tag_df.groupby(["character"])["count"]
        .sum()
        .reset_index()
    )
    char_order = (
        plot_df.sort_values("count", ascending=False)["character"]
        .tolist()
    )

    plt.figure(figsize=(10, max(6, len(char_order) * 0.3)))
    sns.barplot(
        data=plot_df,
        y="character", x="count",
        order=char_order,
        color="#5DA5DA",
        edgecolor="black",
        linewidth=0.4,
        dodge=True,
        errorbar=None
    )
    plt.xlabel("Total score (across chapters)")
    plt.ylabel("")
    plt.title(f"BookNLP {tag} score for All Characters")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=False, type=str) # N, A, C, I, DC, or DN
    args = parser.parse_args()

    outdir = "results/tagging_plots/"
    if args.tag in ['N', 'A', 'C', 'I', 'DC', 'DN']:
        os.makedirs(outdir + args.tag, exist_ok=True)
    else:
        os.makedirs(outdir + "totals", exist_ok=True)

    data_path = 'results/tagging_results/tagging_results_full'
    for file in os.listdir(data_path):
        filename = f"{data_path}/{file}"
        counts_df = pd.read_csv(filename)

        if args.tag in ['N', 'A', 'C', 'I', 'DC', 'DN']:
            plot_tag(counts_df, args.tag, f"{outdir}{args.tag}/{file[:file.rfind('_')]}.pdf")
        else:
            plot_totals(counts_df, f"{outdir}totals/{file[:file.rfind('_')]}.pdf")




