import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Load data
centrality = pd.read_csv("network_stats/corpus_centrality.csv")
tagging = pd.read_csv("tagging_combined.csv")

# Pivot tagging
tagging_wide = tagging.pivot_table(
    index=['novel', 'character'], columns='tag', values='count', fill_value=0
).reset_index()

# Split and merge
cooc = centrality[centrality['graph_type'] == 'cooccurrence'].copy()
dial = centrality[centrality['graph_type'] == 'dialogue'].copy()

merged = tagging_wide.merge(
    cooc[['novel', 'character', 'pagerank', 'eigenvector', 'degree_centrality', 'betweenness']].rename(
        columns={c: f'{c}_cooc' for c in ['pagerank', 'eigenvector', 'degree_centrality', 'betweenness']}),
    on=['novel', 'character'], how='inner'
)
merged = merged.merge(
    dial[['novel', 'character', 'pagerank', 'eigenvector', 'degree_centrality', 'betweenness']].rename(
        columns={c: f'{c}_dial' for c in ['pagerank', 'eigenvector', 'degree_centrality', 'betweenness']}),
    on=['novel', 'character'], how='left'
)

print(f"Data: {len(merged)} character-novel pairs, {merged['novel'].nunique()} novels")

# Define all measures
tags = ['N', 'C', 'I', 'A', 'DC', 'DN']
cooc_cent = ['pagerank_cooc', 'eigenvector_cooc', 'degree_centrality_cooc', 'betweenness_cooc']
dial_cent = ['pagerank_dial', 'eigenvector_dial', 'degree_centrality_dial', 'betweenness_dial']
all_measures = tags + cooc_cent + dial_cent

# 1. rank corr (Spearman)
def avg_spearman(df, m1, m2):
    rhos = []
    for novel in df['novel'].unique():
        ndf = df[df['novel'] == novel].dropna(subset=[m1, m2])
        if len(ndf) < 5:
            continue
        x, y = ndf[m1].values, ndf[m2].values
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        rho, _ = spearmanr(x, y)
        if not np.isnan(rho):
            rhos.append(rho)
    if rhos:
        return np.mean(rhos), np.std(rhos), len(rhos)
    return np.nan, np.nan, 0

print("1. RANK CORRELATION (Spearman rho, averaged across novels)")
# Build full matrix
corr_results = {}
for m1 in all_measures:
    for m2 in all_measures:
        if m1 != m2:
            corr_results[(m1, m2)] = avg_spearman(merged, m1, m2)

# 2. Top-k AGREEMENT (Jaccard similarity)
def topk_jaccard(df, m1, m2, k):
    jaccards = []
    for novel in df['novel'].unique():
        ndf = df[df['novel'] == novel].dropna(subset=[m1, m2])
        if len(ndf) < k:
            continue
        top1 = set(ndf.nlargest(k, m1)['character'])
        top2 = set(ndf.nlargest(k, m2)['character'])
        if len(top1 | top2) > 0:
            jaccards.append(len(top1 & top2) / len(top1 | top2))
    if jaccards:
        return np.mean(jaccards), np.std(jaccards), len(jaccards)
    return np.nan, np.nan, 0

print("2. Top-K agreement (Jaccard similarity)")
# 3. protagonist agreement (Top-1 exact match)
def top1_match(df, m1, m2):
    matches, total = 0, 0
    for novel in df['novel'].unique():
        ndf = df[df['novel'] == novel].dropna(subset=[m1, m2])
        if len(ndf) < 2:
            continue
        t1 = ndf.loc[ndf[m1].idxmax(), 'character']
        t2 = ndf.loc[ndf[m2].idxmax(), 'character']
        if t1 == t2:
            matches += 1
        total += 1
    return matches / total if total > 0 else np.nan, matches, total

print("3. protagonist agreement (Top-1 exact match)")

# 4. measures of inequality/concentration
def gini(x):
    x = np.array(sorted([v for v in x if v > 0]), dtype=float)
    if len(x) == 0:
        return np.nan
    n = len(x)
    return (2 * np.sum((np.arange(1, n+1) * x)) - (n + 1) * np.sum(x)) / (n * np.sum(x))

def top1_share(x):
    x = np.array(x, dtype=float)
    if x.sum() == 0:
        return np.nan
    return x.max() / x.sum()

def top12_ratio(x):
    x = np.sort(np.array(x, dtype=float))
    if len(x) < 2 or x[-2] == 0:
        return np.nan
    return x[-1] / x[-2]

print("4. Concentration Metrics (per measure, averaged across novels)")

concentration = {}
for m in all_measures:
    if m not in merged.columns:
        continue
    ginis, shares, ratios = [], [], []
    for novel in merged['novel'].unique():
        ndf = merged[merged['novel'] == novel]
        vals = ndf[m].dropna().values
        if len(vals) < 2:
            continue
        g = gini(vals)
        s = top1_share(vals)
        r = top12_ratio(vals)
        if not np.isnan(g): ginis.append(g)
        if not np.isnan(s): shares.append(s)
        if not np.isnan(r): ratios.append(r)
    
    concentration[m] = {
        'gini_mean': np.mean(ginis), 'gini_std': np.std(ginis),
        'top1_share_mean': np.mean(shares), 'top1_share_std': np.std(shares),
        'top12_ratio_mean': np.mean(ratios), 'top12_ratio_std': np.std(ratios),
        'n': len(ginis)
    }

# Tags vs all centrality measures
print("\n" + "-"*80)
print("Tags vs Centrality Measures (Spearman rho)")
print("-"*80)
header = f"{'':8}" + "".join([f"{c.replace('_cooc','_co').replace('_dial','_di').replace('centrality','cent')[:10]:>11}" for c in cooc_cent + dial_cent])
print(header)
for tag in tags:
    row = f"{tag:8}"
    for cent in cooc_cent + dial_cent:
        mean, std, n = corr_results.get((tag, cent), (np.nan, np.nan, 0))
        row += f"{mean:11.2f}"
    print(row)

# Inter-tag correlations
print("\n" + "-"*80)
print("Inter-Tag Correlations")
print("-"*80)
header = f"{'':6}" + "".join([f"{t:>8}" for t in tags])
print(header)
for t1 in tags:
    row = f"{t1:6}"
    for t2 in tags:
        if t1 == t2:
            row += f"{'--':>8}"
        else:
            mean, _, _ = corr_results.get((t1, t2), corr_results.get((t2, t1), (np.nan, np.nan, 0)))
            row += f"{mean:8.2f}"
    print(row)

# Inter-centrality correlations (cooc vs dial)
print("\n" + "-"*80)
print("Cross-Network Centrality Correlations (Co-occ vs Dialogue)")
print("-"*80)
for c in ['pagerank', 'eigenvector', 'degree_centrality', 'betweenness']:
    mean, std, n = corr_results.get((f'{c}_cooc', f'{c}_dial'), (np.nan, np.nan, 0))
    print(f"  {c:20}: rho = {mean:.3f} +- {std:.3f} (n={n})")

# Top-K Agreement
print("\n" + "-"*80)
print("Top-K Jaccard Agreement")
print("-"*80)
key_pairs = [
    ('N', 'pagerank_cooc'), ('N', 'eigenvector_cooc'),
    ('C', 'pagerank_dial'), ('C', 'eigenvector_dial'),
    ('N', 'C'), ('pagerank_cooc', 'pagerank_dial')
]
for k in [1, 3, 5]:
    print(f"\nTop-{k}:")
    for m1, m2 in key_pairs:
        mean, std, n = topk_jaccard(merged, m1, m2, k)
        print(f"  {m1:20} vs {m2:20}: {mean:.3f} +- {std:.3f}")

# Protagonist Agreement
print("\n" + "-"*80)
print("Protagonist Agreement (Top-1 Exact Match)")
print("-"*80)
all_pairs = [
    ('N', 'pagerank_cooc'), ('N', 'eigenvector_cooc'), ('N', 'degree_centrality_cooc'),
    ('C', 'pagerank_dial'), ('C', 'eigenvector_dial'), ('C', 'degree_centrality_dial'),
    ('N', 'C'), ('I', 'A'),
    ('pagerank_cooc', 'pagerank_dial'), ('eigenvector_cooc', 'eigenvector_dial'),
]
for m1, m2 in all_pairs:
    rate, matches, total = top1_match(merged, m1, m2)
    print(f"  {m1:25} vs {m2:25}: {rate:5.1%} ({matches}/{total})")

# Concentration
print("\n" + "-"*80)
print("Concentration Metrics")
print("-"*80)
print(f"{'Measure':30} {'Gini':>12} {'Top-1 Share':>14} {'Top-1/2 Ratio':>14}")
print("-"*70)
for m in tags + cooc_cent[:2] + dial_cent[:2]:
    if m in concentration:
        c = concentration[m]
        name = m.replace('_cooc', ' (co)').replace('_dial', ' (di)')
        print(f"{name:30} {c['gini_mean']:6.2f}+-{c['gini_std']:.2f} {c['top1_share_mean']:8.2f}+-{c['top1_share_std']:.2f} {c['top12_ratio_mean']:8.2f}+-{c['top12_ratio_std']:.2f}")