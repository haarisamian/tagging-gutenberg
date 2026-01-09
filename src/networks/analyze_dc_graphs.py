"""
DC Graph Gender Bias Analysis
"""

import argparse
from pathlib import Path
from collections import Counter
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

MALE_PRONOUNS = {"he", "him", "his", "himself"}
FEMALE_PRONOUNS = {"she", "her", "hers", "herself"}
GENDERED_PRONOUNS = MALE_PRONOUNS | FEMALE_PRONOUNS


def get_top_proper_noun(df):
    """Get most frequent proper noun for a coreference cluster."""
    prop = df.loc[df["prop"] == "PROP", "text"].dropna()
    if len(prop):
        return prop.value_counts().idxmax()
    texts = df["text"].dropna()
    return texts.value_counts().idxmax() if len(texts) else None


def build_coref_to_name(entities_path, min_mentions=1):
    """Map coreference IDs to character names via top proper noun."""
    try:
        ents = pd.read_csv(entities_path, sep="\t", quoting=3)
    except Exception as e:
        print(f"  Warning: Could not read {entities_path}: {e}")
        return {}
    
    ents_per = ents[ents["cat"] == "PER"].copy()
    if ents_per.empty:
        return {}
    
    ents_per["COREF"] = ents_per["COREF"].astype(str)
    
    cluster_summary = (
        ents_per.groupby("COREF")
        .apply(lambda df: pd.Series({
            "n_mentions": len(df),
            "n_prop": (df["prop"] == "PROP").sum(),
            "top_name": get_top_proper_noun(df)
        }), include_groups=False)
        .reset_index()
    )
    
    valid = cluster_summary[
        (cluster_summary["n_prop"] > 0) &
        (cluster_summary["n_mentions"] >= min_mentions)
    ].dropna(subset=["top_name"])
    
    return dict(zip(valid["COREF"], valid["top_name"]))


def build_coref_to_gender(entities_path, min_pronouns=2, min_majority=0.7):
    """Assign gender to coreference clusters based on pronoun majority."""
    try:
        ents = pd.read_csv(entities_path, sep="\t", quoting=3)
    except:
        return {}
    
    pron = ents[(ents["cat"] == "PER") & (ents["prop"] == "PRON")].copy()
    if pron.empty:
        return {}
    
    pron["text_lower"] = pron["text"].astype(str).str.lower().str.strip()
    pron = pron[pron["text_lower"].isin(GENDERED_PRONOUNS)]
    if pron.empty:
        return {}
    
    pron["COREF"] = pron["COREF"].astype(str)
    
    gender_map = {}
    for coref, group in pron.groupby("COREF"):
        male_count = group["text_lower"].isin(MALE_PRONOUNS).sum()
        female_count = group["text_lower"].isin(FEMALE_PRONOUNS).sum()
        total = male_count + female_count
        
        if total < min_pronouns:
            gender_map[coref] = "UNK"
            continue
        
        majority_share = max(male_count, female_count) / total
        if majority_share < min_majority:
            gender_map[coref] = "UNK"
        else:
            gender_map[coref] = "M" if male_count > female_count else "F"
    
    return gender_map


def build_name_to_gender(coref_to_name, coref_to_gender):
    """Aggregate gender assignments across coreference clusters for each name."""
    name_genders = {}
    for coref, name in coref_to_name.items():
        gender = coref_to_gender.get(coref, "UNK")
        name_genders.setdefault(name, []).append(gender)
    
    result = {}
    for name, genders in name_genders.items():
        known = [g for g in genders if g in {"M", "F"}]
        if not known:
            result[name] = "UNK"
        else:
            result[name] = Counter(known).most_common(1)[0][0]
    
    return result


def gini(values):
    """Gini coefficient for inequality."""
    values = np.array(sorted([v for v in values if v > 0]), dtype=float)
    if len(values) == 0:
        return np.nan
    n = len(values)
    return (2 * np.sum((np.arange(1, n+1) * values)) - (n + 1) * np.sum(values)) / (n * np.sum(values))


def extract_dc_node_stats(G, novel):
    """Extract node-level statistics from directed DC graph."""
    try:
        pr = nx.pagerank(G, weight="weight")
    except:
        pr = {n: 0 for n in G.nodes()}
    
    rows = []
    for node in G.nodes():
        rows.append({
            "novel": novel,
            "character": node,
            "in_strength": G.in_degree(node, weight="weight"),
            "out_strength": G.out_degree(node, weight="weight"),
            "in_degree": G.in_degree(node),
            "out_degree": G.out_degree(node),
            "pagerank": pr.get(node, 0),
        })
    
    return pd.DataFrame(rows)


def extract_dc_global_stats(G, novel):
    """Extract global statistics from directed DC graph."""
    in_str = [G.in_degree(n, weight="weight") for n in G.nodes()]
    out_str = [G.out_degree(n, weight="weight") for n in G.nodes()]
    
    return {
        "novel": novel,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "in_strength_gini": gini(in_str),
        "out_strength_gini": gini(out_str),
        "reciprocity": nx.reciprocity(G),
        "in_out_correlation": np.corrcoef(in_str, out_str)[0, 1] if len(in_str) > 2 else np.nan,
    }


def analyze_edge_genders(G, gender_map):
    """Analyze directed edge patterns by gender pair."""
    weights = {'M->M': 0, 'M->F': 0, 'F->M': 0, 'F->F': 0}
    
    for u, v, data in G.edges(data=True):
        g_u = gender_map.get(u, 'UNK')
        g_v = gender_map.get(v, 'UNK')
        
        if g_u in {'M', 'F'} and g_v in {'M', 'F'}:
            key = f"{g_u}->{g_v}"
            weights[key] += data.get('weight', 1)
    
    total = sum(weights.values())
    if total == 0:
        return None
    
    f_out = weights['F->M'] + weights['F->F']
    m_out = weights['M->M'] + weights['M->F']
    f_in = weights['M->F'] + weights['F->F']
    m_in = weights['M->M'] + weights['F->M']
    
    return {
        'MM': weights['M->M'],
        'MF': weights['M->F'],
        'FM': weights['F->M'],
        'FF': weights['F->F'],
        'pct_MM': 100 * weights['M->M'] / total,
        'pct_MF': 100 * weights['M->F'] / total,
        'pct_FM': 100 * weights['F->M'] / total,
        'pct_FF': 100 * weights['F->F'] / total,
        'F_out_share': f_out / (f_out + m_out) if (f_out + m_out) > 0 else np.nan,
        'F_in_share': f_in / (f_in + m_in) if (f_in + m_in) > 0 else np.nan,
        'FM_MF_ratio': weights['F->M'] / weights['M->F'] if weights['M->F'] > 0 else np.nan,
    }


def compute_topk_representation(df, metric, k=10):
    """Compute gender representation in top-k vs baseline."""
    rows = []
    
    for novel, ndf in df.groupby("novel"):
        known = ndf[ndf["gender"].isin({"M", "F"})].copy()
        if len(known) < k:
            continue
        
        base_dist = known["gender"].value_counts(normalize=True)
        topk = known.nlargest(k, metric)
        topk_dist = topk["gender"].value_counts(normalize=True)
        
        for gender in ["M", "F"]:
            base = base_dist.get(gender, 0)
            top = topk_dist.get(gender, 0)
            
            rows.append({
                "novel": novel,
                "metric": metric,
                "k": k,
                "gender": gender,
                "base_share": base,
                "topk_share": top,
                "rep_ratio": top / base if base > 0 else np.nan,
                "diff": top - base,
            })
    
    return pd.DataFrame(rows)


def find_entities_file(novel_dir):
    """Find the .entities file in a BookNLP output directory."""
    files = list(Path(novel_dir).glob("*.entities"))
    return files[0] if files else None


def p_str(p):
    """Format p-value for display."""
    if p < 0.001:
        return "<.001***"
    elif p < 0.01:
        return f"{p:.3f}**"
    elif p < 0.05:
        return f"{p:.3f}*"
    else:
        return f"{p:.3f}"


def main(dc_graph_dir, booknlp_dir, k=10):
    dc_graph_dir = Path(dc_graph_dir)
    booknlp_dir = Path(booknlp_dir)
    
    graph_files = sorted(dc_graph_dir.glob("*_discussion.graphml"))
    print(f"Found {len(graph_files)} DC graphs")
    
    all_node_stats = []
    all_global_stats = []
    all_edge_stats = []
    gender_coverage = []
    
    for gpath in graph_files:
        novel = gpath.name.replace("_discussion.graphml", "")
        novel_dir = booknlp_dir / novel
        entities_path = find_entities_file(novel_dir)
        
        if not entities_path:
            print(f"  {novel}: No entities file found, skipping")
            continue
        
        print(f"  {novel}: Processing...")
        
        G = nx.read_graphml(gpath)
        
        coref_to_name = build_coref_to_name(entities_path)
        coref_to_gender = build_coref_to_gender(entities_path)
        name_to_gender = build_name_to_gender(coref_to_name, coref_to_gender)
        
        node_df = extract_dc_node_stats(G, novel)
        node_df["gender"] = node_df["character"].map(name_to_gender).fillna("UNK")
        all_node_stats.append(node_df)
        
        global_stats = extract_dc_global_stats(G, novel)
        all_global_stats.append(global_stats)
        
        # Edge gender analysis
        name_to_gender_map = {row['character']: row['gender'] for _, row in node_df.iterrows()}
        edge_stats = analyze_edge_genders(G, name_to_gender_map)
        if edge_stats:
            edge_stats['novel'] = novel
            all_edge_stats.append(edge_stats)
        
        gcounts = node_df["gender"].value_counts().to_dict()
        gender_coverage.append({
            "novel": novel,
            "n_characters": len(node_df),
            "n_male": gcounts.get("M", 0),
            "n_female": gcounts.get("F", 0),
            "n_unknown": gcounts.get("UNK", 0),
            "pct_known": 100 * (gcounts.get("M", 0) + gcounts.get("F", 0)) / len(node_df),
        })
    
    node_df = pd.concat(all_node_stats, ignore_index=True) if all_node_stats else pd.DataFrame()
    global_df = pd.DataFrame(all_global_stats) if all_global_stats else pd.DataFrame()
    coverage_df = pd.DataFrame(gender_coverage)
    
    node_df.to_csv("dc_node_stats.csv", index=False)
    global_df.to_csv("dc_global_stats.csv", index=False)
    coverage_df.to_csv("dc_gender_coverage.csv", index=False)
    
    print(f"\nSaved: dc_node_stats.csv")
    print(f"Saved: dc_global_stats.csv")
    print(f"Saved: dc_gender_coverage.csv")
    
    # Gender bias analysis
    print("\nGENDER BIAS ANALYSIS")
    
    bias_in = compute_topk_representation(node_df, "in_strength", k=k)
    bias_out = compute_topk_representation(node_df, "out_strength", k=k)
    bias_pr = compute_topk_representation(node_df, "pagerank", k=k)
    
    bias_df = pd.concat([bias_in, bias_out, bias_pr], ignore_index=True)
    bias_df.to_csv("dc_gender_bias.csv", index=False)
    print(f"Saved: dc_gender_bias.csv")
    
    print(f"\nCorpus: {len(global_df)} novels")
    print(f"Gender coverage: {coverage_df['pct_known'].mean():.1f}% of characters")
    
    # Top-k coverage check
    top_k_coverage = []
    for novel, ndf in node_df.groupby('novel'):
        for metric in ['in_strength', 'out_strength']:
            topk = ndf.nlargest(k, metric)
            known = topk['gender'].isin({'M', 'F'}).sum()
            top_k_coverage.append(known)
    print(f"Top-{k} gender coverage: {np.mean(top_k_coverage):.1f}/{k} ({100*np.mean(top_k_coverage)/k:.0f}%)")
    
    print(f"\nTop-{k} Representation Ratios (mean +/- std)")
    print("(>1 = overrepresented, <1 = underrepresented)\n")
    
    significance_results = []
    
    for metric in ["in_strength", "out_strength", "pagerank"]:
        label = {
            "in_strength": "In-strength (discussed)",
            "out_strength": "Out-strength (discussing)", 
            "pagerank": "PageRank"
        }[metric]
        
        subset = bias_df[bias_df["metric"] == metric]
        male = subset[subset["gender"] == "M"]["rep_ratio"].dropna()
        female = subset[subset["gender"] == "F"]["rep_ratio"].dropna()
        
        t_m, p_m = scipy_stats.ttest_1samp(male, 1.0)
        t_f, p_f = scipy_stats.ttest_1samp(female, 1.0)
        
        male_by_novel = subset[subset["gender"] == "M"].set_index("novel")["rep_ratio"]
        female_by_novel = subset[subset["gender"] == "F"].set_index("novel")["rep_ratio"]
        common = male_by_novel.index.intersection(female_by_novel.index)
        
        if len(common) > 2:
            t_paired, p_paired = scipy_stats.ttest_rel(female_by_novel[common], male_by_novel[common])
        else:
            t_paired, p_paired = np.nan, np.nan
        
        significance_results.append({
            "metric": metric,
            "male_mean": male.mean(),
            "male_std": male.std(),
            "male_p": p_m,
            "female_mean": female.mean(),
            "female_std": female.std(),
            "female_p": p_f,
            "paired_t": t_paired,
            "paired_p": p_paired,
        })
        
        print(f"{label}:")
        print(f"  Male:   {male.mean():.3f} +/- {male.std():.3f}  (vs 1.0: p={p_str(p_m)})")
        print(f"  Female: {female.mean():.3f} +/- {female.std():.3f}  (vs 1.0: p={p_str(p_f)})")
        print(f"  Female vs Male: t={t_paired:.2f}, p={p_str(p_paired)}")
        print()
    
    sig_df = pd.DataFrame(significance_results)
    sig_df.to_csv("dc_gender_significance.csv", index=False)
    print(f"Saved: dc_gender_significance.csv")
    
    # Global stats summary
    print("DC GRAPH GLOBAL STATISTICS")
    print(f"In-strength Gini:  {global_df['in_strength_gini'].mean():.3f} +/- {global_df['in_strength_gini'].std():.3f}")
    print(f"Out-strength Gini: {global_df['out_strength_gini'].mean():.3f} +/- {global_df['out_strength_gini'].std():.3f}")
    print(f"Reciprocity:       {global_df['reciprocity'].mean():.3f} +/- {global_df['reciprocity'].std():.3f}")
    print(f"In-out corr:       {global_df['in_out_correlation'].mean():.3f} +/- {global_df['in_out_correlation'].std():.3f}")
    
    # Edge gender analysis
    if all_edge_stats:
        edge_df = pd.DataFrame(all_edge_stats)
        edge_df.to_csv("dc_edge_gender.csv", index=False)
        print(f"\nSaved: dc_edge_gender.csv")
        
        print("\nEDGE GENDER ANALYSIS")
        print(f"Edge type breakdown (mean across {len(edge_df)} novels):")
        print(f"  F->F (women discussing women): {edge_df['pct_FF'].mean():.1f}% +/- {edge_df['pct_FF'].std():.1f}%")
        print(f"  F->M (women discussing men):   {edge_df['pct_FM'].mean():.1f}% +/- {edge_df['pct_FM'].std():.1f}%")
        print(f"  M->F (men discussing women):   {edge_df['pct_MF'].mean():.1f}% +/- {edge_df['pct_MF'].std():.1f}%")
        print(f"  M->M (men discussing men):     {edge_df['pct_MM'].mean():.1f}% +/- {edge_df['pct_MM'].std():.1f}%")
        
        print(f"\nFemale share of outgoing (discussing): {100*edge_df['F_out_share'].mean():.1f}% +/- {100*edge_df['F_out_share'].std():.1f}%")
        print(f"Female share of incoming (discussed):  {100*edge_df['F_in_share'].mean():.1f}% +/- {100*edge_df['F_in_share'].std():.1f}%")
        
        fm_mf = edge_df['FM_MF_ratio'].dropna()
        print(f"\nF->M / M->F ratio: {fm_mf.mean():.2f} +/- {fm_mf.std():.2f}")
        print(f"  (women discuss men {fm_mf.mean():.1f}x more than men discuss women)")

        t, p = scipy_stats.ttest_1samp(fm_mf, 1.0)
        print(f"  t={t:.2f}, p={p_str(p)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DC graph gender bias analysis")
    parser.add_argument("--dc-graphs", required=True, help="Directory with *_discussion.graphml files")
    parser.add_argument("--booknlp", required=True, help="BookNLP output directory")
    parser.add_argument("--k", type=int, default=10, help="Top-k for analysis")
    
    args = parser.parse_args()
    main(args.dc_graphs, args.booknlp, args.k)