"""
Calculates: Centrality measures, Global structure, Geometry, Small-world metrics
"""
import scipy
import argparse
import random
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Centrality Measures
def compute_centrality(G):
    """
    Compute standard centrality measures for all nodes.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame()
    
    results = []
    
    degree_cent = nx.degree_centrality(G)
    
    # For betweenness/closeness, convert weight to distance (higher weight = shorter path)
    # We invert: distance = 1/weight
    G_dist = G.copy()
    for u, v, data in G_dist.edges(data=True):
        data["distance"] = 1.0 / data.get("weight", 1)
    
    betweenness = nx.betweenness_centrality(G_dist, weight="distance", normalized=True)
    
    # Closeness on largest component (using inverted weights)
    if nx.is_connected(G_dist):
        closeness = nx.closeness_centrality(G_dist, distance="distance")
    else:
        largest_cc = max(nx.connected_components(G_dist), key=len)
        G_cc = G_dist.subgraph(largest_cc)
        closeness_cc = nx.closeness_centrality(G_cc, distance="distance")
        closeness = {n: closeness_cc.get(n, 0) for n in G.nodes()}
    
    # FIXED: Eigenvector centrality on largest connected component
    if nx.is_connected(G):
        try:
            eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
        except:
            eigenvector = {n: 0 for n in G.nodes()}
    else:
        # Compute on largest CC, assign 0 to disconnected nodes
        largest_cc = max(nx.connected_components(G), key=len)
        G_cc = G.subgraph(largest_cc).copy()
        try:
            eig_cc = nx.eigenvector_centrality_numpy(G_cc, weight="weight")
            eigenvector = {n: eig_cc.get(n, 0) for n in G.nodes()}
        except:
            eigenvector = {n: 0 for n in G.nodes()}
    
    pagerank = nx.pagerank(G, weight="weight")
    strength = dict(G.degree(weight="weight"))
    
    for node in G.nodes():
        results.append({
            "character": node,
            "mentions": G.nodes[node].get("mentions", G.nodes[node].get("quotes", 0)),
            "degree": G.degree(node),
            "strength": strength[node],
            "degree_centrality": degree_cent[node],
            "betweenness": betweenness[node],
            "closeness": closeness[node],
            "eigenvector": eigenvector[node],
            "pagerank": pagerank[node]
        })
    
    return pd.DataFrame(results).sort_values("degree", ascending=False)

# Global Structural Measures
def gini_coefficient(values):
    """Compute Gini coefficient for a list of values."""
    values = np.array(sorted(values))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0
    cumsum = np.cumsum(values)
    return (2 * np.sum((np.arange(1, n + 1) * values)) / (n * cumsum[-1])) - (n + 1) / n


def degree_centralization(G):
    """
    Compute degree centralization.
    Range: 0 (uniform) to 1 (star graph).
    """
    if G.number_of_nodes() < 3:
        return 0
    
    degrees = [d for _, d in G.degree()]
    max_degree = max(degrees)
    n = G.number_of_nodes()
    
    sum_diff = sum(max_degree - d for d in degrees)
    max_possible = (n - 1) * (n - 2)
    
    if max_possible == 0:
        return 0
    return sum_diff / max_possible


def compute_global_stats(G):
    """Compute global network statistics."""
    if G.number_of_nodes() == 0:
        return {}
    
    stats = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G)
    }
    
    degrees = [d for _, d in G.degree()]
    stats["degree_mean"] = np.mean(degrees)
    stats["degree_std"] = np.std(degrees)
    stats["degree_max"] = max(degrees)
    stats["degree_gini"] = gini_coefficient(degrees)
    stats["degree_centralization"] = degree_centralization(G)
    
    try:
        stats["assortativity"] = nx.degree_assortativity_coefficient(G)
    except:
        stats["assortativity"] = np.nan
    
    stats["clustering_avg"] = nx.average_clustering(G)
    stats["transitivity"] = nx.transitivity(G)
    
    components = list(nx.connected_components(G))
    stats["n_components"] = len(components)
    stats["largest_component_frac"] = len(max(components, key=len)) / G.number_of_nodes()
    
    if nx.is_connected(G):
        stats["avg_path_length"] = nx.average_shortest_path_length(G)
    else:
        largest_cc = max(components, key=len)
        if len(largest_cc) > 1:
            G_cc = G.subgraph(largest_cc)
            stats["avg_path_length"] = nx.average_shortest_path_length(G_cc)
        else:
            stats["avg_path_length"] = np.nan
    
    return stats

# Delta-Hyperbolicity
def four_point_hyperbolicity(G, a, b, c, d):
    """
    Compute delta for a single 4-tuple using the four-point condition.
    """
    try:
        dists = dict(nx.single_source_shortest_path_length(G, a))
        d_ab = dists.get(b, float('inf'))
        d_ac = dists.get(c, float('inf'))
        d_ad = dists.get(d, float('inf'))
        
        dists = dict(nx.single_source_shortest_path_length(G, b))
        d_bc = dists.get(c, float('inf'))
        d_bd = dists.get(d, float('inf'))
        
        dists = dict(nx.single_source_shortest_path_length(G, c))
        d_cd = dists.get(d, float('inf'))
        
        if any(x == float('inf') for x in [d_ab, d_ac, d_ad, d_bc, d_bd, d_cd]):
            return None
        
        s1 = d_ab + d_cd
        s2 = d_ac + d_bd
        s3 = d_ad + d_bc
        
        sums = sorted([s1, s2, s3])
        delta = (sums[2] - sums[1]) / 2
        return delta
    except:
        return None


def compute_delta_hyperbolicity(G, n_samples=1000, seed=42):
    """
    Estimate delta-hyperbolicity by sampling 4-tuples.
    """
    if G.number_of_nodes() < 4:
        return np.nan, np.nan, np.nan
    
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    if G.number_of_nodes() < 4:
        return np.nan, np.nan, np.nan
    
    random.seed(seed)
    nodes = list(G.nodes())
    
    n_possible = len(nodes) * (len(nodes)-1) * (len(nodes)-2) * (len(nodes)-3) // 24
    n_samples = min(n_samples, n_possible)
    
    deltas = []
    sampled = set()
    attempts = 0
    max_attempts = n_samples * 10
    
    while len(deltas) < n_samples and attempts < max_attempts:
        attempts += 1
        sample = tuple(sorted(random.sample(nodes, 4)))
        if sample in sampled:
            continue
        sampled.add(sample)
        
        delta = four_point_hyperbolicity(G, *sample)
        if delta is not None:
            deltas.append(delta)
    
    if not deltas:
        return np.nan, np.nan, np.nan
    
    return max(deltas), np.mean(deltas), np.median(deltas)


# Small-World Analysis
def small_world_coefficient(G, n_random=10, seed=42):
    """
    Compute small-world coefficient sigma = (C/C_rand) / (L/L_rand)
    sigma > 1 indicates small-world structure
    """
    if G.number_of_nodes() < 10:
        return np.nan
    
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    if G.number_of_nodes() < 10:
        return np.nan
    
    C = nx.average_clustering(G)
    L = nx.average_shortest_path_length(G)
    
    random.seed(seed)
    C_rand_list = []
    L_rand_list = []
    
    for i in range(n_random):
        try:
            G_rand = nx.configuration_model(
                [d for _, d in G.degree()], seed=seed + i)
            G_rand = nx.Graph(G_rand)
            G_rand.remove_edges_from(nx.selfloop_edges(G_rand))
            
            if nx.is_connected(G_rand):
                C_rand_list.append(nx.average_clustering(G_rand))
                L_rand_list.append(nx.average_shortest_path_length(G_rand))
        except:
            continue
    
    if not C_rand_list:
        return np.nan
    
    C_rand = np.mean(C_rand_list)
    L_rand = np.mean(L_rand_list)
    
    if C_rand == 0 or L_rand == 0:
        return np.nan
    
    sigma = (C / C_rand) / (L / L_rand)
    return sigma


# Power-Law Fitting
def fit_power_law(degrees):
    """
    Power-law fit. Returns (alpha, xmin, ks_statistic).
    Tries powerlaw library, falls back to simple regression.
    """
    try:
        import powerlaw
        fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
        return fit.alpha, fit.xmin, fit.power_law.KS()
    except ImportError:
        degrees = np.array([d for d in degrees if d > 0])
        if len(degrees) < 5:
            return np.nan, np.nan, np.nan
        
        log_bins = np.logspace(0, np.log10(max(degrees) + 1), 20)
        hist, edges = np.histogram(degrees, bins=log_bins, density=True)
        
        centers = (edges[:-1] + edges[1:]) / 2
        mask = hist > 0
        if mask.sum() < 3:
            return np.nan, np.nan, np.nan
        
        log_x = np.log10(centers[mask])
        log_y = np.log10(hist[mask])
        
        coeffs = np.polyfit(log_x, log_y, 1)
        alpha = -coeffs[0]
        
        return alpha, 1, np.nan
    except:
        return np.nan, np.nan, np.nan

# Main Processing
def process_graph(graph_path):
    """Process a single graph file and compute all statistics."""
    G = nx.read_graphml(graph_path)
    
    if G.is_directed():
        G = G.to_undirected()
    
    novel_name = graph_path.stem.rsplit("_", 1)[0]
    graph_type = graph_path.stem.rsplit("_", 1)[1]
    
    # Global stats
    global_stats = compute_global_stats(G)
    global_stats["novel"] = novel_name
    global_stats["graph_type"] = graph_type
    
    # Delta-hyperbolicity
    delta_max, delta_mean, delta_median = compute_delta_hyperbolicity(G)
    global_stats["delta_max"] = delta_max
    global_stats["delta_mean"] = delta_mean
    global_stats["delta_median"] = delta_median
    
    # Small-world
    global_stats["small_world_sigma"] = small_world_coefficient(G)
    
    # Power-law
    degrees = [d for _, d in G.degree()]
    alpha, xmin, ks = fit_power_law(degrees)
    global_stats["powerlaw_alpha"] = alpha
    global_stats["powerlaw_xmin"] = xmin
    
    # Centrality per character
    centrality_df = compute_centrality(G)
    centrality_df["novel"] = novel_name
    centrality_df["graph_type"] = graph_type
    
    return global_stats, centrality_df


def main():
    parser = argparse.ArgumentParser(
        description="Compute network statistics across corpus")
    parser.add_argument("--graph-dir", required=True,
                        help="Directory containing .graphml files")
    parser.add_argument("--output-dir", default="network_stats",
                        help="Output directory for statistics")
    parser.add_argument("--graph-type", choices=["cooccurrence", "dialogue", "all"],
                        default="all", help="Which graph types to analyze")
    args = parser.parse_args()
    
    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.graph_type == "all":
        graph_files = sorted(graph_dir.glob("*.graphml"))
    else:
        graph_files = sorted(graph_dir.glob(f"*_{args.graph_type}.graphml"))
    
    print(f"Found {len(graph_files)} graph files")
    
    all_global = []
    all_centrality = []
    
    for i, gf in enumerate(graph_files):
        print(f"[{i+1}/{len(graph_files)}] {gf.name}")
        try:
            global_stats, centrality_df = process_graph(gf)
            all_global.append(global_stats)
            all_centrality.append(centrality_df)
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Save results
    if all_global:
        global_df = pd.DataFrame(all_global)
        
        col_order = [
            "novel", "graph_type", "n_nodes", "n_edges", "density",
            "degree_mean", "degree_std", "degree_max", "degree_gini",
            "degree_centralization", "assortativity",
            "clustering_avg", "transitivity", "avg_path_length",
            "n_components", "largest_component_frac",
            "delta_max", "delta_mean", "delta_median",
            "small_world_sigma", "powerlaw_alpha", "powerlaw_xmin"
        ]
        col_order = [c for c in col_order if c in global_df.columns]
        global_df = global_df[col_order]
        
        global_path = output_dir / "corpus_global_stats.csv"
        global_df.to_csv(global_path, index=False)
        print(f"\nSaved global stats to {global_path}")
    
    if all_centrality:
        centrality_df = pd.concat(all_centrality, ignore_index=True)
        centrality_path = output_dir / "corpus_centrality.csv"
        centrality_df.to_csv(centrality_path, index=False)
        print(f"Saved centrality to {centrality_path}")
    
    # Summary
    if all_global:
        print("\n" + "="*60)
        print("CORPUS SUMMARY")
        print("="*60)
        
        for graph_type in global_df["graph_type"].unique():
            subset = global_df[global_df["graph_type"] == graph_type]
            print(f"\n{graph_type.upper()} (n={len(subset)}):")
            print(f"  Nodes: {subset['n_nodes'].mean():.1f} +/- {subset['n_nodes'].std():.1f}")
            print(f"  Edges: {subset['n_edges'].mean():.1f} +/- {subset['n_edges'].std():.1f}")
            print(f"  Gini: {subset['degree_gini'].mean():.3f} +/- {subset['degree_gini'].std():.3f}")
            print(f"  Centralization: {subset['degree_centralization'].mean():.3f} +/- {subset['degree_centralization'].std():.3f}")
            print(f"  Assortativity: {subset['assortativity'].mean():.3f} +/- {subset['assortativity'].std():.3f}")
            print(f"  Delta (max): {subset['delta_max'].mean():.2f} +/- {subset['delta_max'].std():.2f}")
            print(f"  Clustering: {subset['clustering_avg'].mean():.3f} +/- {subset['clustering_avg'].std():.3f}")


if __name__ == "__main__":
    main()
