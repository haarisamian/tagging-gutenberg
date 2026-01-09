"""
Compute Poincare (hyperbolic) embeddings of character networks.

Embeds character graphs into the Poincare disk
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from gensim.models.poincare import PoincareModel
import matplotlib as mpl

warnings.filterwarnings("ignore")


def graph_to_edges(G, min_weight=1, max_rep=50):
    """Convert NetworkX graph to edge list for Poincare model.
    """
    edges = []
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1)
        if w is None:
            w = 1
        try:
            w = float(w)
        except Exception:
            w = 1.0

        if w < min_weight:
            continue

        # log-scaled replication, at least 1
        rep = int(np.ceil(np.log1p(w)))
        rep = max(1, min(rep, max_rep))

        u = str(u)
        v = str(v)
        # Add both directions to ensure all nodes appear on both sides
        # This prevents the "division by zero" error in negative sampling
        edges.extend([(u, v)] * rep)
        edges.extend([(v, u)] * rep)

    return edges


def train_poincare_embedding(edges, dim=2, epochs=100, negative=10):
    """
    Train Poincare embeddings using gensim.
    
    Returns dict: node -> (x, y) coordinates in Poincare disk
    """
    
    model = PoincareModel(edges, size=dim, negative=negative)
    # print_every=0 causes division by zero in gensim 4.x, use large number instead
    model.train(epochs=epochs, print_every=500)
    
    embeddings = {}
    for node in model.kv.index_to_key:
        vec = model.kv[node]
        embeddings[node] = vec
    
    return embeddings, model


def compute_radial_positions(embeddings):
    """
    Compute radial distance from origin for each node.
    Closer to 0 = more central, closer to 1 = more peripheral.
    """
    radii = {}
    for node, vec in embeddings.items():
        r = np.sqrt(np.sum(vec ** 2))
        radii[node] = r
    return radii


def plot_poincare_disk(embeddings, G, title="Poincare Embedding", 
                       output_path=None, top_n_labels=15):
    """
    Visualize embedding in the Poincare disk.
    
    Node size proportional to mentions/degree.
    Labels shown for top N characters by centrality.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw unit circle (boundary of Poincare disk)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1.5, alpha=0.3)
    ax.fill(np.cos(theta), np.sin(theta), alpha=0.05, color='gray')
    
    # Get node sizes (mentions or degree)
    sizes = {}
    for node in embeddings:
        if node in G.nodes():
            sizes[node] = G.nodes[node].get("mentions", G.nodes[node].get("quotes", G.degree(node)))
        else:
            sizes[node] = 1
    
    max_size = max(sizes.values()) if sizes else 1
    
    # Plot nodes
    xs, ys, ss, nodes = [], [], [], []
    for node, vec in embeddings.items():
        xs.append(vec[0])
        ys.append(vec[1])
        ss.append(100 + (sizes.get(node, 1) / max_size) * 1500)
        nodes.append(node)
    
    # Color by radial distance (central = dark, peripheral = light)
    radii = [np.sqrt(x**2 + y**2) for x, y in zip(xs, ys)]
    
    scatter = ax.scatter(xs, ys, s=ss, c=radii, cmap='RdYlBu_r', 
                         alpha=0.7, edgecolors='black', linewidths=0.5,
                         vmin=0, vmax=1)
    
    # Label top characters
    sorted_nodes = sorted(zip(nodes, sizes.values()), key=lambda x: -x[1])
    top_nodes = set(n for n, _ in sorted_nodes[:top_n_labels])
    
    for node, x, y in zip(nodes, xs, ys):
        if node in top_nodes:
            ax.annotate(node, (x, y), fontsize=8, ha='center', va='bottom',
                       xytext=(0, 5), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                alpha=0.8, edgecolor='gray'))
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Radial distance (0=central, 1=peripheral)', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_embedding(embeddings, G):
    """
    Analyze the embedding: correlate radial position with centrality.
    """
    radii = compute_radial_positions(embeddings)
    
    # Get centrality measures for nodes in embedding
    nodes_in_both = [n for n in radii if n in G.nodes()]
    
    if len(nodes_in_both) < 5:
        return None
    
    G_sub = G.subgraph(nodes_in_both)
    
    degree_cent = nx.degree_centrality(G_sub)
    
    try:
        eigenvector = nx.eigenvector_centrality_numpy(G_sub)
    except:
        eigenvector = {n: 0 for n in nodes_in_both}
    
    pagerank = nx.pagerank(G_sub)
    
    # Build dataframe
    rows = []
    for node in nodes_in_both:
        rows.append({
            "character": node,
            "radius": radii[node],
            "mentions": G.nodes[node].get("mentions", G.nodes[node].get("quotes", 0)),
            "degree_centrality": degree_cent.get(node, 0),
            "eigenvector": eigenvector.get(node, 0),
            "pagerank": pagerank.get(node, 0)
        })
    
    df = pd.DataFrame(rows)
    
    # Correlations (radius should be NEGATIVELY correlated with centrality)
    
    correlations = {}
    for col in ["mentions", "degree_centrality", "eigenvector", "pagerank"]:
        r, p = spearmanr(df["radius"], df[col])
        correlations[col] = {"rho": r, "p": p}
    
    return df, correlations


def process_graph(graph_path, output_dir, dim=2, epochs=100):
    """Process a single graph file."""
    G = nx.read_graphml(graph_path)
    # Remove isolates
    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)

    # Keep largest connected component
    if G.number_of_nodes() >= 5:
        if G.is_directed():
            H = G.to_undirected()
        else:
            H = G
        comps = list(nx.connected_components(H))
        if comps:
            largest = max(comps, key=len)
            G = G.subgraph(largest).copy()
    
    if G.is_directed():
        G = G.to_undirected()
    
    # Need enough nodes for meaningful embedding
    if G.number_of_nodes() < 5:
        print(f"  Skipping: too few nodes ({G.number_of_nodes()})")
        return None
    
    novel_name = graph_path.stem.rsplit("_", 1)[0]
    graph_type = graph_path.stem.rsplit("_", 1)[1]
    
    print(f"  Training embedding ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)...")
    
    # Convert to edges
    edges = graph_to_edges(G)
    
    if len(edges) < 5:
        print(f"  Skipping: too few edges ({len(edges)})")
        return None
    
    # Train embedding
    try:
        embeddings, model = train_poincare_embedding(edges, dim=dim, epochs=epochs)
    except Exception as e:
        print(f"  Error training: {e}")
        return None
    
    # Analyze
    result = analyze_embedding(embeddings, G)
    if result is None:
        print(f"  Could not analyze embedding")
        return None
    
    df, correlations = result
    
    # Save embedding coordinates
    embed_path = output_dir / f"{graph_path.stem}_embedding.csv"
    df.to_csv(embed_path, index=False)
    
    # Plot
    title = f"{novel_name.replace('_', ' ')} ({graph_type})"
    fig_path = output_dir / f"{graph_path.stem}_poincare.png"
    plot_poincare_disk(embeddings, G, title=title, output_path=fig_path)
    
    # Return summary
    return {
        "novel": novel_name,
        "graph_type": graph_type,
        "n_nodes": G.number_of_nodes(),
        "n_embedded": len(embeddings),
        "radius_mentions_rho": correlations["mentions"]["rho"],
        "radius_degree_rho": correlations["degree_centrality"]["rho"],
        "radius_eigenvector_rho": correlations["eigenvector"]["rho"],
        "radius_pagerank_rho": correlations["pagerank"]["rho"],
        "mean_radius": df["radius"].mean(),
        "std_radius": df["radius"].std()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute Poincare embeddings of character networks")
    parser.add_argument("--graph-dir", required=True,
                        help="Directory containing .graphml files")
    parser.add_argument("--output-dir", default="embeddings",
                        help="Output directory")
    parser.add_argument("--graph-type", choices=["cooccurrence", "dialogue", "all"],
                        default="cooccurrence", 
                        help="Which graph type (default: cooccurrence)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--dim", type=int, default=2,
                        help="Embedding dimension (default: 2)")
    args = parser.parse_args()
    
    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find graphs
    if args.graph_type == "all":
        graph_files = sorted(graph_dir.glob("*.graphml"))
    else:
        graph_files = sorted(graph_dir.glob(f"*_{args.graph_type}.graphml"))
    
    print(f"Found {len(graph_files)} graph files")
    
    mpl.rcParams.update({
    "text.usetex": True,                
    "font.family": "serif",     
    "font.serif": ["Computer Modern"],  
    "axes.labelsize": 14,
    "font.size": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "axes.titlesize": 14,
    "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
    })
    
    all_results = []
    for i, gf in enumerate(graph_files):
        print(f"[{i+1}/{len(graph_files)}] {gf.name}")
        result = process_graph(gf, output_dir, dim=args.dim, epochs=args.epochs)
        if result:
            all_results.append(result)
    
    # Save summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = output_dir / "embedding_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n{'='*60}")
        print("EMBEDDING SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully embedded: {len(all_results)} / {len(graph_files)} graphs")
        print(f"\nRadius-Centrality Correlations (should be negative):")
        print(f"  Radius vs Mentions:    {summary_df['radius_mentions_rho'].mean():.3f} +/- {summary_df['radius_mentions_rho'].std():.3f}")
        print(f"  Radius vs Degree:      {summary_df['radius_degree_rho'].mean():.3f} +/- {summary_df['radius_degree_rho'].std():.3f}")
        print(f"  Radius vs Eigenvector: {summary_df['radius_eigenvector_rho'].mean():.3f} +/- {summary_df['radius_eigenvector_rho'].std():.3f}")
        print(f"  Radius vs PageRank:    {summary_df['radius_pagerank_rho'].mean():.3f} +/- {summary_df['radius_pagerank_rho'].std():.3f}")
        print(f"\n(Negative correlation = central characters near disk center)")
        print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()