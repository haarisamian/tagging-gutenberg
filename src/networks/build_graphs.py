"""
Build character networks across a corpus of BookNLP outputs.
"""

import argparse
import os
from collections import Counter
from pathlib import Path
import networkx as nx
import pandas as pd

def top_proper_noun(df):
    """Return most common proper noun mention"""
    # Proper nouns first
    prop = df.loc[df["prop"] == "PROP", "text"].dropna()
    vc = prop.value_counts()
    if len(vc):
        return vc.idxmax()

    # Fallback: any mention text
    vc = df["text"].dropna().value_counts()
    return vc.idxmax() if len(vc) else None

def get_character_mapping(base_dir, min_mentions=1):
    """
    Build mapping from COREF ID to character name from .entities file.
    """
    ent_file = f"{base_dir}.entities"
    if not os.path.exists(ent_file):
        return {}
    
    ents = pd.read_csv(ent_file, sep="\t", quoting=3)
    ents_per = ents[ents["cat"] == "PER"].copy()
    ents_per["COREF"] = ents_per["COREF"].astype(str)
    
    cluster_summary = (
        ents_per.groupby("COREF")
        .apply(lambda df: pd.Series({
            "n_mentions": len(df),
            "n_prop": (df["prop"] == "PROP").sum(),
            "top_text": top_proper_noun(df)
        }), include_groups=False)
        .reset_index()
    )
    
    # Filter: must have proper noun and meet min_mentions
    clean = cluster_summary[
        (cluster_summary["n_prop"] > 0) & 
        (cluster_summary["n_mentions"] >= min_mentions)
    ].dropna(subset=["top_text"])

    return dict(zip(clean["COREF"], clean["top_text"]))


def build_cooccurrence_graph(base_dir, min_mentions=1):
    """
    Build co-occurrence network.
    Edge weight = number of paragraphs where both characters appear.
    """
    char_map = get_character_mapping(base_dir, min_mentions)
    # char_map["0"] = "Jane"
    # char_map[0] = "Jane"
    if not char_map:
        return None, {}
    
    tok_file = f"{base_dir}.tokens"
    ent_file = f"{base_dir}.entities"
    
    if not os.path.exists(tok_file) or not os.path.exists(ent_file):
        return None, {}
    
    tokens = pd.read_csv(tok_file, sep="\t", quoting=3)
    ents = pd.read_csv(ent_file, sep="\t", quoting=3)
    
    ents = ents[ents["cat"] == "PER"].copy()
    ents["COREF"] = ents["COREF"].astype(str)
    ents["character"] = ents["COREF"].map(char_map)
    ents = ents.dropna(subset=["character"])
    
    # Align to paragraphs
    ents = ents.merge(
        tokens[["token_ID_within_document", "paragraph_ID"]],
        left_on="start_token",
        right_on="token_ID_within_document",
        how="left"
    )
    
    # Count co-occurrences per paragraph
    cooccur = Counter()
    for _, group in ents.groupby("paragraph_ID"):
        chars = sorted(group["character"].unique())
        for i in range(len(chars)):
            for j in range(i + 1, len(chars)):
                cooccur[(chars[i], chars[j])] += 1
    
    G = nx.Graph()
    
    # Add nodes with mention counts
    freq = ents["character"].value_counts().to_dict()
    for char, count in freq.items():
        G.add_node(char, mentions=count)
    
    # Add weighted edges
    for (a, b), w in cooccur.items():
        G.add_edge(a, b, weight=w)
    
    return G, freq


def build_dialogue_graph(base_dir, min_mentions=1):
    """
    Build dialogue network from turn-taking.
    Edge weight = number of consecutive speaking turns between characters.
    """
    char_map = get_character_mapping(base_dir, min_mentions)
    # char_map["0"] = "Jane"
    # char_map[0] = "Jane"
    if not char_map:
        return None, {}
    
    quot_file = f"{base_dir}.quotes"
    if not os.path.exists(quot_file):
        return None, {}
    
    quotes = pd.read_csv(quot_file, sep="\t", quoting=3)
    
    # Handle char_id column
    quotes = quotes.dropna(subset=["char_id"])
    quotes["char_id_str"] = quotes["char_id"].astype(float).astype(int).astype(str)
    
    # Map to canonical names
    quotes["speaker"] = quotes["char_id_str"].map(char_map)
    quotes = quotes.dropna(subset=["speaker"])
    quotes = quotes.sort_values("quote_start")
    
    # Build turn-taking edges
    edges = Counter()
    prev = None
    for sp in quotes["speaker"]:
        if prev and sp != prev:
            edge = tuple(sorted([prev, sp]))
            edges[edge] += 1
        prev = sp
    
    G = nx.Graph()
    
    # Add nodes with quote counts
    speak_freq = quotes["speaker"].value_counts().to_dict()
    for char, count in speak_freq.items():
        G.add_node(char, quotes=count)
    
    # Add weighted edges
    for (a, b), w in edges.items():
        G.add_edge(a, b, weight=w)
    
    return G, speak_freq

def build_discussion_graph(base_dir, min_mentions=1):
    """
    Build DC discussion network (directed, weighted).
    Edge A->B weight = # sentences where speaker A mentions character B inside quotes.
    """
    char_map = get_character_mapping(base_dir, min_mentions)
    char_map["0"] = "Jane"
    char_map[0] = "Jane"
    if not char_map:
        return None, {}

    ent_file = f"{base_dir}.entities"
    tok_file = f"{base_dir}.tokens"
    quot_file = f"{base_dir}.quotes"
    if not (os.path.exists(ent_file) and os.path.exists(tok_file) and os.path.exists(quot_file)):
        return None, {}

    ents = pd.read_csv(ent_file, sep="\t", quoting=3)
    tokens = pd.read_csv(tok_file, sep="\t", quoting=3)
    quotes = pd.read_csv(quot_file, sep="\t", quoting=3)

    # speakers
    quotes = quotes.dropna(subset=["char_id", "quote_start", "quote_end"]).copy()
    quotes["char_id_str"] = quotes["char_id"].astype(float).astype(int).astype(str)
    quotes["speaker"] = quotes["char_id_str"].map(char_map)
    quotes = quotes.dropna(subset=["speaker"])
    quotes = quotes.sort_values("quote_start")

    # entity mentions to canonical character names
    ents = ents[ents["cat"] == "PER"].copy()
    ents["COREF"] = ents["COREF"].astype(str)
    ents["character"] = ents["COREF"].map(char_map)
    ents = ents.dropna(subset=["character"])

    # Precompute token->sentence lookup
    tok2sent = tokens.set_index("token_ID_within_document")["sentence_ID"].to_dict()

    edges = Counter()
    out_freq = Counter()
    in_freq = Counter()

    # For each quote, collect mentioned characters by sentence
    for _, q in quotes.iterrows():
        speaker = q["speaker"]
        qs, qe = int(q["quote_start"]), int(q["quote_end"])

        # mentions fully inside quote span (same idea as your find_chars)
        m = ents[(ents["start_token"] >= qs) & (ents["end_token"] <= qe)]

        if m.empty:
            continue

        # group mentions by sentence via their start_token
        sent2chars = {}
        for _, r in m.iterrows():
            mentioned = r["character"]
            if mentioned == speaker:
                continue
            sent_id = tok2sent.get(int(r["start_token"]))
            if sent_id is None:
                continue
            sent2chars.setdefault(sent_id, set()).add(mentioned)

        # increment 1 per sentence per mentioned character
        for _, chars_in_sent in sent2chars.items():
            for mentioned in chars_in_sent:
                edges[(speaker, mentioned)] += 1
                out_freq[speaker] += 1
                in_freq[mentioned] += 1

    G = nx.DiGraph()

    # Node attributes: in/out discussion volume (useful for prominence)
    all_nodes = set(list(out_freq.keys()) + list(in_freq.keys()))
    for n in all_nodes:
        G.add_node(
            n,
            dc_out=int(out_freq.get(n, 0)),
            dc_in=int(in_freq.get(n, 0))
        )

    # Edge weights
    for (a, b), w in edges.items():
        G.add_edge(a, b, weight=int(w))

    return G, {"dc_out": dict(out_freq), "dc_in": dict(in_freq)}


def process_novel(novel_dir, output_dir, min_mentions=1):
    """Process a single novel, building all graph types"""
    novel_name = Path(novel_dir).name
    
    # Find the base filename
    ent_files = list(Path(novel_dir).glob("*.entities"))
    if not ent_files:
        print(f"  Skipping {novel_name}: no .entities file")
        return None
    
    base_dir = str(ent_files[0]).replace(".entities", "")
    results = {"novel": novel_name}
    
    # # Build co-occurrence graph
    # G_cooc, freq_cooc = build_cooccurrence_graph(base_dir, min_mentions)
    # if G_cooc and G_cooc.number_of_nodes() > 0:
    #     results["cooc_nodes"] = G_cooc.number_of_nodes()
    #     results["cooc_edges"] = G_cooc.number_of_edges()
    #     results["cooc_density"] = nx.density(G_cooc)
        
    #     out_path = Path(output_dir) / f"{novel_name}_cooccurrence.graphml"
    #     nx.write_graphml(G_cooc, out_path)
    # else:
    #     results["cooc_nodes"] = 0
    #     results["cooc_edges"] = 0
    #     results["cooc_density"] = 0
    
    # # Build dialogue graph
    # G_dial, freq_dial = build_dialogue_graph(base_dir, min_mentions)
    # if G_dial and G_dial.number_of_nodes() > 0:
    #     results["dial_nodes"] = G_dial.number_of_nodes()
    #     results["dial_edges"] = G_dial.number_of_edges()
    #     results["dial_density"] = nx.density(G_dial)
        
    #     out_path = Path(output_dir) / f"{novel_name}_dialogue.graphml"
    #     nx.write_graphml(G_dial, out_path)
    # else:
    #     results["dial_nodes"] = 0
    #     results["dial_edges"] = 0
    #     results["dial_density"] = 0
    
    # Build discussion graph
    G_dc, freq_dc = build_discussion_graph(base_dir, min_mentions)
    if G_dc and G_dc.number_of_nodes() > 0:
        results["dc_nodes"] = G_dc.number_of_nodes()
        results["dc_edges"] = G_dc.number_of_edges()
        results["dc_density"] = nx.density(G_dc)
        out_path = Path(output_dir) / f"{novel_name}_discussion.graphml"
        nx.write_graphml(G_dc, out_path)
    else:
        results["dc_nodes"] = 0
        results["dc_edges"] = 0
        results["dc_density"] = 0
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Build character networks from BookNLP corpus")
    parser.add_argument("--corpus-dir", required=True,
                        help="Directory with BookNLP outputs (one folder per novel)")
    parser.add_argument("--output-dir", default="graphs",
                        help="Output directory for graph files")
    parser.add_argument("--min-mentions", type=int, default=1,
                        help="Minimum mentions to include character (default: 1)")
    args = parser.parse_args()
    
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    novel_dirs = sorted([d for d in corpus_dir.iterdir() if d.is_dir()])
    print(f"Found {len(novel_dirs)} novels in {corpus_dir}")
    
    all_results = []
    for i, novel_dir in enumerate(novel_dirs):
        print(f"[{i+1}/{len(novel_dirs)}] {novel_dir.name}")
        result = process_novel(novel_dir, output_dir, args.min_mentions)
        if result:
            all_results.append(result)
    
    # Save summary
    if all_results:
        df = pd.DataFrame(all_results)
        summary_path = output_dir / "graph_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"\nSaved {len(all_results)} novels to {summary_path}")
        
        print(f"\nCorpus summary:")
        print(f"  Co-occurrence nodes: {df['cooc_nodes'].mean():.1f} +/- {df['cooc_nodes'].std():.1f}")
        print(f"  Co-occurrence edges: {df['cooc_edges'].mean():.1f} +/- {df['cooc_edges'].std():.1f}")
        print(f"  Dialogue nodes: {df['dial_nodes'].mean():.1f} +/- {df['dial_nodes'].std():.1f}")
        print(f"  Dialogue edges: {df['dial_edges'].mean():.1f} +/- {df['dial_edges'].std():.1f}")


if __name__ == "__main__":
    main()
