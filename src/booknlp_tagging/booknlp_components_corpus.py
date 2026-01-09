"""
Tag character components (N, C, I, A, DC, DN) across a corpus of BookNLP outputs.
"""

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import networkx as nx
from nltk.corpus import verbnet

# Character mapping (from build_graphs.py)

def top_proper_noun(df):
    """Return most common proper noun mention"""
    prop = df.loc[df["prop"] == "PROP", "text"].dropna()
    vc = prop.value_counts()
    if len(vc):
        return vc.idxmax()
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
    
    clean = cluster_summary[
        (cluster_summary["n_prop"] > 0) & 
        (cluster_summary["n_mentions"] >= min_mentions)
    ].dropna(subset=["top_text"])

    return dict(zip(clean["COREF"], clean["top_text"]))

def is_communicative(verb):
    """Check if verb is communicative using VerbNet"""
    classes = verbnet.classids(verb)
    for cid in classes:
        if any(k in cid.lower() for k in ['37.', 'judgment']) and verb != "read":
            return True
    return False


def is_mental(verb):
    """Check if verb is mental using VerbNet"""
    classes = verbnet.classids(verb)
    for cid in classes:
        if any(k in cid.lower() for k in [
            'conjecture', 'consider', 'see-30.1-1', 'want', 'long', 
            'wish', 'care', 'comprehend', 'empathize', 'admire', 
            'marvel', 'characterize'
        ]):
            return True
    return False


def is_mental_ss(ss):
    """Check if supersense indicates mental verb"""
    if pd.isna(ss):
        return False
    return any(t in str(ss).lower() for t in ['cognition', 'emotion', 'feeling'])


def is_communicative_ss(ss):
    """Check if supersense indicates communicative verb"""
    if pd.isna(ss):
        return False
    return 'communication' in str(ss).lower()

def find_char(tok_id, entities_df):
    """Find character name at token position"""
    mask = (entities_df["start_token"] <= tok_id) & (entities_df["end_token"] >= tok_id)
    temp = entities_df.loc[mask, 'character']
    if len(temp) == 0:
        return ""
    return temp.iloc[0]


def in_quote(tok_id, quotes_df):
    """Check if token is inside a quote"""
    mask = (quotes_df["quote_start"] <= tok_id) & (quotes_df["quote_end"] >= tok_id)
    return len(quotes_df.loc[mask]) > 0


def find_chars_in_span(start, end, entities_df):
    """Find all characters mentioned in a token span"""
    mask = (start <= entities_df['start_token']) & (entities_df['end_token'] <= end)
    temp = entities_df.loc[mask, ['character', 'start_token']]
    if len(temp) == 0:
        return []
    return list(zip(temp['character'], temp['start_token']))


def get_chapter_df(df, key, chapter_start, chapter_end):
    """Filter dataframe to chapter boundaries"""
    if chapter_end == -1:
        return df[df[key] >= chapter_start]
    return df[(df[key] >= chapter_start) & (df[key] <= chapter_end)]

def tag_novel(base_dir, min_mentions=1, by_chapter=True):
    """
    Tag all components for a single novel.
    """
    char_map = get_character_mapping(base_dir, min_mentions)
    if not char_map:
        return None
    
    # Load files
    ent_file = f"{base_dir}.entities"
    tok_file = f"{base_dir}.tokens"
    quot_file = f"{base_dir}.quotes"
    ss_file = f"{base_dir}.supersense"
    
    required = [ent_file, tok_file, quot_file]
    if not all(os.path.exists(f) for f in required):
        return None
    
    # Load dataframes
    entities_df = pd.read_csv(ent_file, sep="\t", quoting=3)
    tokens_df = pd.read_csv(tok_file, sep="\t", quoting=3)
    quotes_df = pd.read_csv(quot_file, sep="\t", quoting=3)
    
    # Load supersense if available
    if os.path.exists(ss_file):
        supersense_df = pd.read_csv(ss_file, sep="\t")
    else:
        supersense_df = pd.DataFrame(columns=['start_token', 'supersense_category'])
    
    # Filter to PER entities and map to character names
    entities_df = entities_df[entities_df["cat"] == "PER"].copy()
    entities_df["COREF"] = entities_df["COREF"].astype(str)
    entities_df["character"] = entities_df["COREF"].map(char_map)
    entities_df = entities_df.dropna(subset=["character"])
    
    # Process quotes, map speakers
    quotes_df = quotes_df.dropna(subset=["char_id", "quote_start", "quote_end"]).copy()
    quotes_df["char_id_str"] = quotes_df["char_id"].astype(float).astype(int).astype(str)
    quotes_df["speaker"] = quotes_df["char_id_str"].map(char_map)
    quotes_df = quotes_df.dropna(subset=["speaker"])
    quotes_df = quotes_df.sort_values("quote_start")
    
    # Build subjects dataframe for I, A, DN
    subjects_df = tokens_df[
        (tokens_df['dependency_relation'] == 'nsubj') | 
        (tokens_df['dependency_relation'] == 'nsubjpass')
    ].copy()
    subjects_df = subjects_df[subjects_df['POS_tag'].isin(['NOUN', 'PRON', 'PROPN'])]
    subjects_df['character'] = subjects_df['token_ID_within_document'].apply(
        lambda x: find_char(x, entities_df)
    )
    subjects_df = subjects_df[subjects_df['character'] != '']
    
    # Get verb info
    subjects_df['verb_lemma'] = subjects_df['syntactic_head_ID'].apply(
        lambda x: tokens_df.loc[x, 'lemma'] if x in tokens_df.index else ''
    )
    subjects_df['verb_token'] = subjects_df['syntactic_head_ID'].apply(
        lambda x: tokens_df.loc[x, 'token_ID_within_document'] if x in tokens_df.index else -1
    )
    
    # Get supersense
    ss_lookup = supersense_df.set_index('start_token')['supersense_category'].to_dict()
    subjects_df['supersense'] = subjects_df['verb_token'].map(lambda x: ss_lookup.get(x, ''))
    
    # Classify verbs
    subjects_df['mental'] = subjects_df.apply(
        lambda x: is_mental_ss(x['supersense']) or is_mental(x['verb_lemma']), 
        axis=1
    )
    subjects_df['communicative'] = subjects_df.apply(
        lambda x: is_communicative_ss(x['supersense']) or is_communicative(x['verb_lemma']), 
        axis=1
    )
    subjects_df['action'] = subjects_df.apply(
        lambda x: not x['communicative'] and not x['mental'] and 'stative' not in str(x['supersense']),
        axis=1
    )
    subjects_df['description'] = subjects_df['verb_lemma'].apply(lambda x: x == 'be')
    subjects_df['in_quote'] = subjects_df['token_ID_within_document'].apply(
        lambda x: in_quote(x, quotes_df)
    )
    
    # Get all characters for output
    all_characters = sorted(entities_df['character'].unique())
    
    # Find chapters if doing by_chapter
    if by_chapter:
        # Try common chapter markers
        chapter_markers = ['Chapter', 'CHAPTER', 'chapter']
        chapters_found = None
        for marker in chapter_markers:
            temp = tokens_df[tokens_df['word'] == marker]
            if len(temp) > 1:
                chapters_found = temp
                break
        
        if chapters_found is not None and len(chapters_found) > 1:
            chapters = {}
            for i in range(len(chapters_found) - 1):
                start = chapters_found.iloc[i]['token_ID_within_document']
                end = chapters_found.iloc[i + 1]['token_ID_within_document']
                chapters[i + 1] = (start, end)
            chapters[len(chapters_found)] = (
                chapters_found.iloc[-1]['token_ID_within_document'], 
                -1
            )
        else:
            # No chapters found, treat whole novel as one unit
            chapters = {1: (0, -1)}
    else:
        chapters = {1: (0, -1)}
    
    # Precompute token->sentence lookup for DC
    tok2sent = tokens_df.set_index("token_ID_within_document")["sentence_ID"].to_dict()
    
    all_results = []
    
    for chapter, (chapter_start, chapter_end) in chapters.items():
        e_df = get_chapter_df(entities_df, 'start_token', chapter_start, chapter_end)
        q_df = get_chapter_df(quotes_df, 'quote_start', chapter_start, chapter_end)
        s_df = get_chapter_df(subjects_df, 'token_ID_within_document', chapter_start, chapter_end)
        
        # N - Name mentions (proper nouns)
        n_counts = e_df[e_df['prop'] == 'PROP'].groupby('character').size().to_dict()
        
        # C - Communication (quote attribution)
        c_counts = q_df.groupby('speaker').size().to_dict()
        
        # I - Interiority (mental verbs as subject)
        i_counts = s_df.groupby('character')['mental'].sum().to_dict()
        
        # A - Action (action verbs as subject)
        a_counts = s_df.groupby('character')['action'].sum().to_dict()
        
        # DN - Description by Narrator (description verbs outside quotes)
        dn_counts = s_df[~s_df['in_quote']].groupby('character')['description'].sum().to_dict()
        
        # DC - Discussion in Character speech
        dc_counts = defaultdict(int)
        for _, row in q_df.iterrows():
            speaker = row['speaker']
            qs, qe = int(row['quote_start']), int(row['quote_end'])
            
            # Find characters mentioned in this quote
            mentioned = find_chars_in_span(qs, qe, e_df)
            mentioned = [(name, start) for name, start in mentioned if name != speaker]
            
            if not mentioned:
                continue
            
            # Group by sentence, count once per sentence per character
            sent2chars = defaultdict(set)
            for name, start_tok in mentioned:
                sent_id = tok2sent.get(int(start_tok))
                if sent_id is not None:
                    sent2chars[sent_id].add(name)
            
            for chars_in_sent in sent2chars.values():
                for name in chars_in_sent:
                    dc_counts[name] += 1
        
        # Build results for all characters
        tag_dicts = {
            'N': n_counts,
            'C': c_counts,
            'I': i_counts,
            'A': a_counts,
            'DC': dict(dc_counts),
            'DN': dn_counts
        }
        
        for char in all_characters:
            for tag, counts in tag_dicts.items():
                count = counts.get(char, 0)
                if isinstance(count, bool):
                    count = int(count)
                
                result = {
                    "character": char,
                    "tag": tag,
                    "count": int(count)
                }
                if by_chapter:
                    result["chapter"] = chapter
                
                all_results.append(result)
    
    return all_results


def process_novel(novel_dir, output_dir, min_mentions=1, by_chapter=True):
    """Process a single novel directory"""
    novel_name = Path(novel_dir).name
    
    # Find the base filename
    ent_files = list(Path(novel_dir).glob("*.entities"))
    if not ent_files:
        print(f"  Skipping {novel_name}: no .entities file")
        return None
    
    base_dir = str(ent_files[0]).replace(".entities", "")
    
    results = tag_novel(base_dir, min_mentions, by_chapter)
    if results is None:
        print(f"  Skipping {novel_name}: could not process")
        return None
    
    # Add novel name to results
    for r in results:
        r["novel"] = novel_name
    
    # Save per-novel results
    df = pd.DataFrame(results)
    out_path = Path(output_dir) / f"{novel_name}_components.csv"
    df.to_csv(out_path, index=False)
    
    # Summary stats
    n_chars = df['character'].nunique()
    n_chapters = df['chapter'].nunique() if 'chapter' in df.columns else 1
    total_tags = df['count'].sum()
    
    return {
        "novel": novel_name,
        "n_characters": n_chars,
        "n_chapters": n_chapters,
        "total_tags": total_tags,
        "N_total": df[df['tag'] == 'N']['count'].sum(),
        "C_total": df[df['tag'] == 'C']['count'].sum(),
        "I_total": df[df['tag'] == 'I']['count'].sum(),
        "A_total": df[df['tag'] == 'A']['count'].sum(),
        "DC_total": df[df['tag'] == 'DC']['count'].sum(),
        "DN_total": df[df['tag'] == 'DN']['count'].sum(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Tag character components across BookNLP corpus")
    parser.add_argument("--corpus-dir", required=True,
                        help="Directory with BookNLP outputs (one folder per novel)")
    parser.add_argument("--output-dir", default="tagging_results",
                        help="Output directory for tagging results")
    parser.add_argument("--min-mentions", type=int, default=1,
                        help="Minimum mentions to include character (default: 1)")
    parser.add_argument("--no-chapters", action="store_true",
                        help="Don't split by chapters, tag whole novel")
    args = parser.parse_args()
    
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    novel_dirs = sorted([d for d in corpus_dir.iterdir() if d.is_dir()])
    print(f"Found {len(novel_dirs)} novels in {corpus_dir}")
    
    all_results = []
    for i, novel_dir in enumerate(novel_dirs):
        print(f"[{i+1}/{len(novel_dirs)}] {novel_dir.name}")
        result = process_novel(
            novel_dir, output_dir, 
            min_mentions=args.min_mentions,
            by_chapter=not args.no_chapters
        )
        if result:
            all_results.append(result)
            print(f"    {result['n_characters']} characters, {result['n_chapters']} chapters, {result['total_tags']} total tags")
    
    # Save corpus summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = output_dir / "tagging_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n{'='*60}")
        print("CORPUS SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully tagged: {len(all_results)} / {len(novel_dirs)} novels")
        print(f"\nAverage per novel:")
        print(f"  Characters: {summary_df['n_characters'].mean():.1f}")
        print(f"  Chapters:   {summary_df['n_chapters'].mean():.1f}")
        print(f"\nTotal tag counts across corpus:")
        print(f"  N (Name):        {summary_df['N_total'].sum():,}")
        print(f"  C (Communication): {summary_df['C_total'].sum():,}")
        print(f"  I (Interiority):   {summary_df['I_total'].sum():,}")
        print(f"  A (Action):        {summary_df['A_total'].sum():,}")
        print(f"  DC (Discussion):   {summary_df['DC_total'].sum():,}")
        print(f"  DN (Description):  {summary_df['DN_total'].sum():,}")
        print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()