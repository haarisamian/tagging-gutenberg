import pandas as pd
import json
from openai import OpenAI
import os
from collections import Counter, defaultdict
from nltk.corpus import verbnet


client = OpenAI()

def process_prompt_reg(prompt, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "developer",  "content": "You are an expert in literary analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.,
        top_p=1.
    )
    return response.choices[0].message.content

def find_char(tok_id, people_df):
    temp = people_df.loc[(people_df["start_token"] <= tok_id) & (people_df["end_token"] >= tok_id), 'merged']
    if len(temp) == 0:
        temp = ""
    else:
        temp = temp.iloc[0]
    return temp

def in_quote(tok_id, quotes_df):
    temp = quotes_df.loc[(quotes_df["quote_start"] <= tok_id) & (quotes_df["quote_end"] >= tok_id), 'merged']
    return len(temp) > 0

def find_chars(start, end, people_df):
    temp = people_df.loc[(start <= people_df['start_token']) & (people_df["end_token"] <= end), 'merged']
    if len(temp) == 0:
        temp = []
    else:
        temp = list(temp)
    return temp

def char_map(df, key1, key2, characters, mention_map, sharon_chars):
    df['name'] = df[key1].map(lambda x: characters[x]['canonical_name'] if x in characters else '')
    df['name'] = df['name'].map(lambda x: mention_map[x] if x in mention_map else x)
    df['sharon'] = df[key2].map(lambda x: x.lower() if x in sharon_chars else '')
    df['merged'] = df.apply(lambda r: r["sharon"] if r['sharon'] != '' else r['name'], axis=1)
    df = df[df['merged'] != '']
    return df

def is_communicative(verb):
    classes = verbnet.classids(verb)
    for cid in classes:
        if any(k in cid.lower() for k in ['37.', 'judgment']) and verb != "read":
            return True
    return False

def is_mental(verb):
    classes = verbnet.classids(verb)
    for cid in classes:
        if any(k in cid.lower() for k in ['conjecture', 'consider', 'see-30.1-1', 'want', 'long', 'wish', 'care', 'comprehend', 'empathize', 'admire', 'marvel', 'characterize']):
            return True
    return False

def is_mental_ss(ss):
    for t in ['cognition', 'emotion', 'feeling']:
        if t in ss:
            return True
    return False

def is_communicative_ss(ss):
    for t in ['communication']:
        if t in ss:
            return True
    return False

def make_mention_map(entities_df, sharon_chars, book):
    mention_map = {}
    for c in list(entities_df['name'].value_counts().keys()):
        if c == "" or c in [x.lower() for x in sharon_chars]:
            continue
        temp = [x for x in sharon_chars if c in x.lower()]
        if len(temp) > 0:
            mention_map[c] = temp[0].lower()
            continue
        print(f"Processing {c}")
        prompt = f"""
        Consider the full text of {book} and all the characters it contains. Each character can be referred to by multiple different variants of ther name. Is {c} a character in {book}? Answer yes or no.

        Answer:"""
        answer = process_prompt_reg(prompt).strip().lower().startswith('yes')
        print("answer", answer)
        if answer:
            prompt = f"""
            Consider the following list of characters in {book}:
            {sharon_chars}

            Which character from this list does the name {c} refer to? Answer with just the character name.

            Answer:"""
            mention_map[c] = process_prompt_reg(prompt).strip().lower()
            print("map", mention_map[c])
    return mention_map

def get_tag_dict(sharon_chars, mention_counts):
    try:
        mention_counts.sort_values(ascending=False)
    except Exception as e:
        pass
    booknlp_chars = list(mention_counts.keys())
    final_dict = {}
    for name in sharon_chars:
        temp = 0
        if name.lower() in booknlp_chars:
            temp = int(mention_counts[name.lower()])
        final_dict[name] = temp
    return final_dict

def get_chapter(df, key, chapter_start, chapter_end):
    return df[(df[key] >= chapter_start) & (df[key] <= chapter_end)]

if __name__ == "__main__":
    # Load custom files
    datadir = "booknlp/jane"
    book = "Jane Eyre"
    file_start = 'jane_eyre'
    chapter_str = 'CHAPTER'
    # sharon_data = pd.read_excel("data/manual/pride_and_prejudice.xlsx", sheet_name="ALL INSTANCES")
    # sharon_data["chapter"] = pd.to_numeric(sharon_data["Graph Chapter"], errors="coerce").astype("Int64")
    # sharon_data = sharon_data.drop(columns=["Volume", "B", "FID", "Chapter", "Graph Chapter"])
    sharon_data = pd.read_csv("data/manual/Jane_Eyre_tagging.csv")
    sharon_data["chapter"] = pd.to_numeric(sharon_data["CHAPTER"], errors="coerce").astype("Int64")
    sharon_data = sharon_data.drop(columns=["B", "FID", "CHAPTER"]).rename(columns={"CHARACTER": "Character"})
    sharon_chars = list(sharon_data['Character'].value_counts().keys())

    # Load characters
    characters = {}
    with open(f'{datadir}/{file_start}.characters.json', 'r') as f:
        raw_characters = json.load(f)
    for character in raw_characters['characters']:
        characters[character['character_id']] = character

    # Load entities
    entities_df = pd.read_csv(f'{datadir}/{file_start}.entities', sep='\t')
    entities_df = entities_df[entities_df['cat'] == 'PER']
    entities_df['name'] = entities_df['COREF'].map(lambda x: characters[x]['canonical_name'] if x in characters else '')
    
    mention_map = {}
    if os.path.exists(f'{datadir}/mention_map.json'):
        with open(f'{datadir}/mention_map.json', 'r') as f:
            mention_map = json.load(f)
    else:
        mention_map = make_mention_map(entities_df[entities_df['prop'] == 'PROP'], sharon_chars, book)
        with open(f'{datadir}/mention_map.json', 'w') as f:
            json.dump(mention_map, f)

    if book == 'Jane Eyre':
        characters[0]['canonical_name'] = 'jane'
    entities_df = char_map(entities_df, 'COREF', 'text', characters, mention_map, sharon_chars)
    entities_df['count'] = 1

    # Load quotes
    quotes_df = pd.read_csv(f'{datadir}/{file_start}.quotes', sep='\t')
    quotes_df = char_map(quotes_df, 'char_id', 'mention_phrase', characters, mention_map, sharon_chars)
    quotes_df['count'] = 1
    quotes_df['discussed'] = quotes_df.apply(lambda row: find_chars(row['quote_start'], row['quote_end'], entities_df), axis=1)
    quotes_df['discussed'] = quotes_df.apply(lambda row: [x for x in row['discussed'] if x != row['merged']], axis=1)

    # Load tokens
    tokens_df = pd.read_csv(f'{datadir}/{file_start}.tokens', sep='\t', quoting=3)
    supersense_df = pd.read_csv(f'{datadir}/{file_start}.supersense', sep='\t')
    subjects_df = tokens_df[(tokens_df['dependency_relation'] == 'nsubj') | (tokens_df['dependency_relation'] == 'nsubjpass')]
    subjects_df = subjects_df[subjects_df['POS_tag'].isin(['NOUN', 'PRON', 'PROPN'])]
    subjects_df['character'] = subjects_df['token_ID_within_document'].apply(lambda x: find_char(x, entities_df))
    subjects_df = subjects_df[subjects_df['character'] != '']
    subjects_df['verb_lemma'] = subjects_df['syntactic_head_ID'].apply(lambda x: tokens_df.loc[x, 'lemma'])
    subjects_df['verb_token'] = subjects_df['syntactic_head_ID'].apply(lambda x: tokens_df.loc[x, 'token_ID_within_document'])
    subjects_df['supersense'] = subjects_df['verb_token'].apply(
        lambda x: supersense_df[supersense_df['start_token'] == x]['supersense_category'].iloc[0] if len(supersense_df[supersense_df['start_token'] == x]['supersense_category']) > 0 else '')
    #subjects_df['mental'] = subjects_df['verb_lemma'].apply(lambda x: is_mental(x))
    #subjects_df['mental'] = subjects_df['supersense'].apply(lambda x: is_mental_ss(x))
    subjects_df['mental'] = subjects_df.apply(lambda x: is_mental_ss(x['supersense']) or is_mental(x['verb_lemma']), axis=1)
    subjects_df['communicative'] = subjects_df.apply(lambda x: is_communicative_ss(x['supersense']) or is_communicative(x['verb_lemma']), axis=1)
    # subjects_df['action'] = subjects_df['verb_lemma'].apply(
    #     lambda x: not is_mental(x) and not is_communicative(x) and x != 'be' and len(verbnet.classids(x)) > 0)
    #subjects_df['action'] = subjects_df['supersense'].apply(
    #    lambda x: not is_mental_ss(x) and not is_communicative_ss(x) and 'stative' not in x)
    subjects_df['action'] = subjects_df.apply(
        lambda x: not x['communicative'] and not x['mental'] and 'stative' not in x['supersense'], axis=1)
    subjects_df['description'] = subjects_df['verb_lemma'].apply(lambda x: x == 'be')
    #subjects_df['description'] = subjects_df['supersense'].apply(lambda x: 'stative' in x)
    subjects_df['in_quote'] = subjects_df['token_ID_within_document'].apply(lambda x: in_quote(x, quotes_df))

    # Get chapters
    chapters_df = tokens_df[tokens_df['word'] == chapter_str]
    print("Found", len(chapters_df), "chapters")
    chapters = {}
    for i in range(len(chapters_df)-1):
        start = chapters_df.iloc[i]['token_ID_within_document']
        end = chapters_df.iloc[i+1]['token_ID_within_document']
        chapters[i+1] = (start, end)
    chapters[i+2] = (chapters_df.iloc[i+1]['token_ID_within_document'], -1)

    all_results = []
    for chapter, (chapter_start, chapter_end) in chapters.items():
        e_df = get_chapter(entities_df, 'start_token', chapter_start, chapter_end)
        q_df = get_chapter(quotes_df, 'quote_start', chapter_start, chapter_end)
        s_df = get_chapter(subjects_df, 'token_ID_within_document', chapter_start, chapter_end)

        all_dicts = {}

        # N dict
        mention_counts = e_df[e_df['prop'] == 'PROP'].groupby('merged')['count'].sum()
        all_dicts['N'] = get_tag_dict(sharon_chars, mention_counts)

        # C dict
        mention_counts = q_df.groupby('merged')['count'].sum()
        all_dicts['C'] = get_tag_dict(sharon_chars, mention_counts)

        # I dict
        mention_counts = s_df.groupby('character')['mental'].sum()
        all_dicts['I'] = get_tag_dict(sharon_chars, mention_counts)

        # A dict
        mention_counts = s_df[~s_df['in_quote']].groupby('character')['action'].sum()
        all_dicts['A'] = get_tag_dict(sharon_chars, mention_counts)

        # DC dict
        temp_counts = defaultdict(int)
        for idx, row in q_df.iterrows():
            for name in row['discussed']:
                temp_counts[name] += 1
        all_dicts['DC'] = get_tag_dict(sharon_chars, temp_counts)

        # DN dict
        mention_counts = s_df[~s_df['in_quote']].groupby('character')['description'].sum()
        all_dicts['DN'] = get_tag_dict(sharon_chars, mention_counts)

        for tag, d in all_dicts.items():
            for character, count in d.items():
                all_results.append({
                    "chapter": chapter,
                    "method": "booknlp",
                    "character": character,
                    "tag": tag,
                    "count": count
                })
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{datadir}/{file_start}_booknlp_results.csv", index=False)

