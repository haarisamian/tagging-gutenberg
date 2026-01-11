import os, pandas as pd, json, glob
from tqdm import tqdm
from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
from collections import defaultdict
import tiktoken
import argparse

client = OpenAI()

# Settings to compare
SETTINGS = {
    'chunk_size': ['1000', 'chapter'],
    'characters': ['all', 'individual'],
    'LLM': ['gpt-4o-mini'],
    'components': ['all', 'individual']
}
COMBOS = [{
    'chunk_size': c,
    'characters': k,
    'LLM': l,
    'components': s
} for c in SETTINGS['chunk_size'] for k in SETTINGS['characters'] for l in SETTINGS['LLM'] for s in SETTINGS['components']]

# Estimate token-length of a prompt or response
def estimate_tokens(text, model="gpt-4o-mini"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# Process a single prompt in JSON format
def process_prompt_json(prompt, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in literary analysis."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content

# Process a single prompt regularly
def process_prompt_reg(prompt, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in literary analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.,
        top_p=1.
    )
    return response.choices[0].message.content


# Split text into chunks of length max_tokens
def split_text(text, max_tokens=3000):
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_tokens:
            current_chunk += paragraph + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
    if current_chunk:
        chunks.append(current_chunk.strip())
    return [c for c in chunks if len(c) > 0]

# All Tags
def tag_text_all_tags(text, character_set, model):
    prompt = f"""
    Please analyze the following text and tag all instances of the following characters:
    {', '.join(character_set)}

    For each character, include all name variants under the normalized form above (e.g., tags for "Lizzy" should be counted under "Elizabeth")

    For each character, provide the following tags:
    - N: Count all named mentions of this person (including name variants).
    - A: Count each verb of physical action (exclude speech, thought, and feeling verbs).
    - C: Count blocks of directly quoted dialogue, paraphrased speech, and letters.
    - I: Count each verb expressing thought, feeling, intention, or interpretation.
    - DN: Count sentences describing the character by the narrator.
    - DC: Count sentences where other characters discuss the character.
    
    Text:
    {text}

    Return the results as a JSON object with each character as a key and their tags as values. Only include characters that have at least one non-zero tag.
    If a character has all tags equal to zero, omit it from the JSON output entirely.

    Strictly output JSON in this format:
    {{
        "Character1": {{"N": count1, "A": countA1, "C": countC1, "I": countI1, "DN": countDN1, "DC": countDC1}},
        "Character2": {{"N": count2, "A": countA2, "C": countC2, "I": countI2, "DN": countDN2, "DC": countDC2}},
        ...
    }}
    """

    start = time.time()
    response = process_prompt_json(prompt, model)
    end = time.time()
    json_response = {'counts': json.loads(response)}
    json_response['input_tokens'] = [estimate_tokens(prompt)]
    json_response['output_tokens'] = [estimate_tokens(response)]
    json_response['elapsed_time'] = [end - start]
    return json_response


# Approach 3: Whole Chapter, One Tag at a Time
def tag_text_one_tag(text, character_set, tag, model):
    descriptions = {"N": "Count all named mentions of this person (including name variants).",
                    "A": "Count each verb of physical action (exclude speech, thought, and feeling verbs).",
                    "C": "Count blocks of directly quoted dialogue, paraphrased speech, and letters.",
                    "I": "Count each verb expressing thought, feeling, intention, or interpretation.",
                    "DN": "Count sentences describing the character by the narrator.",
                    "DC": "Count sentences where other characters discuss the character."
                    }

    prompt = f"""
    Please analyze the following text and tag all instances of the following characters:
    {', '.join(character_set)}

    For each character, include all name variants under the normalized form above (e.g., tags for "Lizzy" should be counted under "Elizabeth")

    Provide the counts for the '{tag}' tag only.

    The tag '{tag}' is defined as:
    - {descriptions[tag]}
    
    Text:
    {text}

    Return the results as a JSON object with each character as a key and its '{tag}' count as value. Only include characters that have a non-zero count.
    If a character has '{tag}' count equal to zero, omit it from the JSON output entirely.

    Strictly output JSON in this format:
    {{
        "Character1": count1,
        "Character2": count2,
        ...
    }}
    """

    start = time.time()
    response = process_prompt_json(prompt, model)
    end = time.time()
    json_response = {'counts': json.loads(response)}
    json_response['input_tokens'] = estimate_tokens(prompt)
    json_response['output_tokens'] = estimate_tokens(response)
    json_response['elapsed_time'] = end - start
    return json_response

def tag_text(text, character_set, setting):
    tags = ["N", "A", "C", "I", "DN", "DC"]
    model = setting['LLM']
    components = setting['components']
    if components == 'individual':
        combined_results = {'counts':{}}
        for tag in tags:
            print(f"Processing Tag: {tag}")
            tag_results = tag_text_one_tag(text, character_set, tag, model)
            for key, val in tag_results.items():
                if key == 'counts':
                    for character, count in val.items():
                        if character not in combined_results['counts']:
                            combined_results['counts'][character] = {t: 0 for t in tags}
                        combined_results['counts'][character][tag] = int(count)
                else:
                    if key not in combined_results:
                        combined_results[key] = []
                    combined_results[key].append(val)
    else:
        combined_results = tag_text_all_tags(text, character_set, model)
    return combined_results


def normalize_results(results, character_set):
    """Ensure all characters and all six tags are present with integer counts."""
    EXPECTED_TAGS = ["N", "A", "C", "I", "DN", "DC"]
    out = {}
    for ch in character_set:
        ch_dict = results.get(ch, {}) or {}
        norm = {}
        for t in EXPECTED_TAGS:
            val = ch_dict.get(t, 0)
            try:
                val = int(val)
            except Exception:
                val = 0
            norm[t] = val
        out[ch] = norm
    return out

def get_character_set(booktitle, text, character_set, setting):
    prompt = f"""
    {booktitle} section: {text}
    
    Character list: {', '.join(character_set)}
    
    Which characters from the character list are present in the {booktitle} section? Include any character that are mentioned or referred to, even if they are not physically present in the scenes.
    
    Strictly output a list of the characters who are present in this format: [character1, character2, etc.]
    """
    present_chars = process_prompt_reg(prompt, setting['LLM'])
    curr_character_set = set([x for x in character_set if x in present_chars])
    return curr_character_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", required=True, type=str) # 'pride' or 'jane'
    args = parser.parse_args()

    if args.book == 'pride':
        gold_data = pd.read_excel("data/manual/pride_and_prejudice.xlsx", sheet_name="ALL INSTANCES")
        character_set = gold_data['Character'].unique().tolist()
        booktitle = "Pride and Prejudice"
        datadir = "data/pride_and_jane/Pride_and_Prejudice"
        file_start = "pride_and_prejudice"
    elif args.book == 'jane':
        gold_data = pd.read_csv("data/manual/Jane_Eyre_tagging.csv")
        character_set = gold_data['CHARACTER'].unique().tolist()
        booktitle = "Jane Eyre"
        datadir = "data/pride_and_jane/Jane_Eyre"
        file_start = "chapter"

    # Load chapters
    chapters = {}
    for chapter_file in glob.glob(os.path.join(datadir, "chapters", f"{file_start}_*.txt")):
        chap_num = chapter_file.split('_')[-1][:-4]
        with open(chapter_file, 'r') as f:
            chapter_text = f.read()
            chapters[int(chap_num)] = chapter_text

    for setting in COMBOS:
        print('SETTING ', setting)
        filename = '_'.join(setting.values())
        all_results = []
        all_timing = []
        if os.path.exists(f"{datadir}/{filename}_full.csv"):
            all_results = pd.read_csv(f"{datadir}/{filename}_full.csv").to_dict(orient='records')
            all_timing = pd.read_csv(f"{datadir}/{filename}_full_timing").to_dict(orient='records')
        for i, chapter_text in chapters.items():
            if i in [x['chapter'] for x in all_timing]:
                continue
            print(f"Processing Chapter {i}...")
            curr_character_set = get_character_set(booktitle, chapter_text, character_set, setting)
            ccs = [curr_character_set]
            if setting['characters'] == 'individual':
                ccs = [[x] for x in curr_character_set]

            chunks = [chapter_text]
            if setting['chunk_size'] != 'chapter':
                chunks = split_text(chapter_text, int(setting['chunk_size']))

            combined_results = {}
            combined_timing = defaultdict(list)
            for chunk in tqdm(chunks, ncols=70):
                for cc in ccs:
                    chunk_results = tag_text(chunk, cc, setting)

                    for key, val in chunk_results.items():
                        if key == 'counts':
                            for character, tags in val.items():
                                if character not in combined_results:
                                    combined_results[character] = {}
                                for tag, cnt in tags.items():
                                    combined_results[character][tag] = combined_results[character].get(tag, 0) + cnt
                        else:
                            combined_timing[key].extend(val)

            combined_results = normalize_results(combined_results, curr_character_set)
            for character, tags in combined_results.items():
                for tag, count in tags.items():
                    all_results.append({
                        "chapter": i,
                        "method": '_'.join(setting.values()),
                        "character": character,
                        "tag": tag,
                        "count": count,
                    })

            temp = {
                "chapter": i,
                "method": '_'.join(setting.values()),
            }
            for key in ['input_tokens', 'output_tokens', 'elapsed_time']:
                temp[key] = sum(combined_timing[key])
            all_timing.append(temp)

            results_df = pd.DataFrame(all_results)
            results_df.to_csv(f"{datadir}/{filename}_llm_tags.csv", index=False)
            results_df = pd.DataFrame(all_timing)
            results_df.to_csv(f"{datadir}/{filename}_llm_tags_timing", index=False)