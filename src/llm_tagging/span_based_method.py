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
import re
import nltk
import argparse

client = OpenAI()

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

def tag_text_N(text, character_set, model):
    prompt = f"""
    Please analyze the following text for mentions of the following characters:
    {', '.join(character_set)}

    For each character, list all variants of their name that are used to refer to them in the text. Do not include pronouns, only proper names. If they are referred to with and without a title, include both variants.

    Text:
    {text}

    Return the results as a JSON object with each character as a key and the list of name variants as the value.

    Strictly output JSON in this format:
    {{
        "Character1": [name1, name2, ...],
        "Character2": [name1, ...],
        ...
    }}
    """
    spans = []
    start = time.time()
    response = process_prompt_json(prompt, model)
    end = time.time()
    temp = defaultdict(int)
    search_text = re.sub(r'\s+', ' ', text).lower()
    for character, names in json.loads(response).items():
        if character in character_set:
            new_names = []
            for name in names:
                found = False
                for x in names:
                    if name.lower() in x.lower() and name.lower() != x.lower():
                        found = True
                if not found:
                    new_names.append(name)
            for name in new_names:
                temp[character] += search_text.count(name.lower())
                spans.append({
                    'text': name,
                    'character': character,
                    'count': search_text.count(name.lower()),
                })
    json_response = {'counts': temp}
    json_response['input_tokens'] = estimate_tokens(prompt)
    json_response['output_tokens'] = estimate_tokens(response)
    json_response['elapsed_time'] = end - start
    return json_response, spans

def tag_text_A(text, character_set, model):
    prompt = f"""
    Please analyze the following text for actions performed by the following characters:
    {', '.join(character_set)}

    For each character, list all physical actions they perform. Do not include acts of talking, speaking, asking questions, or writing. Do not include acts of thinking or feeling.

    Text:
    {text}

    Return the results as a JSON object with each character as a key and the list of actions as the value.

    Strictly output JSON in this format:
    {{
        "Character1": [phrase1, phrase2, ...],
        "Character2": [phrase1, ...],
        ...
    }}
    """
    spans = []
    start = time.time()
    response = process_prompt_json(prompt, model)
    end = time.time()
    temp = defaultdict(int)
    for character, actions in json.loads(response).items():
        if character in character_set:
            temp[character] = len(actions)
            for action in actions:
                spans.append({
                    'text': action,
                    'character': character,
                })
    json_response = {'counts': temp}
    json_response['input_tokens'] = estimate_tokens(prompt)
    json_response['output_tokens'] = estimate_tokens(response)
    json_response['elapsed_time'] = end - start
    return json_response, spans

def tag_text_I(text, character_set, model):
    dialogue_pattern = r'["“](.*?)(?=["”]|\n\n|$)'  # add ‘ and ’ if needed
    dialogue_turns = re.findall(dialogue_pattern, text, re.DOTALL)
    dialogue_turns = [d.replace('\n', ' ').strip() for d in dialogue_turns if d.strip()]
    prompt = f"""
    Please analyze the following text for descriptions of the feelings and thoughts of the following characters:
    {', '.join(character_set)}

    For each character, list all instances where the narration shows us the characters thoughts, feelings, intentions, or perceptions.

    Text:
    {text}

    Return the results as a JSON object with each character as a key and the list of thoughts/feelings as the value. Do not include instances of the character speaking, only descriptions from the narrator.

    Strictly output JSON in this format:
    {{
        "Character1": [phrase1, phrase2, ...],
        "Character2": [phrase1, ...],
        ...
    }}
    """
    spans = []
    start = time.time()
    response = process_prompt_json(prompt, model)
    end = time.time()
    temp = defaultdict(int)
    updated_response = defaultdict(list)
    for character, glimpses in json.loads(response).items():
        for glimpse in glimpses:
            if sum([glimpse in d for d in dialogue_turns]) == 0:
                if character in character_set:
                    updated_response[character].append(glimpse)
                    temp[character] += 1
                    spans.append({
                        'text': glimpse,
                        'character': character,
                    })
    json_response = {'counts': temp}
    json_response['input_tokens'] = estimate_tokens(prompt)
    json_response['output_tokens'] = estimate_tokens(response)
    json_response['elapsed_time'] = end - start
    return json_response, spans

def tag_text_C(text, character_set, model):
    dialogue_pattern = r'["“](.*?)(?=["”]|\n\n|$)'  # add ‘ and ’ if needed
    dialogue_turns = re.findall(dialogue_pattern, text, re.DOTALL)
    dialogue_turns = [d.replace('\n', ' ').strip() for d in dialogue_turns if d.strip()]
    temp = defaultdict(int)
    json_response = defaultdict(float)
    spans, speakers = [], []
    for idx, turn in enumerate(dialogue_turns):
        prompt = f"""
        Please analyze the dialogue and letters in the following text.
        
        Full text for context:
        {text}
        
        Dialogue turn (or letter):{turn}
        
        Character list: {character_set}
        
        Which character from the character list is the speaker (or writer) of this dialogue turn (or letter)?
        
        Give just the character name as your response.
        
        Answer:"""
        start = time.time()
        response = process_prompt_reg(prompt, model)
        end = time.time()
        if response.strip() in character_set:
            temp[response.strip()] += 1
            spans.append({
                'turn_id': idx,
                'text': turn,
                'character': response.strip(),
            })
            speakers.append(response.strip())
        else:
            speakers.append('')
        json_response['input_tokens'] += estimate_tokens(prompt)
        json_response['output_tokens'] += estimate_tokens(response)
        json_response['elapsed_time'] += end - start

    json_response['counts'] = temp
    return json_response, spans, dialogue_turns, speakers


def tag_text_DC(text, character_set, model, dialogue_turns, speakers):
    temp = defaultdict(int)
    json_response = defaultdict(float)
    spans = []
    chapter_sents = nltk.sent_tokenize(text.replace('\t', ''))
    for idx, turn in tqdm(enumerate(dialogue_turns), total=len(dialogue_turns), ncols=70):
        for sent in nltk.sent_tokenize(turn):
            clean = re.sub(r'[^A-Za-z]+', '', sent)
            if clean == "":
                continue
            doc_idx = [i for i in range(len(chapter_sents)) if sent in chapter_sents[i].replace('\n', ' ')]
            if len(doc_idx) == 0:
                chunk = chapter_text[chapter_text.find(sent)-100:chapter_text.find(sent)] + sent
            else:
                doc_idx = doc_idx[0]
                chunk = ""
                if doc_idx > 5:
                    chunk += ' '.join(chapter_sents[doc_idx-5:doc_idx+1])
                else:
                    chunk += ' '.join(chapter_sents[:doc_idx+1])

            prompt = f"""
            Please analyze the following dialogue for mentions of the following characters:
            {', '.join(character_set)}
                
            Surrounding text for context:
            {chunk}
            
            Dialogue sentence (speaker: {speakers[idx]}):
            {sent}
            
            Resolve any pronouns, relational mentions, or name variants in the dialogue sentence to the proper character name.
            
            Give your response as a JSON object with pronouns/mentions/names as the key and the character it refers to as the value in this format:
            {{
                "pronoun": character1,
                "pronoun": character2,
                ...
            }}
            """
            start = time.time()
            response = process_prompt_json(prompt, model)
            end = time.time()
            temp_names = []
            clean = re.sub(r'\s+', ' ', sent)
            for pronoun, character in json.loads(response).items():
                if character != speakers[idx] and pronoun in clean:
                    if str(character) in character_set and str(character) not in temp_names:
                        temp[character] += 1
                        spans.append({
                            'turn_id': idx,
                            'text': sent,
                            'speaker': speakers[idx],
                            'character': character
                        })
                        temp_names.append(character)
            for character in character_set:
                if character.lower() in clean.lower() and character not in temp_names:
                    temp[character] += 1
                    spans.append({
                        'turn_id': idx,
                        'text': sent,
                        'speaker': speakers[idx],
                        'character': character
                    })
                    temp_names.append(character)
            json_response['input_tokens'] += estimate_tokens(prompt)
            json_response['output_tokens'] += estimate_tokens(response)
            json_response['elapsed_time'] += end - start

    json_response['counts'] = temp
    return json_response, spans


def tag_text_DN(text, character_set, model, dialogue_turns):
    temp = defaultdict(int)
    json_response = defaultdict(float)
    dialogue = ' [/] '.join(dialogue_turns)
    prompt = f"""
    Please analyze the following text for places where the narrator describes the following characters:
    {', '.join(character_set)}

    For each character, list all instances where the narration describes something about the character's looks, manner, or dress. If a character is not described, do not include them.

    Text:
    {text}

    Return the results as a JSON object with each character as a key and the list of descriptions as the value. Do not include instances of the character speaking, only descriptions from the narrator.

    Strictly output JSON in this format:
    {{
        "Character1": [phrase1, phrase2, ...],
        "Character2": [phrase1, ...],
        ...
    }}
    """
    start = time.time()
    response = process_prompt_json(prompt, model)
    end = time.time()
    spans = []
    for character, mentions in json.loads(response).items():
        if character in character_set:
            for mention in mentions:
                if mention not in dialogue:
                    temp[character] += 1
                    spans.append({
                        'character': character,
                        'mention': mention
                    })
    json_response['input_tokens'] += estimate_tokens(prompt)
    json_response['output_tokens'] += estimate_tokens(response)
    json_response['elapsed_time'] += end - start
    json_response['counts'] = temp
    return json_response, spans

def tag_text(text, character_set):
    tags = ["N", "A", "C", "I", "DN", "DC"]
    model = 'gpt-4o-mini'
    combined_results = {'counts':{}}
    spans = {}
    dialogue_turns, speakers = None, None
    for tag in tags:
        print(f"Processing Tag: {tag}")
        if tag == 'N':
            tag_results, spans['N'] = tag_text_N(text, character_set, model)
        elif tag == 'A':
            tag_results, spans['A'] = tag_text_A(text, character_set, model)
        elif tag == 'C':
            tag_results, spans['C'], dialogue_turns, speakers = tag_text_C(text, character_set, model)
        elif tag == 'I':
            tag_results, spans['I'] = tag_text_I(text, character_set, model)
        elif tag == 'DC':
            tag_results, spans['DC'] = tag_text_DC(text, character_set, model, dialogue_turns, speakers)
        elif tag == 'DN':
            tag_results, spans['DN'] = tag_text_DN(text, character_set, model, dialogue_turns)
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

    return combined_results, spans


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

def get_character_set(booktitle, text, character_set):
    prompt = f"""
    {booktitle} section: {text}
    
    Character list: {', '.join(character_set)}
    
    Which characters from the character list are present in the {booktitle} section? Include any character that are mentioned or referred to, even if they are not physically present in the scenes.
    
    Strictly output a list of the characters who are present in this format: [character1, character2, etc.]
    """
    present_chars = process_prompt_reg(prompt)
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
        gold_data = pd.read_csv("data/manual/JaneEyreTagging_January11.csv")
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

    all_results = []
    all_timing = []
    all_spans = defaultdict(list)
    if os.path.exists(f"{datadir}/llm_spans.csv"):
        all_results = pd.read_csv(f"{datadir}/llm_spans.csv").to_dict(orient='records')
        all_timing = pd.read_csv(f"{datadir}/llm_spans_timing.csv").to_dict(orient='records')
    for i, chapter_text in chapters.items():
        if i in [x['chapter'] for x in all_timing]:
            continue
        print(f"Processing Chapter {i}...")
        curr_character_set = get_character_set(booktitle, chapter_text, character_set)

        chapter_results, spans = tag_text(chapter_text, curr_character_set)

        final_results = {}
        final_timing = defaultdict(list)
        for key, val in chapter_results.items():
            if key == 'counts':
                for character, tags in val.items():
                    if character not in final_results:
                        final_results[character] = {}
                    for tag, cnt in tags.items():
                        final_results[character][tag] = final_results[character].get(tag, 0) + cnt
            else:
                final_timing[key].extend(val)

        final_results = normalize_results(final_results, curr_character_set)
        for character, tags in final_results.items():
            for tag, count in tags.items():
                all_results.append({
                    "chapter": i,
                    "method": 'span-based',
                    "character": character,
                    "tag": tag,
                    "count": count,
                })

        for tag in ["N", "A", "C", "I", "DN", "DC"]:
            for span in spans[tag]:
                temp = {
                    "chapter": i,
                    "method": 'span-based',
                }
                for key, val in span.items():
                    temp[key] = val
                all_spans[tag].append(temp)

        temp = {
            "chapter": i,
            "method": 'span-based',
        }
        for key in ['input_tokens', 'output_tokens', 'elapsed_time']:
            temp[key] = sum(final_timing[key])
        all_timing.append(temp)

        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"{datadir}/llm_spans.csv", index=False)
        results_df = pd.DataFrame(all_timing)
        results_df.to_csv(f"{datadir}/llm_spans_timing.csv", index=False)
        for tag in ["N", "A", "C", "I", "DN", "DC"]:
            results_df = pd.DataFrame(all_spans[tag])
            results_df.to_csv(f"{datadir}/llm_spans_{tag}.csv", index=False)