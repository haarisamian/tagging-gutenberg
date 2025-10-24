import os, pandas as pd, json, glob
from tqdm import tqdm
from openai import OpenAI
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

client = OpenAI()

def load_sharons_data(file_path):
    sharon_data = pd.read_excel(file_path, sheet_name="ALL INSTANCES")
    return sharon_data

# Function to process a single prompt
def process_prompt(prompt, model = "gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

# Function to split text into chunks
def split_text(text, max_tokens = 1000):
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
    return chunks

# Approach 1: Whole Chapter, All Tags
def tag_whole_chapter(chapter_text, character_set):
    prompt = f"""
    You are an expert in literary analysis. Please analyze the following text and tag all instances of the following characters if they appear in the given chapter:
    {', '.join(character_set)}

    For each character, include all name variants under the normalized form above (e.g., tags for "Lizzy" should be counted under "Elizabeth")

    For each character, provide the following tags:
    - N: Count all named mentions of this person (including variants).
    - A: Count each verb of physical action (exclude speech, thought, and feeling verbs).
    - C: Count blocks of directly quoted dialogue, paraphrased speech, and letters.
    - I: Count each verb expressing thought, feeling, intention, or interpretation.
    - DN: Count sentences describing the character by the narrator.
    - DC: Count sentences where other characters discuss the character.

    Text:
    {chapter_text}

    Return the results as a JSON object with each character as a key and their tags as values. Only include characters that have at least one non-zero tag.
    If a character has all tags equal to zero, omit it from the JSON output entirely.
    
    Strictly output JSON in this format:
    {{
        "Character1": {{"N": count1, "A": countA1, "C": countC1, "I": countI1, "DN": countDN1, "DC": countDC1}},
        "Character2": {{"N": count2, "A": countA2, "C": countC2, "I": countI2, "DN": countDN2, "DC": countDC2}},
        ...
    }}
    """
    response = process_prompt(prompt)
    return json.loads(response)

# Approach 2: Chunk Chapter, All Tags
def tag_chunked_chapter(chapter_text, character_set, max_tokens = 1000):
    '''Process a chapter in chunks and aggregate results.'''
    
    chunks = split_text(chapter_text, max_tokens)
    chunk_results = []
    for i, chunk in enumerate(chunks):
        print(f"Processing Chunk {i+1}...")
        prompt = f"""
        You are an expert in literary analysis. Please analyze the following text and tag all instances of the following characters:
        {', '.join(character_set)}

        For each character, include all name variants under the normalized form above (e.g., tags for "Lizzy" should be counted under "Elizabeth")

        For each character, provide the following tags:
        - N: Count all named mentions of this person (including variants).
        - A: Count each verb of physical action (exclude speech, thought, and feeling verbs).
        - C: Count blocks of directly quoted dialogue, paraphrased speech, and letters.
        - I: Count each verb expressing thought, feeling, intention, or interpretation.
        - DN: Count sentences describing the character by the narrator.
        - DC: Count sentences where other characters discuss the character.

        Text:
        {chunk}

        Return the results as a JSON object with each character as a key and their tags as values. Only include characters that have at least one non-zero tag.
        If a character has all tags equal to zero, omit it from the JSON output entirely.
        
        Strictly output JSON in this format:
        {{
            "Character1": {{"N": count1, "A": countA1, "C": countC1, "I": countI1, "DN": countDN1, "DC": countDC1}},
            "Character2": {{"N": count2, "A": countA2, "C": countC2, "I": countI2, "DN": countDN2, "DC": countDC2}},
            ...
        }}
        """
        response = process_prompt(prompt)
        chunk_results.append(json.loads(response))
    
    # Sum results across chunks (robust to missing tags)
    combined_results = {}
    for chunk_result in chunk_results:
        for character, tags in chunk_result.items():
            if character not in combined_results:
                combined_results[character] = {}
            for tag, cnt in tags.items():
                combined_results[character][tag] = combined_results[character].get(tag, 0) + cnt

    # Ensure all six tags exist for every character
    expected_tags = ["N", "A", "C", "I", "DN", "DC"]
    for char in combined_results:
        for t in expected_tags:
            combined_results[char].setdefault(t, 0)
    return combined_results
    
# Approach 3: Whole Chapter, One Tag at a Time
def tag_whole_chapter_one_tag(chapter_text, character_set, tag):
    descriptions = {"N": "Count all named mentions of this person (including variants).",
                    "A": "Count each verb of physical action (exclude speech, thought, and feeling verbs).",
                    "C": "Count blocks of directly quoted dialogue, paraphrased speech, and letters.",
                    "I": "Count each verb expressing thought, feeling, intention, or interpretation.",
                    "DN": "Count sentences describing the character by the narrator.",
                    "DC": "Count sentences where other characters discuss the character."
                    }
    
    prompt = f"""
    You are an expert in literary analysis. Please analyze the following text and tag all instances of the following characters:
    {', '.join(character_set)}

    For each character, include all name variants under the normalized form above (e.g., tags for "Lizzy" should be counted under "Elizabeth")
    
    Provide the counts for the '{tag}' tag only.
    
    The tag '{tag}' is defined as:
    - {descriptions[tag]}
    
    Text:
    {chapter_text}

    Return the results as a JSON object with each character as a key and its '{tag}' count as value. Only include characters that have a non-zero count.
    If a character has '{tag}' count equal to zero, omit it from the JSON output entirely.
    
    Strictly output JSON in this format:
    {{
        "Character1": count1,
        "Character2": count2,
        ...
    }}
    """
    response = process_prompt(prompt)
    return json.loads(response)

def tag_whole_chapter_all_tags_separately(chapter_text, character_set):
    tags = ["N", "A", "C", "I", "DN", "DC"]
    combined_results = {}
    for tag in tags:
        print(f"Processing Tag: {tag}")
        tag_results = tag_whole_chapter_one_tag(chapter_text, character_set, tag)
        for character, count in tag_results.items():
            if character not in combined_results:
                combined_results[character] = {t: 0 for t in tags}
            try:
                combined_results[character][tag] = int(count)
            except Exception:
                combined_results[character][tag] = 0
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

if __name__ == "__main__":
    sharon_data = load_sharons_data("data/manual/pride_and_prejudice.xlsx")
    character_set = sharon_data['Character'].unique().tolist()

    chapter_dir = "data/chapters/"
    # Load chapters (example: list of chapter texts)
    chapters = []
    for chapter_file in sorted(glob.glob(os.path.join(chapter_dir, "pride_and_prejudice_*.txt"))):
        with open(chapter_file, 'r') as f:
            chapter_text = f.read()
            chapters.append(chapter_text)
    all_results = []

    for i, chapter_text in enumerate(chapters):
        print(f"Processing Chapter {i+1}...")

        # Run all three approaches
        whole = normalize_results(tag_whole_chapter(chapter_text, character_set), character_set)
        chunked = normalize_results(tag_chunked_chapter(chapter_text, character_set), character_set)
        separate = normalize_results(tag_whole_chapter_all_tags_separately(chapter_text, character_set), character_set)

        for method_name, results in [("whole", whole), ("chunked", chunked), ("separate", separate)]:
            for character, tags in results.items():
                for tag, count in tags.items():
                    all_results.append({
                        "chapter": i + 1,
                        "method": method_name,
                        "character": character,
                        "tag": tag,
                        "count": count
                    })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"results/model_results_all_chapters_{datetime.now().strftime('%H%M')}.csv", index=False)