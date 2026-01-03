# import requests
# import pdfplumber

# # grab PDF for story of a modern woman by Dixon
# r = requests.get('http://www.searchengine.org.uk/ebooks/91/96.pdf')
# with open('book.pdf', 'wb') as f:
#     f.write(r.content)

# with pdfplumber.open('book.pdf') as pdf:
#     text = '\n'.join(page.extract_text() or '' for page in pdf.pages)

# with open('story_of_a_modern_woman.txt', 'w', encoding='utf-8') as f:
#     f.write(text)


# # Cleaning the text
# with open('story_of_a_modern_woman.txt', 'r') as f:
#     lines = f.readlines()

# # skip front matter, remove lines with page numbers at end
# clean = []
# for line in lines[65:]:
#     if line.strip().startswith('CHAPTER') and line.strip()[-1].isdigit():
#         continue
#     if line.strip() == 'The Story of a Modern Woman':
#         continue
#     clean.append(line)

# with open('story_of_a_modern_woman_clean.txt', 'w') as f:
#     f.writelines(clean)

with open('life_final.txt', 'r') as f:
    lines = f.readlines()

clean = []
for l in lines:
    s = l.strip()
    # skip short lines containing these phrases (headers)
    if ('LIFE AND' in s or 'MICHAEL ARMSTRONG' in s) and len(s) < 40:
        continue
    clean.append(l)

with open('life_cleaned.txt', 'w') as f:
    f.writelines(clean)

print(f'{len(lines)} -> {len(clean)} lines')