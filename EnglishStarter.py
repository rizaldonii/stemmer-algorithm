import os
import string
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize

folder_path = 'BING_original'
all_words = []

stop_words = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'while', 'with', 'of', 'at', 'by', 'for', 'to', 'in', 'on', 'from',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'shall', 'should', 'can', 'could', 'may', 'might', 'must', 'this', 'that', 'these', 'those', 'as', 'it', 'its',
    'he', 'she', 'they', 'them', 'his', 'her', 'their', 'you', 'your', 'i', 'we', 'us', 'me', 'my', 'mine', 'ours'
}

if not os.path.isdir(folder_path):
    raise FileNotFoundError(f"The folder '{folder_path}' does not exist. Please check the path.")

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = word_tokenize(line)
                processed = [
                    word.lower() for word in tokens
                    if word.isalpha() and word.lower() not in stop_words
                ]
                all_words.extend(processed)

total_words = len(all_words)
unique_words = set(all_words)
total_unique_words = len(unique_words)

print(f"Total words: {total_words}")
print(f"Total unique words: {total_unique_words}")
print(f"All words: {all_words[:10]}")
print(f"Sample unique words: {list(unique_words)[:10]}")

with open("word word today.txt", "w", encoding="utf-8") as out_file:
    for word in sorted(unique_words):
        out_file.write(word + "\n")