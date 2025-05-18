import os
import re
import chardet
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

input_folder = 'BING preprocessed'
output_folder = 'english stemmed output'
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        raw = f.read()
        if raw.startswith(b'\xef\xbb\xbf'):
            return 'utf-8-sig'
        res = chardet.detect(raw)
        return res['encoding'] or 'utf-8'

def preprocess(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

metrics = []
os.makedirs(output_folder, exist_ok=True)
files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

total_ui = 0
total_oi = 0
total_mwc = 0

for fname in files:
    path = os.path.join(input_folder, fname)
    enc = detect_encoding(path)
    with open(path, encoding=enc, errors='replace') as f:
        raw = f.read()
    
    clean = preprocess(raw)
    tokens = [w for w in word_tokenize(clean) if w not in stop_words and len(w)>2]
    stems = [stemmer.stem(w) for w in tokens]
    w2s = dict(zip(tokens, stems))
    
    stem_map = {}
    for w,s in w2s.items():
        stem_map.setdefault(s, set()).add(w)
    
    mwc = sum(len(ws) for ws in stem_map.values()) / len(stem_map) if stem_map else 0
    oi = sum(1 for ws in stem_map.values() if len(ws)>1) / len(stem_map) if stem_map else 0
    prefix_map = {}

    for w,s in w2s.items():
        p = w[:4] if len(w)>=4 else w
        prefix_map.setdefault(p, set()).add(s)
    under = sum(1 for st in prefix_map.values() if len(st)>1)
    ui = under / len(prefix_map) if prefix_map else 0
    
    total_ui += ui
    total_oi += oi
    total_mwc += mwc

    # Output file lebih rapi
    out_text = (
        "Stemmed Words:\n"
        + " ".join(stems)
        + "\n\nMetrics:\n"
        + f"{'Metric':<25}{'Value':>10}\n"
        + f"{'-'*35}\n"
        + f"{'Understemming Index (UI)':<25}{ui:>10.2f}\n"
        + f"{'Overstemming Index (OI)':<25}{oi:>10.2f}\n"
        + f"{'Mean Word Conflation (MWC)':<25}{mwc:>10.2f}\n"
    )
    with open(os.path.join(output_folder, fname), 'w', encoding='utf-8') as out:
        out.write(out_text)
    
    metrics.append({
        'Filename': fname,
        'UI': round(ui,2),
        'OI': round(oi,2),
        'MWC': round(mwc,2),
    })

df = pd.DataFrame(metrics)
print(df.to_string(index=False, justify='right'))

n_files = len(files)
if n_files:
    print("\nAverages:")
    print(f"{'Average Understemming Index (UI)':<35}: {total_ui/n_files:.2f}")
    print(f"{'Average Overstemming Index (OI)':<35}: {total_oi/n_files:.2f}")
    print(f"{'Average Mean Word Conflation (MWC)':<35}: {total_mwc/n_files:.2f}")
else:
    print("No files processed.")