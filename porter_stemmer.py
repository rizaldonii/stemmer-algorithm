import os
import re
import nltk
import chardet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Pastikan resource NLTK tersedia
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Inisialisasi stemmer dan stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Folder input dan output
input_folder = 'BING preprocessed'
output_folder = 'english stemmed output'

# Deteksi encoding file
def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read()
        if raw_data.startswith(b'\xef\xbb\xbf'):
            return 'utf-8-sig'
        result = chardet.detect(raw_data)
        if result['encoding'] in ['ascii', 'Windows-1252', 'Windows-1254']:
            try:
                raw_data.decode('utf-8')
                return 'utf-8'
            except UnicodeDecodeError:
                pass
        return result['encoding'] or 'utf-8'

# Fungsi preprocessing yang lebih komprehensif
def preprocess_text(text):
    """
    Melakukan preprocessing pada teks:
    1. Mengubah ke lowercase
    2. Menghapus URL
    3. Menghapus tag HTML
    4. Menghapus tanda baca dan karakter khusus
    5. Menghapus angka
    6. Menghapus multiple spaces
    """
    # Mengubah ke lowercase
    text = text.lower()
    
    # Menghapus URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Menghapus tag HTML jika ada
    text = re.sub(r'<.*?>', '', text)
    
    # Menghapus tanda baca dan karakter non-huruf
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Menghapus multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Menghapus whitespace di awal dan akhir
    return text.strip()

# Fungsi porter stemming untuk satu kata
def porter_stem(word):
    """
    Implementasi Porter Stemmer dengan pengecekan
    """
    # NLTK Porter Stemmer
    return stemmer.stem(word)

# Fungsi preprocessing + stemming
def preprocess_and_stem(text):
    # Preprocessing
    text = preprocess_text(text)

    # Tokenisasi
    tokens = word_tokenize(text)

    # Hilangkan stopwords dan kata pendek (opsional)
    filtered = [word for word in tokens if word not in stop_words and len(word) > 2]

    # Stemming
    stemmed = [porter_stem(word) for word in filtered]

    return ' '.join(stemmed)

# Pastikan folder output ada
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print(f"Memproses file dari folder '{input_folder}'...")

# Proses semua file
file_count = 0
processed_count = 0
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        file_count += 1
        input_path = os.path.join(input_folder, filename)
        
        try:
            # Deteksi encoding
            encoding = detect_encoding(input_path)
            
            # Baca file dengan encoding yang sesuai
            with open(input_path, 'r', encoding=encoding, errors='replace') as file:
                text = file.read()

            # Preprocess dan stem teks
            stemmed_text = preprocess_and_stem(text)

            # Simpan hasil
            output_path = os.path.join(output_folder, filename)
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(stemmed_text)
                
            processed_count += 1
            print(f"[{file_count}/{len([f for f in os.listdir(input_folder) if f.endswith('.txt')])}] Berhasil memproses: {filename}")
            
        except Exception as e:
            print(f"[{file_count}/{len([f for f in os.listdir(input_folder) if f.endswith('.txt')])}] Gagal memproses {filename}: {str(e)}")

print(f"\nStemming dengan Porter Stemmer selesai untuk {processed_count}/{file_count} file.")

# --- Evaluasi MWC ---
from collections import Counter

print("\nMemulai evaluasi Mean Word Conflation (MWC)...")

all_words = []
all_stemmed = []

# Kumpulkan semua kata dari file mentah dan hasil stemming
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        try:
            # Deteksi encoding
            encoding = detect_encoding(os.path.join(input_folder, filename))
            
            with open(os.path.join(input_folder, filename), 'r', encoding=encoding, errors='replace') as file:
                text = file.read()
                # Gunakan fungsi preprocessing yang sama
                text = preprocess_text(text)
                tokens = word_tokenize(text)
                filtered = [word for word in tokens if word not in stop_words]

                all_words.extend(filtered)
                stemmed = [porter_stem(w) for w in filtered]
                all_stemmed.extend(stemmed)
        except Exception as e:
            print(f"Gagal mengevaluasi {filename}: {str(e)}")

# Hitung MWC
total_words = len(all_words)
unique_stems = len(set(all_stemmed))
MWC = total_words / unique_stems if unique_stems != 0 else 0

print("\nHASIL EVALUASI PORTER STEMMER:")
print("="*50)
print(f"Total kata: {total_words}")
print(f"Unique stems: {unique_stems}")
print(f"Mean Word Conflation (MWC): {MWC:.2f}")

# Hitung distribusi stem
stem_counts = Counter(all_stemmed)
top_stems = stem_counts.most_common(10)

print("\nTOP 10 STEM:")
print("="*50)
for stem, count in top_stems:
    print(f"{stem}: {count} occurrences")

# Evaluasi tambahan: contoh group kata
print("\nCONTOH GROUP KATA DENGAN STEM YANG SAMA:")
print("="*50)

# Pilih beberapa stem populer dan tampilkan kata aslinya
stem_to_words = defaultdict(list)
for word, stem in zip(all_words, all_stemmed):
    if word != stem:  # Hanya kata yang berubah ketika di-stem
        stem_to_words[stem].append(word)

# Tampilkan 5 grup stem terbanyak
counter = 0
for stem, count in top_stems:
    if stem in stem_to_words and len(stem_to_words[stem]) > 1:
        print(f"Stem '{stem}' ({count} occurrences):")
        # Tampilkan maksimal 7 kata asli yang menjadi stem ini (unik)
        unique_words = list(set(stem_to_words[stem]))[:7]
        print(f"  Original words: {', '.join(unique_words)}")
        counter += 1
        if counter >= 5:
            break