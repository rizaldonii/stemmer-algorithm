import os
import chardet
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load kamus kata dasar
def load_dictionary(path="dictionary.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return set(word.strip().lower() for word in f)

dictionary_set = load_dictionary()

# ECS Stemmer
def ecs_stem(word):
    stemmed = stemmer.stem(word)
    return stemmed if stemmed in dictionary_set else word

# Preprocessing teks
def preprocess_text(text):
    """
    Melakukan preprocessing pada teks:
    1. Mengubah ke lowercase
    2. Menghapus tanda baca
    3. Menghapus angka
    4. Menghapus multiple spaces
    5. Menghapus karakter khusus
    """
    # Mengubah ke lowercase
    text = text.lower()
    
    # Menghapus URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Menghapus tag HTML jika ada
    text = re.sub(r'<.*?>', '', text)
    
    # Menghapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Menghapus angka
    text = re.sub(r'\d+', '', text)
    
    # Menghapus karakter khusus dan non-ASCII
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # Menghapus multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Menghapus whitespace di awal dan akhir
    return text.strip()

# Modifikasi fungsi stem_document untuk menggabungkan preprocessing
def stem_document(text):
    # Lakukan preprocessing terlebih dahulu
    preprocessed_text = preprocess_text(text)
    # Pisahkan teks menjadi token kata
    words = preprocessed_text.split()
    # Lakukan stemming pada setiap kata
    return ' '.join(ecs_stem(word) for word in words)


# Deteksi encoding dengan peningkatan
def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read()
        # Periksa BOM (Byte Order Mark) untuk UTF-8
        if raw_data.startswith(b'\xef\xbb\xbf'):
            return 'utf-8-sig'
        result = chardet.detect(raw_data)
        # Prioritaskan beberapa encoding yang umum untuk teks Indonesia
        if result['encoding'] in ['ascii', 'Windows-1252', 'Windows-1254']:
            # Coba UTF-8 dulu karena chardet kadang salah deteksi UTF-8 sebagai ASCII
            try:
                raw_data.decode('utf-8')
                return 'utf-8'
            except UnicodeDecodeError:
                pass
        return result['encoding'] or 'utf-8'

# Proses folder input dengan penanganan encoding yang lebih baik
def process_opinions(input_folder="BIND preprocessed", output_folder="hasil_stemming"):
    os.makedirs(output_folder, exist_ok=True)
    txt_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")][:30]

    for filename in txt_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            encoding = detect_encoding(input_path)
            if encoding.lower() != "utf-8":
                print(f"Encoding non-utf8 terdeteksi pada {filename}: {encoding}")

            # Baca dengan encoding yang terdeteksi
            with open(input_path, "r", encoding=encoding, errors="replace") as infile:
                original_text = infile.read()

            # Proses stemming
            stemmed_text = stem_document(original_text)

            # Simpan hasil ke file output dalam UTF-8 dengan BOM
            with open(output_path, "w", encoding="utf-8-sig") as outfile:
                outfile.write(stemmed_text)

            print(f"Diproses: {filename} (Encoding asli: {encoding})")

        except Exception as e:
            print(f"Gagal memproses {filename}: {str(e)}")

# Eksekusi
if __name__ == "__main__":
    process_opinions()