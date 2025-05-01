import os
import re
import string
import chardet

def detect_encoding(file_path):
    """Detect the encoding of a file."""
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

def preprocess_bind_folder(folder_path="BIND", backup=True):
    """
    Preprocess semua file txt di folder BIND dan simpan kembali ke file yang sama.
    Jika backup=True, maka file asli akan disimpan dengan tambahan .original
    """
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} tidak ditemukan.")
        return

    backup_folder = f"{folder_path}_original"
    if backup and not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
        
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    total_files = len(txt_files)
    
    print(f"Memproses {total_files} file di folder {folder_path}...")
    
    for i, filename in enumerate(txt_files, 1):
        file_path = os.path.join(folder_path, filename)
        
        try:
            # Deteksi encoding file
            encoding = detect_encoding(file_path)
            
            # Baca file dengan encoding yang sesuai
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            # Buat backup jika diperlukan
            if backup:
                backup_path = os.path.join(backup_folder, filename)
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Preprocess teks
            preprocessed_content = preprocess_text(content)
            
            # Tulis kembali ke file asli
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(preprocessed_content)
            
            print(f"[{i}/{total_files}] Berhasil memproses {filename}")
            
        except Exception as e:
            print(f"[{i}/{total_files}] Gagal memproses {filename}: {str(e)}")
    
    print(f"\nSelesai! Total {total_files} file telah diproses.")
    if backup:
        print(f"File asli telah dicadangkan di folder {backup_folder}")

if __name__ == "__main__":
    preprocess_bind_folder()