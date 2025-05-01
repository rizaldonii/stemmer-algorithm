import os
from collections import defaultdict
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def load_gold_standard(gold_folder="BING preprocessed"):
    gold_data = {}
    for filename in os.listdir(gold_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(gold_folder, filename), 'r', encoding='utf-8') as f:
                content = f.read().split()
                gold_data[filename] = content
    return gold_data

def load_stemmed_results(result_folder="english stemmed output"):
    stemmed_data = {}
    for filename in os.listdir(result_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(result_folder, filename), 'r', encoding='utf-8') as f:
                content = f.read().split()
                stemmed_data[filename] = content
    return stemmed_data

def evaluate_stemming_performance(gold_folder, result_folder):
    nltk.download('punkt', quiet=True)
    stemmer = PorterStemmer()
    
    all_words = []
    all_stemmed = []
    stem_map = defaultdict(set)
    
    # Loop through all files in the gold folder
    for filename in os.listdir(gold_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(gold_folder, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                tokens = word_tokenize(text.lower())
                all_words.extend(tokens)
                stemmed = [stemmer.stem(w) for w in tokens]
                all_stemmed.extend(stemmed)

                # Create stem -> original words mapping
                for orig, stem in zip(tokens, stemmed):
                    stem_map[stem].add(orig)
    
    # Calculate MWC (Mean Word Conflation)
    total_words = len(all_words)
    unique_stems = len(set(all_stemmed))
    MWC = total_words / unique_stems if unique_stems != 0 else 0
    
    # Calculate Overstemming Index
    overstem_pairs = 0
    for stem, originals in stem_map.items():
        if len(originals) > 1:
            overstem_pairs += len(originals) - 1  # How many words are excessively merged
    
    # Calculate Understemming Index
    # Find words with the same prefix but different stems
    understem_pairs = 0
    prefix_map = defaultdict(set)
    for word, stem in zip(all_words, all_stemmed):
        if len(word) >= 4:  # Make sure word is long enough for prefix
            prefix = word[:4]  # Take first 4 characters as prefix
            prefix_map[prefix].add(stem)
    
    for stems in prefix_map.values():
        if len(stems) > 1:
            understem_pairs += len(stems) - 1
    
    # Normalize OI and UI
    normalized_OI = overstem_pairs / unique_stems if unique_stems != 0 else 0
    normalized_UI = understem_pairs / unique_stems if unique_stems != 0 else 0
    
    return {
        'Total_Words': total_words,
        'Unique_Stems': unique_stems,  # Still keeping in return value for internal use if needed
        'Mean_Word_Conflation': MWC,
        'Overstemming_Index': normalized_OI,
        'Understemming_Index': normalized_UI
    }

def generate_error_report(gold_folder, result_folder, output_file="error_report_porter.txt"):
    gold_standard = load_gold_standard(gold_folder)
    stemmed_results = load_stemmed_results(result_folder)
    
    with open(output_file, 'w', encoding='utf-8') as report:
        report.write("ERROR ANALYSIS REPORT\n")
        report.write("="*50 + "\n")
        
        error_types = defaultdict(int)
        word_errors = defaultdict(int)
        
        for filename in gold_standard:
            if filename not in stemmed_results:
                continue
                
            gold_tokens = gold_standard[filename]
            stemmed_tokens = stemmed_results[filename]
            
            report.write(f"\nFile: {filename}\n")
            report.write("-"*50 + "\n")
            
            for i, (gold_word, stemmed_word) in enumerate(zip(gold_tokens, stemmed_tokens)):
                if stemmed_word != gold_word:
                    error_type = "UNDER" if len(stemmed_word) > len(gold_word) else "OVER"
                    error_types[error_type] += 1
                    word_errors[f"{gold_word}→{stemmed_word}"] += 1
                    
                    report.write(f"Token {i+1}: {gold_word} → {stemmed_word} ({error_type})\n")
        
        # Summary statistics
        report.write("\n\nSUMMARY STATISTICS\n")
        report.write("="*50 + "\n")
        report.write(f"Total Errors: {sum(error_types.values())}\n")
        report.write(f"Understemming Errors: {error_types.get('UNDER', 0)}\n")
        report.write(f"Overstemming Errors: {error_types.get('OVER', 0)}\n")
        
        # Most common errors
        report.write("\nTOP 10 MOST COMMON ERRORS\n")
        for error, count in sorted(word_errors.items(), key=lambda x: x[1], reverse=True)[:10]:
            report.write(f"{error}: {count} occurrences\n")


if __name__ == "__main__":
    gold_folder = "BING preprocessed"  # Folder with gold standard texts
    result_folder = "english stemmed output"  # Stemming output folder
    
    results = evaluate_stemming_performance(gold_folder, result_folder)
    
    print("\nEVALUATION RESULTS:")
    print("="*50)
    print(f"Total Tokens Analyzed: {results['Total_Words']}")  # Changed from "Total Words Analyzed" to "Total Tokens Analyzed"
    print(f"Mean Word Conflation (MWC): {results['Mean_Word_Conflation']:.2f}")
    print(f"Overstemming Index (OI): {results['Overstemming_Index']:.4f}")
    print(f"Understemming Index (UI): {results['Understemming_Index']:.4f}")
    
    # Generate detailed error report
    generate_error_report(gold_folder, result_folder)
    print("\nError report generated: error_report_porter.txt")