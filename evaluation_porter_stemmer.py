import os
from collections import defaultdict

def load_gold_standard(gold_folder="english texts"):
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
    gold_standard = load_gold_standard(gold_folder)
    stemmed_results = load_stemmed_results(result_folder)
    
    total_tokens = 0
    understemming = 0
    overstemming = 0
    correct = 0
    consistency_groups = defaultdict(set)
    
    for filename in gold_standard:
        if filename not in stemmed_results:
            continue
            
        gold_tokens = gold_standard[filename]
        stemmed_tokens = stemmed_results[filename]
        
        for gold_word, stemmed_word in zip(gold_tokens, stemmed_tokens):
            total_tokens += 1
            if stemmed_word == gold_word:
                correct += 1
            elif len(stemmed_word) > len(gold_word):
                understemming += 1
            else:
                overstemming += 1
                
            consistency_groups[stemmed_word].add(gold_word)
    
    # Hitung metrik evaluasi
    accuracy = correct / total_tokens
    UI = understemming / total_tokens
    OI = overstemming / total_tokens
    
    # Hitung MWC (Mean Weighted Consistency)
    mwc_numerator = 0
    for stem in consistency_groups:
        group_size = len(consistency_groups[stem])
        if group_size > 1:
            majority = max([list(consistency_groups[stem]).count(w) for w in consistency_groups[stem]])
            consistency = majority / group_size
            mwc_numerator += group_size * consistency
    
    MWC = mwc_numerator / total_tokens if total_tokens > 0 else 0
    
    return {
        'Total_Tokens': total_tokens,
        'Accuracy': accuracy,
        'Understemming_Index': UI,
        'Overstemming_Index': OI,
        'Mean_Weighted_Consistency': MWC,
        'Error_Rate': (understemming + overstemming) / total_tokens
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
    gold_folder = "english texts"  # Folder berisi file teks dengan stem benar
    result_folder = "english stemmed output"  # Folder output stemming
    
    results = evaluate_stemming_performance(gold_folder, result_folder)
    
    print("\nEVALUATION RESULTS:")
    print("="*50)
    print(f"Total Tokens Analyzed: {results['Total_Tokens']}")
    print(f"Accuracy: {results['Accuracy']:.2%}")
    print(f"Understemming Index (UI): {results['Understemming_Index']:.4f}")
    print(f"Overstemming Index (OI): {results['Overstemming_Index']:.4f}")
    print(f"Mean Weighted Consistency (MWC): {results['Mean_Weighted_Consistency']:.4f}")
    print(f"Overall Error Rate: {results['Error_Rate']:.2%}")
    
    # Generate detailed error report
    generate_error_report(gold_folder, result_folder)
    print("\nError report generated: error_report_porter.txt")