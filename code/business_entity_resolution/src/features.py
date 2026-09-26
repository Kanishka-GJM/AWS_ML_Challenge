import pandas as pd
import numpy as np
import os
import argparse
import time
import re
from rapidfuzz import fuzz

def clean_text(text):
    if pd.isna(text) or not text:
        return ""
    text = str(text).lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Standardize abbreviations
    replacements = {
        r'\bcorp\b': 'corporation',
        r'\binc\b': 'incorporated',
        r'\bltd\b': 'limited',
        r'\bco\b': 'company',
        r'\bllc\b': 'limited liability company',
        r'\bpvt\b': 'private',
        r'\bmfg\b': 'manufacturing',
        r'\bst\b': 'street',
        r'\brd\b': 'road',
        r'\bave\b': 'avenue',
        r'\bblvd\b': 'boulevard',
        r'\bdr\b': 'drive',
        r'\bste\b': 'suite',
        r'\bfl\b': 'floor',
        r'\bapt\b': 'apartment',
        r'\bbldg\b': 'building'
    }
    for k, v in replacements.items():
        text = re.sub(k, v, text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_jaccard(str1, str2):
    if not str1 or not str2: return 0.0
    set1 = set(str1.split())
    set2 = set(str2.split())
    if not set1 or not set2: return 0.0
    union = len(set1.union(set2))
    return len(set1.intersection(set2)) / union if union > 0 else 0.0

def build_features(mode, cand_file, output_file):
    print(f"Building features for {mode} set...")
    start_time = time.time()
    candidates = pd.read_csv(cand_file, sep='\t')
    
    print("Loading datasets for feature extraction...")
    s1_path = f'../../../dataset/{mode}/{mode}_source1.tsv'
    s2_path = f'../../../dataset/{mode}/{mode}_source2.tsv'
    s3_path = f'../../../dataset/{mode}/{mode}_source3.tsv'
    
    s1_df = pd.read_csv(s1_path, sep='\t', usecols=['entity_id', 'business_name', 'business_address'])
    s2_df = pd.read_csv(s2_path, sep='\t', usecols=['entity_id', 'business_name', 'business_address'])
    s3_df = pd.read_csv(s3_path, sep='\t', usecols=['entity_id', 'business_name', 'business_address'])
    
    s23_df = pd.concat([s2_df, s3_df])
    
    print("Creating lookup dictionaries with cleaned text...")
    s1_dict = s1_df.set_index('entity_id').to_dict('index')
    s23_dict = s23_df.set_index('entity_id').to_dict('index')
    
    # Pre-clean the dictionaries to save time in the loop
    for d in [s1_dict, s23_dict]:
        for k, v in d.items():
            v['business_name'] = clean_text(v.get('business_name', ''))
            v['business_address'] = clean_text(v.get('business_address', ''))
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    batch_size = 50000
    batch_data = []
    total_pairs = 0
    
    processed_id1s = set()
    if os.path.exists(output_file):
        print(f"Found existing {output_file}. Finding processed id1s to resume...")
        try:
            processed_df = pd.read_csv(output_file, usecols=['id1'])
            processed_id1s = set(processed_df['id1'].astype(str).unique())
            if not processed_df.empty:
                last_id1 = str(processed_df['id1'].iloc[-1])
                processed_id1s.discard(last_id1)
            print(f"Found {len(processed_id1s)} fully processed id1s to skip.")
            total_pairs = len(processed_df)
        except Exception as e:
            print(f"Error reading existing file for resume: {e}")

    if not processed_id1s:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('id1,id2,name_jaccard,addr_jaccard,name_fuzz,addr_fuzz,name_tokensort,addr_tokensort,name_tokenset,addr_tokenset\n')
        
    def write_batch(data):
        df_batch = pd.DataFrame(data)
        df_batch.to_csv(output_file, mode='a', header=False, index=False)
        
    print("Calculating features and saving in chunks (this might take a moment)...")
    for _, row in candidates.iterrows():
        id1 = row['source1_entity_id']
        if str(id1) in processed_id1s:
            continue
        cands = str(row['candidate_entity_ids']).split(',')
        
        info1 = s1_dict.get(id1, {})
        n1 = info1.get('business_name', '')
        a1 = info1.get('business_address', '')
        
        for id2 in cands:
            if not id2 or id2 == 'nan': continue
            
            info2 = s23_dict.get(id2, {})
            n2 = info2.get('business_name', '')
            a2 = info2.get('business_address', '')
            
            batch_data.append({
                'id1': id1, 'id2': id2,
                'name_jaccard': calculate_jaccard(n1, n2),
                'addr_jaccard': calculate_jaccard(a1, a2),
                'name_fuzz': fuzz.ratio(n1, n2) / 100.0 if n1 and n2 else 0.0,
                'addr_fuzz': fuzz.ratio(a1, a2) / 100.0 if a1 and a2 else 0.0,
                'name_tokensort': fuzz.token_sort_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0,
                'addr_tokensort': fuzz.token_sort_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0,
                'name_tokenset': fuzz.token_set_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0,
                'addr_tokenset': fuzz.token_set_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
            })
            total_pairs += 1
            
            if len(batch_data) >= batch_size:
                write_batch(batch_data)
                batch_data = []
                
    if batch_data:
        write_batch(batch_data)
        
    print(f"Features saved to {output_file} in {time.time() - start_time:.2f} seconds. Total pairs: {total_pairs}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'])
    parser.add_argument('--candidate-file', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    build_features(args.mode, args.candidate_file, args.output)
