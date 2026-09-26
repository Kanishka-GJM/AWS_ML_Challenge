import pandas as pd
import os
import argparse
import time
import re

def get_block_key(row):
    try:
        n = str(row['business_name']).lower()
        a = str(row['business_address']).lower()
        c = str(row['country']).lower()
        n_toks = re.findall(r'[a-z]{3,}', n)
        n_tok = n_toks[0] if n_toks else ''
        a_toks = re.findall(r'[a-z0-9]{2,}', a)
        a_tok = a_toks[0] if a_toks else ''
        if n_tok and a_tok:
            return f'{c}_{n_tok}_{a_tok}'
    except: pass
    return ''

def generate_candidates(mode, output_file):
    print(f"Generating candidates for {mode} set...")
    start_time = time.time()
    
    s1_path = f'../../../dataset/{mode}/{mode}_source1.tsv'
    s2_path = f'../../../dataset/{mode}/{mode}_source2.tsv'
    s3_path = f'../../../dataset/{mode}/{mode}_source3.tsv'
    
    def process_file(filepath):
        print(f"Reading {filepath}...")
        df = pd.read_csv(filepath, sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'])
        df['business_name'] = df['business_name'].astype(str)
        df['business_address'] = df['business_address'].astype(str)
        df['country'] = df['country'].astype(str)
        df['block_key'] = df.apply(get_block_key, axis=1)
        df = df[df['block_key'] != '']
        return df[['entity_id', 'block_key']]

    df1 = process_file(s1_path)
    df2 = process_file(s2_path)
    df3 = process_file(s3_path)
    
    print("Concatenating Source 2 and Source 3...")
    df23 = pd.concat([df2, df3])
    
    print("Joining on blocking key (this should be fast now)...")
    candidates = pd.merge(df1, df23, on='block_key')
    
    print("Grouping by S1 entity...")
    cand_grouped = candidates.groupby('entity_id_x')['entity_id_y'].apply(lambda x: ','.join(set(x))).reset_index()
    cand_grouped.columns = ['source1_entity_id', 'candidate_entity_ids']
    
    print("Adding missing singletons...")
    s1_all = pd.read_csv(s1_path, sep='\t', usecols=['entity_id'])
    s1_all.columns = ['source1_entity_id']
    
    final_cands = pd.merge(s1_all, cand_grouped, on='source1_entity_id', how='left')
    final_cands['candidate_entity_ids'] = final_cands['candidate_entity_ids'].fillna('')
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_cands.to_csv(output_file, sep='\t', index=False)
    print(f"Candidates saved to {output_file} in {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'])
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    generate_candidates(args.mode, args.output)
