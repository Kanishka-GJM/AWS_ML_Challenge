import pandas as pd
import numpy as np
import os
import argparse

def split_and_prepare_data(input_dir, output_dir, train_ratio=0.8):
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    # Process sources and assign train/test splits randomly
    np.random.seed(42)
    
    train_ids = set()
    test_ids = set()
    
    def process_source(source_name):
        print(f"Reading {source_name}...")
        df = pd.read_parquet(os.path.join(input_dir, source_name))
        
        id_col = 'entity_id' if 'entity_id' in df.columns else 'id'
        if id_col not in df.columns:
            df = df.rename(columns={df.columns[0]: 'entity_id'})
            id_col = 'entity_id'
            
        # Randomly assign
        mask = np.random.rand(len(df)) < train_ratio
        train_df = df[mask]
        test_df = df[~mask]
        
        train_out = os.path.join(output_dir, 'train', source_name)
        test_out = os.path.join(output_dir, 'test', source_name)
        
        train_df.to_parquet(train_out)
        test_df.to_parquet(test_out)
        
        # Add to sets for ground truth filtering
        train_ids.update(train_df[id_col].values)
        test_ids.update(test_df[id_col].values)
        
        print(f"Saved {source_name} - Train: {len(train_df)}, Test: {len(test_df)}")

    process_source('S1')
    process_source('S2')
    process_source('S3')
    
    print("Reading ground truth...")
    gt = pd.read_parquet(os.path.join(input_dir, 'ground_truth'))
    
    print("Splitting ground truth...")
    train_gt = gt[(gt['source1_entity_id'].isin(train_ids)) & (gt['matched_entity_id'].isin(train_ids))]
    test_gt = gt[(gt['source1_entity_id'].isin(test_ids)) & (gt['matched_entity_id'].isin(test_ids))]
    
    train_gt.to_parquet(os.path.join(output_dir, 'train', 'ground_truth'))
    test_gt.to_parquet(os.path.join(output_dir, 'test', 'ground_truth'))
    print(f"Saved ground truth - Train: {len(train_gt)}, Test: {len(test_gt)}")
    
    print("Data preparation complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../../../preprocessed_data_parquets')
    parser.add_argument('--output', type=str, default='../../../preprocessed_data_parquets')
    args = parser.parse_args()
    
    split_and_prepare_data(args.input, args.output)
