import pandas as pd
import xgboost as xgb
import os

def train_model(features_file, ground_truth_file, model_output_path):
    print("Training XGBoost model...")
    if not os.path.exists(features_file):
        print("Features file not found.")
        return
        
    df_features = pd.read_csv(features_file)
    gt = pd.read_csv(ground_truth_file, sep='\t')
    
    # Create ground truth pairs
    true_pairs = set()
    for _, row in gt.iterrows():
        s1 = row['source1_entity_id']
        cands = str(row['matched_entity_id']).split(',')
        for c in cands:
            if c and c != 'nan':
                true_pairs.add(f"{s1}_{c}")
                
    # Label features (Optimized Vectorized Approach)
    print("Labeling features (Vectorized)...")
    # Create the pair IDs using vectorized string concatenation
    pair_ids = df_features['id1'].astype(str) + '_' + df_features['id2'].astype(str)
    
    # Check if each pair is in the ground truth true_pairs set
    df_features['is_match'] = pair_ids.isin(true_pairs).astype(int)
    
    # Use all 8 features
    feature_cols = ['name_jaccard', 'addr_jaccard', 'name_fuzz', 'addr_fuzz', 
                    'name_tokensort', 'addr_tokensort', 'name_tokenset', 'addr_tokenset']
    X = df_features[feature_cols]
    y = df_features['is_match']
    
    # Free memory
    del df_features
    del pair_ids
    import gc
    gc.collect()
    
    # Train model (Optimized for large datasets)
    print("Fitting model using 'hist' method...")
    model = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',  # Highly optimized for large datasets
        n_jobs=-1            # Use all CPU cores
    )
    model.fit(X, y)
    
    model.save_model(model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == '__main__':
    # Notice we look for train_features.csv now
    train_model(
        '../../../output/train_features.csv',
        '../../../dataset/train/train_ground_truth.tsv',
        '../../../output/xgb_model.json'
    )
