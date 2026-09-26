import pandas as pd
import xgboost as xgb
import os
import argparse

def predict_matches(features_file, cands_file, model_path, output_file, threshold=0.8):
    print("Running inference...")
    if not os.path.exists(features_file):
        print("Features file not found.")
        return
        
    df_features = pd.read_csv(features_file)
    feature_cols = ['name_jaccard', 'addr_jaccard', 'name_fuzz', 'addr_fuzz', 
                    'name_tokensort', 'addr_tokensort', 'name_tokenset', 'addr_tokenset']
    X = df_features[feature_cols]
    
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # Predict probability of match
    probs = model.predict_proba(X)[:, 1]
    df_features['match_prob'] = probs
    
    # Filter by strict threshold
    matches = df_features[df_features['match_prob'] > threshold]
    
    # Group by S1 entity
    match_grouped = matches.groupby('id1')['id2'].apply(lambda x: ','.join(set(x))).reset_index()
    match_grouped.columns = ['source1_entity_id', 'matched_entity_ids']
    
    # Ensure all S1 entities from the candidate pairs are included
    cands = pd.read_csv(cands_file, sep='\t')
    final_output = pd.merge(cands[['source1_entity_id']], match_grouped, on='source1_entity_id', how='left')
    final_output['matched_entity_ids'] = final_output['matched_entity_ids'].fillna('')
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_output.to_csv(output_file, sep='\t', index=False)
    print(f"Final matches saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.8)
    args = parser.parse_args()
    
    predict_matches(
        '../../../output/test_features.csv',
        '../../../output/test_candidate_pairs.tsv',
        '../../../output/xgb_model.json',
        '../../../output/matching_results.tsv',
        threshold=args.threshold
    )
