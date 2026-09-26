import os
from blocking import generate_candidates
from features import build_features
from train import train_model
from predict import predict_matches

if __name__ == '__main__':
    print("Starting Entity Resolution Pipeline...")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    s1_dir = os.path.join(base_dir, 'S1')
    s2_dir = os.path.join(base_dir, 'S2')
    s3_dir = os.path.join(base_dir, 'S3')
    out_dir = os.path.join(base_dir, 'output')
    
    cand_file = os.path.join(out_dir, 'candidate_pairs.tsv')
    feat_file = os.path.join(out_dir, 'features.csv')
    model_file = os.path.join(out_dir, 'xgb_model.json')
    gt_file = os.path.join(base_dir, 'dataset', 'train', 'train_ground_truth.tsv')
    match_file = os.path.join(out_dir, 'matching_results.tsv')
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Phase 1: Blocking
    generate_candidates(s1_dir, s2_dir, s3_dir, cand_file)
    
    # Phase 2: Feature Engineering
    build_features(cand_file, s1_dir, s2_dir, s3_dir, feat_file)
    
    # Phase 3: Train
    train_model(feat_file, gt_file, model_file)
    
    # Phase 4: Predict
    predict_matches(feat_file, model_file, match_file, threshold=0.8)
    
    print("Pipeline complete. Please run utils/validate_submission.py to verify.")
