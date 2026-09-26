import os
import subprocess
import time

def run_command(cmd):
    print(f"\n[{time.strftime('%X')}] Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    start_time = time.time()
    
    # --- 1. TRAIN PIPELINE ---
    print("\n" + "="*40)
    print("PHASE 1: TRAINING ON dataset/train")
    print("="*40)
    
    # Generate train candidates
    run_command("python blocking.py --mode train --output ../../../output/train_candidate_pairs.tsv")
    
    # Generate train features
    run_command("python features.py --mode train --candidate-file ../../../output/train_candidate_pairs.tsv --output ../../../output/train_features.csv")
    
    # Train model
    run_command("python train.py")
    
    # --- 2. TEST PIPELINE ---
    print("\n" + "="*40)
    print("PHASE 2: INFERENCE ON dataset/test")
    print("="*40)
    
    # Generate test candidates
    run_command("python blocking.py --mode test --output ../../../output/test_candidate_pairs.tsv")
    
    # Generate test features
    run_command("python features.py --mode test --candidate-file ../../../output/test_candidate_pairs.tsv --output ../../../output/test_features.csv")
    
    # Predict matches
    run_command("python predict.py --threshold 0.8")
    
    print(f"\nPipeline finished in {(time.time() - start_time) / 60:.1f} minutes!")
    print("Output files are ready in the output/ directory:")
    print(" - matching_results.tsv")
    print(" - test_candidate_pairs.tsv (rename this to candidate_pairs.tsv if the validator requires it)")

if __name__ == '__main__':
    main()
