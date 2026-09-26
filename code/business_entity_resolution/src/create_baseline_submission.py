import pandas as pd
import os

def create_baseline():
    print("Reading test_source1.tsv...")
    test_s1_path = '../../../dataset/test/test_source1.tsv'
    out_matching = '../../../output/matching_results.tsv'
    out_candidate = '../../../output/candidate_pairs.tsv'
    
    # Read only the entity_id column to save memory and time
    df = pd.read_csv(test_s1_path, sep='\t', usecols=['entity_id'])
    
    # Create the required format
    df = df.rename(columns={'entity_id': 'source1_entity_id'})
    df['matched_entity_ids'] = ''
    
    print("Saving matching_results.tsv...")
    os.makedirs(os.path.dirname(out_matching), exist_ok=True)
    df.to_csv(out_matching, sep='\t', index=False)
    
    print("Saving candidate_pairs.tsv...")
    df.rename(columns={'matched_entity_ids': 'candidate_entity_ids'}).to_csv(out_candidate, sep='\t', index=False)
    
    print("Done! You can now validate and submit the files in the output/ directory.")

if __name__ == '__main__':
    create_baseline()
