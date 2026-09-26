import pandas as pd

def compute_f05_score(predictions_file, ground_truth_file):
    print(f"Evaluating {predictions_file} against {ground_truth_file}")
    
    preds = pd.read_csv(predictions_file, sep='\t')
    gt = pd.read_csv(ground_truth_file, sep='\t')
    
    # Aggregate ground truth matches per source1_entity_id
    if 'matched_entity_id' in gt.columns:
        gt = gt.groupby('source1_entity_id')['matched_entity_id'].apply(lambda x: ','.join(x.dropna().astype(str))).reset_index()
        gt = gt.rename(columns={'matched_entity_id': 'matched_entity_ids'})
    
    # Merge predictions with ground truth
    df = pd.merge(gt, preds, on='source1_entity_id', how='left')
    df['matched_entity_ids_x'] = df['matched_entity_ids_x'].fillna('')
    df['matched_entity_ids_y'] = df['matched_entity_ids_y'].fillna('')
    
    def calc_f05(row):
        gt_list = set(str(row['matched_entity_ids_x']).split(',')) - {''}
        pred_list = set(str(row['matched_entity_ids_y']).split(',')) - {''}
        
        tp = len(gt_list.intersection(pred_list))
        fp = len(pred_list - gt_list)
        fn = len(gt_list - pred_list)
        
        if len(gt_list) == 0 and len(pred_list) == 0: return 1.0
        if len(gt_list) == 0 and len(pred_list) > 0: return 0.0
            
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall == 0: return 0.0
        return (1.25 * precision * recall) / (0.25 * precision + recall)

    df['f05'] = df.apply(calc_f05, axis=1)
    macro_f05 = df['f05'].mean()
    print(f"\nMacro-Averaged F0.5 Score: {macro_f05:.4f}")
    return macro_f05

if __name__ == '__main__':
    compute_f05_score(
        '../../../output/matching_results.tsv',
        '../../../dataset/test/test_ground_truth.tsv'
    )
