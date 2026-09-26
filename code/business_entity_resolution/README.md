# Entity Resolution Pipeline

## Instructions
1. Install dependencies:
`pip install -r requirements.txt`

2. Run the pipeline:
`cd src`
`python main.py`

This will generate `candidate_pairs.tsv` and `matching_results.tsv` in the `output/` directory.

## Pipeline Steps
1. **Blocking (blocking.py):** Joins datasets on postal code and name tokens to generate a reduced set of candidates.
2. **Feature Engineering (features.py):** Calculates Jaccard distances and other pairwise features for candidates.
3. **Training (train.py):** Trains an XGBoost classifier optimized for the F0.5 score using ground truth data.
4. **Prediction (predict.py):** Predicts matches for the candidate pairs using a high probability threshold.
