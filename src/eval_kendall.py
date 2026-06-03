import argparse
import pandas as pd
import numpy as np
from scipy.stats import kendalltau

def calculate_weighted_kendall_tau(predictions_path='data/val_predictions.csv'):
    """
    Calculates the mean Kendall's tau_b weighted by holdout size n_b,
    comparing predicted ranks against true scores per branch.
    """
    print(f"Loading validation predictions from {predictions_path}...")
    df = pd.read_csv(predictions_path)
    
    total_weighted_tau = 0.0
    total_weight = 0
    skipped_branches = 0
    
    # Iterate through each branch
    for branch_id, group in df.groupby('branch_id'):
        n_b = len(group)
        
        # Skip branches with fewer than 2 rows (cannot compute tau)
        if n_b < 2:
            skipped_branches += 1
            continue
            
        # The assignment defines a concordant pair as one where 
        # (r_i - r_j) and (s_j - s_i) have the SAME sign.
        # Standard scipy kendalltau checks if (X_i - X_j) and (Y_i - Y_j) have the SAME sign.
        # Because lower rank is better but higher score is better, they naturally move 
        # in opposite directions. Multiplying the standard tau by -1 aligns the metric 
        # with the assignment's definition (where +1.0 is perfect ordering).
        tau_b, _ = kendalltau(group['pred_rank'], group['score'])
        tau_b = -tau_b 
        
        # Handle edge cases where tau_b might be NaN (e.g., all scores are perfectly identical)
        if np.isnan(tau_b):
            tau_b = 0.0
            
        total_weighted_tau += n_b * tau_b
        total_weight += n_b
        print(f"Branch {branch_id:2d} | n={n_b:2d} | Tau: {tau_b: .3f}")
        
    # Compute the final weighted mean
    final_tau = total_weighted_tau / total_weight if total_weight > 0 else 0.0
    
    print("-" * 30)
    print("EVALUATION RESULTS")
    print("-" * 30)
    print(f"Total valid branches: {len(df['branch_id'].unique()) - skipped_branches}")
    print(f"Skipped branches (n < 2): {skipped_branches}")
    print(f"Total rows evaluated: {total_weight}")
    print(f"Weighted Mean Kendall's Tau: {final_tau:.5f}")
    
    return final_tau

if __name__ == "__main__":
    # Run the evaluation
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_path', type=str, default='data/val_predictions.csv',
                        help='Path to the CSV file containing validation predictions.')
    args = parser.parse_args()
    predictions_path = args.predictions_path
    calculate_weighted_kendall_tau(predictions_path)