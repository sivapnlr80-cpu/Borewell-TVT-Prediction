import os, glob
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.signal import savgol_filter

def run_local_inference():
    test_dir = 'test'
    test_files = sorted(glob.glob(os.path.join(test_dir, '*_horizontal_well.csv')))
    print(f"[+] Local validation: found {len(test_files)} mock test wells.")
    
    submission_rows = []
    for f in test_files:
        wellname = os.path.basename(f).split('__')[0]
        df = pd.read_csv(f)
        
        # Split known vs eval sections
        known_mask = df['TVT_input'].notna()
        eval_mask = df['TVT_input'].isna()
        eval_indices = np.where(eval_mask.values)[0]
        
        known_df = df[known_mask]
        eval_df = df[eval_mask]
        
        y_train = known_df['TVT_input']
        
        # Model 1: Ridge scaled all coords
        X_cols1 = ['MD', 'X', 'Y', 'Z']
        X_train1 = known_df[X_cols1]
        X_eval1 = eval_df[X_cols1]
        mean1 = X_train1.mean(axis=0)
        std1 = X_train1.std(axis=0)
        std1[std1 == 0] = 1.0
        X_train1_scaled = (X_train1 - mean1) / std1
        X_eval1_scaled = (X_eval1 - mean1) / std1
        model1 = Ridge(alpha=20.0).fit(X_train1_scaled, y_train)
        preds1 = model1.predict(X_eval1_scaled)
        
        # Model 2: Ridge scaled MD, Z
        X_cols2 = ['MD', 'Z']
        X_train2 = known_df[X_cols2]
        X_eval2 = eval_df[X_cols2]
        mean2 = X_train2.mean(axis=0)
        std2 = X_train2.std(axis=0)
        std2[std2 == 0] = 1.0
        X_train2_scaled = (X_train2 - mean2) / std2
        X_eval2_scaled = (X_eval2 - mean2) / std2
        model2 = Ridge(alpha=10.0).fit(X_train2_scaled, y_train)
        preds2 = model2.predict(X_eval2_scaled)
        
        # Ensemble
        preds = 0.5 * preds1 + 0.5 * preds2
        if len(preds) > 15:
            preds = savgol_filter(preds, window_length=11, polyorder=2)
            
        print(f"  Mock Well {wellname}: range [{preds.min():.4f}, {preds.max():.4f}], length {len(preds)}")
        
        for i, idx in enumerate(eval_indices):
            submission_rows.append({'id': f'{wellname}_{idx}', 'tvt': preds[i]})
            
    sub = pd.DataFrame(submission_rows)
    print(f"\n[+] Generated local submission file with {len(sub)} rows.")
    print(sub.head(5).to_string())

run_local_inference()
