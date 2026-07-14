import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

def run_hybrid_dp(h_df, t_df, search_window=6.0, n_states=80, smoothness=20.0, max_tvt_change_per_ft=0.15):
    # Resolve columns
    h_md_col = 'MD'
    h_x_col = 'X'
    h_y_col = 'Y'
    h_z_col = 'Z'
    h_gr_col = 'GR'
    h_tvt_col = 'TVT_input'
    
    n_steps = len(h_df)
    eval_start = int(n_steps * 0.7)
    
    # 1. Fit Double-Ridge Ensemble to get base trend
    known_df = h_df.iloc[:eval_start]
    eval_df = h_df.iloc[eval_start:]
    
    y_train = known_df[h_tvt_col]
    
    # Model 1: Ridge scaled all coords
    X_cols1 = [h_md_col, h_x_col, h_y_col, h_z_col]
    X_train1 = known_df[X_cols1]
    mean1 = X_train1.mean(axis=0)
    std1 = X_train1.std(axis=0)
    std1[std1 == 0] = 1.0
    X_train1_scaled = (X_train1 - mean1) / std1
    X_all1_scaled = (h_df[X_cols1] - mean1) / std1
    model1 = Ridge(alpha=20.0).fit(X_train1_scaled, y_train)
    pred1 = model1.predict(X_all1_scaled)
    
    # Model 2: Ridge scaled MD, Z
    X_cols2 = [h_md_col, h_z_col]
    X_train2 = known_df[X_cols2]
    mean2 = X_train2.mean(axis=0)
    std2 = X_train2.std(axis=0)
    std2[std2 == 0] = 1.0
    X_train2_scaled = (X_train2 - mean2) / std2
    X_all2_scaled = (h_df[X_cols2] - mean2) / std2
    model2 = Ridge(alpha=10.0).fit(X_train2_scaled, y_train)
    pred2 = model2.predict(X_all2_scaled)
    
    # Base trend
    tvt_trend = 0.5 * pred1 + 0.5 * pred2
    
    # 2. Setup typewell reference
    tw_tvt = t_df['MD'].values.astype(np.float64)
    tw_gr = t_df['GR'].values.astype(np.float64)
    
    # Scale typewell TVT axis to match known TVT range for local validation
    known_tvt = h_df[h_tvt_col].values[:eval_start]
    tw_tvt_scaled = known_tvt.min() + (tw_tvt - tw_tvt.min()) * (known_tvt.max() - known_tvt.min()) / (tw_tvt.max() - tw_tvt.min())
    
    si = np.argsort(tw_tvt_scaled)
    tw_tvt_scaled, tw_gr = tw_tvt_scaled[si], tw_gr[si]
    interp_func = interp1d(tw_tvt_scaled, tw_gr, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Standardize observed GR
    obs_gr = h_df[h_gr_col].values
    obs_gr_s = savgol_filter(obs_gr, 11, 2) if len(obs_gr) > 15 else obs_gr
    gr_sigma = max(np.std(obs_gr_s), 1.0)
    
    # Standardize typewell GR reference
    tw_gr_mean, tw_gr_std = tw_gr.mean(), max(tw_gr.std(), 1.0)
    tw_gr_scaled = (tw_gr - tw_gr_mean) / tw_gr_std
    interp_func_scaled = interp1d(tw_tvt_scaled, tw_gr_scaled, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    obs_gr_scaled = (obs_gr_s - obs_gr_s.mean()) / gr_sigma
    
    # DP search grid offsets relative to the trend line
    offsets = np.linspace(-search_window, search_window, n_states)
    offset_step = offsets[1] - offsets[0]
    
    INF = 1e15
    cost = np.full(n_states, INF, dtype=np.float64)
    backptr = np.full((n_steps, n_states), -1, dtype=np.int32)
    
    # Initialize at boundary step using the exact known TVT
    last_known_tvt = h_df[h_tvt_col].values[eval_start-1]
    start_offset = last_known_tvt - tvt_trend[eval_start-1]
    start_state = np.argmin(np.abs(offsets - start_offset))
    cost[start_state] = 0.0
    
    # Step trajectory changes
    z_vals = np.abs(h_df[h_z_col].values)
    md_vals = h_df[h_md_col].values
    
    # Run DP only for the evaluation zone
    for i in range(eval_start, n_steps):
        d_md = md_vals[i] - md_vals[i-1]
        d_z = z_vals[i] - z_vals[i-1] # expected change in TVT
        
        # Expected shift in offsets: new_offset = old_offset + expected_offset_change
        expected_offset_change = d_z - (tvt_trend[i] - tvt_trend[i-1])
        expected_shift_idx = int(round(expected_offset_change / offset_step))
        
        max_step_change = int(np.ceil(max_tvt_change_per_ft * d_md / offset_step)) + 2
        
        obs_gr_val = obs_gr_scaled[i]
        
        # Candidate TVT values at each offset state
        candidate_tvts = tvt_trend[i] + offsets
        ref_gr = interp_func_scaled(candidate_tvts)
        obs_cost_vec = (obs_gr_val - ref_gr) ** 2
        
        # Vectorized transition search
        shifted_costs = []
        prev_indices = []
        
        for k in range(-max_step_change, max_step_change + 1):
            shift = expected_shift_idx + k
            
            if shift > 0:
                c = np.concatenate([np.full(shift, INF), cost[:-shift]])
            elif shift < 0:
                c = np.concatenate([cost[-shift:], np.full(-shift, INF)])
            else:
                c = cost.copy()
                
            trans = smoothness * ((k * offset_step) ** 2)
            shifted_costs.append(c + trans)
            
            prev_s = np.arange(n_states) - shift
            prev_s = np.clip(prev_s, 0, n_states - 1)
            prev_indices.append(prev_s)
            
        shifted_costs = np.array(shifted_costs)
        best_shift_idx = np.argmin(shifted_costs, axis=0)
        
        cost = shifted_costs[best_shift_idx, np.arange(n_states)] + obs_cost_vec
        prev_indices = np.array(prev_indices)
        backptr[i, :] = prev_indices[best_shift_idx, np.arange(n_states)]
        
    # Backtrack
    path = np.zeros(n_steps, dtype=np.int32)
    path[-1] = np.argmin(cost)
    for i in range(n_steps - 1, eval_start, -1):
        path[i-1] = backptr[i, path[i]]
        
    pred_tvt = tvt_trend[eval_start:] + offsets[path[eval_start:]]
    
    if len(pred_tvt) > 15:
        pred_tvt = savgol_filter(pred_tvt, 15, 2)
        
    return pred_tvt

# Evaluate on all training wells
rmses = []
for f in sorted(glob.glob('train/*_horizontal_well.csv')):
    wellname = os.path.basename(f).split('__')[0]
    h_df = pd.read_csv(f)
    t_df = pd.read_csv(f.replace('horizontal_well', 'typewell'))
    
    n = len(h_df)
    eval_start = int(n * 0.7)
    true_tvt = h_df['TVT_input'].values[eval_start:]
    
    pred = run_hybrid_dp(h_df, t_df)
    rmse = np.sqrt(np.mean((pred - true_tvt)**2))
    rmses.append(rmse)
    print(f"Well {wellname}: Hybrid DP RMSE = {rmse:.4f}")
    
print(f"\nAverage Hybrid DP RMSE = {np.mean(rmses):.4f}")
