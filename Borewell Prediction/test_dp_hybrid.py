import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression

def run_hybrid_geosteering(h_df, t_df, n_states=80, smoothness=1.0, window_width=15.0):
    # Resolve columns
    tw_depth_col = 'MD'
    tw_gr_col = 'GR'
    h_gr_col = 'GR'
    h_tvt_col = 'TVT_input'
    h_z_col = 'Z'
    h_md_col = 'MD'
    
    n = len(h_df)
    eval_start = int(n * 0.7)
    
    # 1. Fit linear regression trend on known section
    known_df = h_df.iloc[:eval_start]
    lr = LinearRegression().fit(known_df[[h_md_col, h_z_col]], known_df[h_tvt_col])
    
    # Predict trend for the entire well
    all_md_z = h_df[[h_md_col, h_z_col]].values
    tvt_trend = lr.predict(all_md_z)
    
    # 2. Setup typewell reference
    tw_tvt = t_df[tw_depth_col].values.astype(np.float64)
    tw_gr = t_df[tw_gr_col].values.astype(np.float64)
    si = np.argsort(tw_tvt)
    tw_tvt, tw_gr = tw_tvt[si], tw_gr[si]
    interp_func = interp1d(tw_tvt, tw_gr, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # 3. DP alignment only on the evaluation zone
    eval_trend = tvt_trend[eval_start-1:]
    eval_gr = h_df[h_gr_col].values[eval_start-1:]
    
    # Denoise observed GR
    eval_gr_s = savgol_filter(eval_gr, min(11, len(eval_gr)-1 if len(eval_gr)%2==0 else len(eval_gr)), 2)
    gr_sigma = max(np.std(eval_gr_s), 1.0)
    
    n_steps = len(eval_gr)
    
    # State grid offset (relative to trend)
    offsets = np.linspace(-window_width, window_width, n_states)
    offset_step = offsets[1] - offsets[0]
    
    INF = 1e15
    cost = np.full(n_states, INF, dtype=np.float64)
    backptr = np.full((n_steps, n_states), -1, dtype=np.int32)
    
    # Initialize at step 0 (which is eval_start - 1) using the exact known TVT
    last_known_tvt = h_df[h_tvt_col].values[eval_start-1]
    start_offset = last_known_tvt - eval_trend[0]
    start_state = np.argmin(np.abs(offsets - start_offset))
    cost[start_state] = 0.0
    
    # Max allowed change in offset per step (geological dip variance constraint)
    max_d_offset = 1.0
    max_jump = max(1, int(np.ceil(max_d_offset / offset_step)))
    
    # Forward pass
    for i in range(1, n_steps):
        # Precompute reference GR for candidate TVT values at step i
        candidate_tvts = eval_trend[i] + offsets
        ref_gr = interp_func(candidate_tvts)
        obs_cost_vec = ((eval_gr_s[i] - ref_gr) / gr_sigma) ** 2
        
        new_cost = np.full(n_states, INF, dtype=np.float64)
        for s in range(n_states):
            lo = max(0, s - max_jump)
            hi = min(n_states, s + max_jump + 1)
            
            preds = np.arange(lo, hi)
            # Smoothness penalty on offset changes
            trans = smoothness * ((offsets[s] - offsets[preds]) ** 2)
            totals = cost[lo:hi] + trans
            
            bi = np.argmin(totals)
            new_cost[s] = totals[bi] + obs_cost_vec[s]
            backptr[i, s] = lo + bi
            
        cost = new_cost
        
    # Backtrack
    path = np.zeros(n_steps, dtype=np.int32)
    path[-1] = np.argmin(cost)
    for i in range(n_steps - 2, -1, -1):
        path[i] = backptr[i + 1, path[i + 1]]
        
    pred_tvt = eval_trend + offsets[path]
    return pred_tvt

# Evaluate on all training wells
rmses = []
for f in sorted(glob.glob('train/*_horizontal_well.csv')):
    wellname = os.path.basename(f).split('__')[0]
    h_df = pd.read_csv(f)
    t_df = pd.read_csv(f.replace('horizontal_well', 'typewell'))
    
    n = len(h_df)
    eval_start = int(n * 0.7)
    true_tvt = h_df['TVT_input'].values[eval_start-1:]
    
    pred = run_hybrid_geosteering(h_df, t_df)
    rmse = np.sqrt(np.mean((pred - true_tvt)**2))
    rmses.append(rmse)
    print(f"Well {wellname}: Hybrid RMSE = {rmse:.4f}")

print(f"\nAverage Hybrid RMSE across all train wells = {np.mean(rmses):.4f}")
