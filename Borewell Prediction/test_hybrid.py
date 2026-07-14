import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

def dp_track_evaluation_zone(h_df, t_df, n_states=300, smoothness=1.0, trend_weight=0.05):
    # Resolve columns
    tw_depth_col = 'MD'
    tw_gr_col = 'GR'
    h_gr_col = 'GR'
    h_tvt_col = 'TVT_input'
    h_z_col = 'Z'
    h_md_col = 'MD'
    
    n = len(h_df)
    eval_start = int(n * 0.7)
    
    # 1. Fit geometric trend on known section
    known_df = h_df.iloc[:eval_start]
    known_y = known_df[h_tvt_col].values + known_df[h_z_col].values
    known_md = known_df[h_md_col].values
    
    slope, intercept = np.polyfit(known_md, known_y, 1)
    
    # Predict trend for the entire well
    all_md = h_df[h_md_col].values
    all_z = h_df[h_z_col].values
    tvt_trend = (slope * all_md + intercept) - all_z
    
    # 2. Setup typewell reference
    tw_tvt = t_df[tw_depth_col].values.astype(np.float64)
    tw_gr = t_df[tw_gr_col].values.astype(np.float64)
    si = np.argsort(tw_tvt)
    tw_tvt, tw_gr = tw_tvt[si], tw_gr[si]
    interp_func = interp1d(tw_tvt, tw_gr, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # 3. Setup DP search grid
    eval_trend = tvt_trend[eval_start-1:]
    search_min = min(eval_trend) - 20.0
    search_max = max(eval_trend) + 20.0
    search_min = max(tw_tvt.min(), search_min)
    search_max = min(tw_tvt.max(), search_max)
    
    tvt_grid = np.linspace(search_min, search_max, n_states)
    tvt_step = tvt_grid[1] - tvt_grid[0]
    
    # Denoise observed GR
    obs_gr = h_df[h_gr_col].values.astype(np.float64)
    obs_gr_s = savgol_filter(obs_gr, 11, 2)
    gr_sigma = max(np.std(obs_gr_s), 1.0)
    
    # Run DP only for the evaluation zone
    # Number of steps in DP is len(eval_zone) + 1 (including the boundary step)
    eval_gr = obs_gr_s[eval_start-1:]
    eval_trend_dp = tvt_trend[eval_start-1:]
    n_steps = len(eval_gr)
    
    INF = 1e15
    cost = np.full(n_states, INF, dtype=np.float64)
    backptr = np.full((n_steps, n_states), -1, dtype=np.int32)
    
    # Initialize at step 0 (which is eval_start - 1) using the exact known TVT
    last_known_tvt = h_df[h_tvt_col].values[eval_start-1]
    start_state = np.argmin(np.abs(tvt_grid - last_known_tvt))
    cost[start_state] = 0.0
    
    # Max allowed change per step
    max_dtvt = 1.5
    max_jump = max(1, int(np.ceil(max_dtvt / tvt_step)))
    
    ref_gr = interp_func(tvt_grid)
    
    # Forward pass
    for i in range(1, n_steps):
        new_cost = np.full(n_states, INF, dtype=np.float64)
        for s in range(n_states):
            lo = max(0, s - max_jump)
            hi = min(n_states, s + max_jump + 1)
            
            preds = np.arange(lo, hi)
            trans = smoothness * ((tvt_grid[s] - tvt_grid[preds]) ** 2)
            totals = cost[lo:hi] + trans
            
            bi = np.argmin(totals)
            new_cost[s] = totals[bi]
            backptr[i, s] = lo + bi
            
        # Add observation cost and trend guidance cost
        obs_cost = ((eval_gr[i] - ref_gr) / gr_sigma) ** 2
        trend_cost = trend_weight * ((tvt_grid - eval_trend_dp[i]) ** 2)
        new_cost += obs_cost + trend_cost
        cost = new_cost
        
    # Backtrack
    path = np.zeros(n_steps, dtype=np.int32)
    path[-1] = np.argmin(cost)
    for i in range(n_steps - 2, -1, -1):
        path[i] = backptr[i + 1, path[i + 1]]
        
    pred_tvt = tvt_grid[path]
    return pred_tvt, tvt_trend[eval_start-1:]

# Load data and run
h_df = pd.read_csv('train/WELL_1__horizontal_well.csv')
t_df = pd.read_csv('train/WELL_1__typewell.csv')
pred, trend = dp_track_evaluation_zone(h_df, t_df)

n = len(h_df)
eval_start = int(n * 0.7)
true_tvt = h_df['TVT_input'].values[eval_start-1:]

rmse_dp = np.sqrt(np.mean((pred - true_tvt)**2))
rmse_trend = np.sqrt(np.mean((trend - true_tvt)**2))

print(f"WELL_1: Geometric Trend RMSE = {rmse_trend:.4f}")
print(f"WELL_1: DP Alignment RMSE     = {rmse_dp:.4f}")
print(f"True range: [{true_tvt.min():.2f}, {true_tvt.max():.2f}]")
print(f"Pred range: [{pred.min():.2f}, {pred.max():.2f}]")
