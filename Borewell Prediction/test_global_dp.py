import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def run_global_dp(h_df, t_df, grid_step=0.2, smoothness=100.0, max_tvt_change_per_ft=1.0):
    # Resolve columns
    h_md_col = 'MD'
    h_z_col = 'Z'
    h_gr_col = 'GR'
    h_tvt_col = 'TVT_input'
    
    # 1. Setup typewell reference
    tw_tvt = t_df['MD'].values.astype(np.float64) # in typewell, depth is MD/TVT
    tw_gr = t_df['GR'].values.astype(np.float64)
    si = np.argsort(tw_tvt)
    tw_tvt, tw_gr = tw_tvt[si], tw_gr[si]
    
    # Create discretized TVT states
    tvt_min = tw_tvt.min()
    tvt_max = tw_tvt.max()
    states = np.arange(tvt_min, tvt_max + grid_step, grid_step)
    n_states = len(states)
    
    # Interpolate typewell GR onto our states
    tw_gr_states = np.interp(states, tw_tvt, tw_gr)
    
    # Denoise/normalize GR to make cost scale-invariant
    obs_gr = h_df[h_gr_col].values
    gr_sigma = max(np.std(obs_gr), 1.0)
    
    n_steps = len(h_df)
    eval_start = int(n_steps * 0.7)
    
    # 2. Forward DP
    # cost[s] is the min cost to be at state s at the current step
    INF = 1e15
    cost = np.full(n_states, INF, dtype=np.float64)
    backptr = np.full((n_steps, n_states), -1, dtype=np.int32)
    
    # Initialize at step 0 using the exact known TVT
    last_known_tvt = h_df[h_tvt_col].values[eval_start-1]
    start_state = np.argmin(np.abs(states - last_known_tvt))
    cost[start_state] = 0.0
    
    # Step trajectory changes
    z_vals = np.abs(h_df[h_z_col].values)
    md_vals = h_df[h_md_col].values
    
    # Maximum transition index offset based on MD step and max speed
    # Average MD step is 5.0 ft, so max TVT change is ~5.0 ft
    max_idx_change = int(np.ceil(max_tvt_change_per_ft * 5.0 / grid_step)) + 2
    
    # Run DP only for the evaluation zone (from eval_start to end)
    for i in range(eval_start, n_steps):
        d_md = md_vals[i] - md_vals[i-1]
        d_z = z_vals[i] - z_vals[i-1] # expected change in TVT
        
        # Max index change for this specific step
        max_step_change = int(np.ceil(max_tvt_change_per_ft * d_md / grid_step)) + 2
        
        obs_gr_val = obs_gr[i]
        # Precompute observation cost vector for all states
        obs_cost_vec = ((obs_gr_val - tw_gr_states) / gr_sigma) ** 2
        
        new_cost = np.full(n_states, INF, dtype=np.float64)
        
        # Vectorized transition search for speed
        for s in range(n_states):
            # We search for previous state s' such that |states[s] - states[s'] - d_z| is small
            # Center of search is state corresponding to states[s] - d_z
            prev_tvt_center = states[s] - d_z
            center_idx = int(round((prev_tvt_center - tvt_min) / grid_step))
            
            lo = max(0, center_idx - max_step_change)
            hi = min(n_states, center_idx + max_step_change + 1)
            
            if lo >= hi:
                continue
                
            prev_states_idx = np.arange(lo, hi)
            # Transition cost
            trans_cost = smoothness * ((states[s] - states[prev_states_idx] - d_z) ** 2)
            totals = cost[prev_states_idx] + trans_cost
            
            bi = np.argmin(totals)
            new_cost[s] = totals[bi] + obs_cost_vec[s]
            backptr[i, s] = prev_states_idx[bi]
            
        cost = new_cost
        
    # 3. Backtrack
    path = np.zeros(n_steps, dtype=np.int32)
    path[-1] = np.argmin(cost)
    for i in range(n_steps - 1, eval_start, -1):
        path[i-1] = backptr[i, path[i]]
        
    pred_tvt = states[path[eval_start:]]
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
    
    pred = run_global_dp(h_df, t_df)
    rmse = np.sqrt(np.mean((pred - true_tvt)**2))
    rmses.append(rmse)
    print(f"Well {wellname}: DP RMSE = {rmse:.4f}")
    
print(f"\nAverage DP RMSE = {np.mean(rmses):.4f}")
