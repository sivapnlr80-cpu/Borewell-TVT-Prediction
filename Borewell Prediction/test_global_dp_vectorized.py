import os
import glob
import numpy as np
import pandas as pd

def run_global_dp_vectorized(h_df, t_df, grid_step=0.2, smoothness=10.0, max_tvt_change_per_ft=0.15):
    # Resolve columns
    h_md_col = 'MD'
    h_z_col = 'Z'
    h_gr_col = 'GR'
    h_tvt_col = 'TVT_input'
    
    n_steps = len(h_df)
    eval_start = int(n_steps * 0.7)
    
    # 1. Setup typewell reference
    tw_tvt = t_df['MD'].values.astype(np.float64)
    tw_gr = t_df['GR'].values.astype(np.float64)
    
    # Scale typewell TVT axis to match known TVT range for local validation
    known_tvt = h_df[h_tvt_col].values[:eval_start]
    tw_tvt_scaled = known_tvt.min() + (tw_tvt - tw_tvt.min()) * (known_tvt.max() - known_tvt.min()) / (tw_tvt.max() - tw_tvt.min())
    
    si = np.argsort(tw_tvt_scaled)
    tw_tvt_scaled, tw_gr = tw_tvt_scaled[si], tw_gr[si]
    
    # Create discretized TVT states
    tvt_min = tw_tvt_scaled.min()
    tvt_max = tw_tvt_scaled.max()
    states = np.arange(tvt_min, tvt_max + grid_step, grid_step)
    n_states = len(states)
    
    # Interpolate typewell GR onto our states
    tw_gr_states = np.interp(states, tw_tvt_scaled, tw_gr)
    
    # Denoise/normalize GR
    obs_gr = h_df[h_gr_col].values
    gr_sigma = max(np.std(obs_gr), 1.0)
    
    # 2. Forward DP
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
    
    # Run DP only for the evaluation zone (from eval_start to end)
    for i in range(eval_start, n_steps):
        d_md = md_vals[i] - md_vals[i-1]
        d_z = z_vals[i] - z_vals[i-1] # expected change in TVT
        
        # Expected shift in indices
        expected_shift_idx = int(round(d_z / grid_step))
        
        # Max index change for this specific step
        max_step_change = int(np.ceil(max_tvt_change_per_ft * d_md / grid_step)) + 2
        
        obs_gr_val = obs_gr[i]
        obs_cost_vec = ((obs_gr_val - tw_gr_states) / gr_sigma) ** 2
        
        # Vectorized transition search
        shifted_costs = []
        prev_indices = []
        
        for k in range(-max_step_change, max_step_change + 1):
            shift = expected_shift_idx + k
            
            # Construct candidate previous cost vector shifted by `shift`
            if shift > 0:
                c = np.concatenate([np.full(shift, INF), cost[:-shift]])
            elif shift < 0:
                c = np.concatenate([cost[-shift:], np.full(-shift, INF)])
            else:
                c = cost.copy()
                
            # Transition cost penalty
            trans = smoothness * ((k * grid_step) ** 2)
            shifted_costs.append(c + trans)
            
            prev_s = np.arange(n_states) - shift
            prev_s = np.clip(prev_s, 0, n_states - 1)
            prev_indices.append(prev_s)
            
        shifted_costs = np.array(shifted_costs)
        best_shift_idx = np.argmin(shifted_costs, axis=0)
        
        cost = shifted_costs[best_shift_idx, np.arange(n_states)] + obs_cost_vec
        
        prev_indices = np.array(prev_indices)
        backptr[i, :] = prev_indices[best_shift_idx, np.arange(n_states)]
        
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
    
    pred = run_global_dp_vectorized(h_df, t_df)
    rmse = np.sqrt(np.mean((pred - true_tvt)**2))
    rmses.append(rmse)
    print(f"Well {wellname}: DP Vectorized RMSE = {rmse:.4f}")
    
print(f"\nAverage DP Vectorized RMSE = {np.mean(rmses):.4f}")
