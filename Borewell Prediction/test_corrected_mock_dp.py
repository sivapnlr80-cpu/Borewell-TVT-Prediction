import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# 1. Generate Corrected Mock Data
def generate_corrected_mock():
    os.makedirs("mock_train", exist_ok=True)
    
    def get_ref_gr(depths):
        gr = 50 + 30 * np.sin(depths / 100) + 15 * np.sin(depths / 20)
        gr[(depths > 2200) & (depths < 2600)] += 40
        gr[(depths > 3200) & (depths < 3500)] -= 30
        return np.clip(gr + np.random.normal(0, 3, len(depths)), 10, 150)

    # Typewell
    t_depths = np.arange(1000, 5000, 10, dtype=np.float32)
    gr_vals = get_ref_gr(t_depths)
    typewell_df = pd.DataFrame({'MD': t_depths, 'GR': gr_vals})
    typewell_df.to_csv("mock_train/WELL_1__typewell.csv", index=False)
    
    # Horizontal Well
    h_md = np.arange(4000, 8000, 5, dtype=np.float32)
    N = len(h_md)
    x = np.cumsum(np.random.uniform(3, 5, N)).astype(np.float32)
    y = (100 * np.sin(h_md / 500) + np.random.normal(0, 1, N)).astype(np.float32)
    z = (2400 + 120 * np.cos(h_md / 600) + np.random.normal(0, 2, N)).astype(np.float32)
    
    # TVT true
    tvt_true = (2000 + 15 * np.sin(h_md / 1000) + (z - 2400) * 0.95 + np.random.normal(0, 0.5, N)).astype(np.float32)
    
    # Horizontal GR is a function of tvt_true (stratigraphic position)
    h_gr = get_ref_gr(tvt_true)
    
    horiz_df = pd.DataFrame({
        'MD': h_md,
        'X': x,
        'Y': y,
        'Z': z,
        'GR': h_gr,
        'TVT_input': tvt_true
    })
    horiz_df.to_csv("mock_train/WELL_1__horizontal_well.csv", index=False)

generate_corrected_mock()

# 2. Run DP on the corrected mock data
h_df = pd.read_csv("mock_train/WELL_1__horizontal_well.csv")
t_df = pd.read_csv("mock_train/WELL_1__typewell.csv")

n_steps = len(h_df)
eval_start = int(n_steps * 0.7)
true_tvt = h_df['TVT_input'].values[eval_start:]

# DP algorithm
grid_step = 0.2
smoothness = 50.0
max_tvt_change_per_ft = 0.15

tw_tvt = t_df['MD'].values.astype(np.float64)
tw_gr = t_df['GR'].values.astype(np.float64)
si = np.argsort(tw_tvt)
tw_tvt, tw_gr = tw_tvt[si], tw_gr[si]

# Discretize states
tvt_min, tvt_max = tw_tvt.min(), tw_tvt.max()
states = np.arange(tvt_min, tvt_max + grid_step, grid_step)
n_states = len(states)

# Standardize GR logs
obs_gr = h_df['GR'].values
obs_gr_scaled = (obs_gr - obs_gr.mean()) / (obs_gr.std() + 1e-8)
tw_gr_scaled = (tw_gr - tw_gr.mean()) / (tw_gr.std() + 1e-8)
tw_gr_states = np.interp(states, tw_tvt, tw_gr_scaled)

INF = 1e15
cost = np.full(n_states, INF, dtype=np.float64)
backptr = np.full((n_steps, n_states), -1, dtype=np.int32)

last_known_tvt = h_df['TVT_input'].values[eval_start-1]
start_state = np.argmin(np.abs(states - last_known_tvt))
cost[start_state] = 0.0

z_vals = np.abs(h_df['Z'].values)
md_vals = h_df['MD'].values

for i in range(eval_start, n_steps):
    d_md = md_vals[i] - md_vals[i-1]
    d_z = z_vals[i] - z_vals[i-1]
    
    expected_shift_idx = int(round(d_z / grid_step))
    max_step_change = int(np.ceil(max_tvt_change_per_ft * d_md / grid_step)) + 2
    
    obs_gr_val = obs_gr_scaled[i]
    obs_cost_vec = (obs_gr_val - tw_gr_states) ** 2
    
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

path = np.zeros(n_steps, dtype=np.int32)
path[-1] = np.argmin(cost)
for i in range(n_steps - 1, eval_start, -1):
    path[i-1] = backptr[i, path[i]]
    
pred_tvt = states[path[eval_start:]]
rmse = np.sqrt(np.mean((pred_tvt - true_tvt)**2))
print(f"Corrected Mock Well DP RMSE = {rmse:.4f}")
