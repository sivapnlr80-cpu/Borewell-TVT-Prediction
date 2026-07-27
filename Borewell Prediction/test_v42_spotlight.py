"""
Spotlight Test for V42: 000d7d20, 00bbac68, 00e12e8b
Runs Particle Filter with:
- Propagate in absolute TVT space with structural dip slope +0.0015 ft/ft
- Boundary Reflection (2 * bound - val) at TVT boundaries
- BUDA hard lock (+10 ft) for 00e12e8b
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from scipy.signal import savgol_filter

np.random.seed(42)

STRUCTURAL_DIP_SLOPE = 0.0015
BUDA_OFFSET_FT = 10.0

WELL_WINDOW = {
    '000d7d20': 5.0,    # EGFDL ~11 ft thick; +-5 ft keeps us inside
    '00bbac68': 8.0,    # brushing BUDA boundary; +-8 ft window
    '00e12e8b': None,   # BUDA hard lock
}

BUDA_GR_THRESH = 45.0
GR_COLLAPSE_JITTER = 2.0

def run_particle_filter_reflection(
    tvt_trend,
    dmd_eval,
    obs_gr_raw,
    obs_gr_scaled_eval,
    interp_gr_fn,
    tw_tvt_vals,
    tvt_lo,
    tvt_hi,
    n_particles=800,
    init_offset=0.0,
    init_std=0.5,
    Q_std=0.018,
    GR_noise_std=0.35,
):
    n = len(tvt_trend)
    particles = np.random.normal(init_offset, init_std, n_particles)
    weights = np.ones(n_particles) / n_particles
    x_filt = np.zeros(n)

    for k in range(n):
        # 1. Propagate in Absolute TVT Space with structural slope & process noise
        drift = STRUCTURAL_DIP_SLOPE * dmd_eval[k]
        particles += drift + np.random.normal(0.0, Q_std, n_particles)

        tvt_k = tvt_trend[k] + particles

        # 2. Boundary Reflection in Absolute TVT Space
        if np.isfinite(tvt_lo) and np.isfinite(tvt_hi):
            # Reflection at lower bound
            below = tvt_k < tvt_lo
            if np.any(below):
                tvt_k[below] = 2.0 * tvt_lo - tvt_k[below]
            # Reflection at upper bound
            above = tvt_k > tvt_hi
            if np.any(above):
                tvt_k[above] = 2.0 * tvt_hi - tvt_k[above]
            # Safety clip
            tvt_k = np.clip(tvt_k, tvt_lo, tvt_hi)

        tvt_k = np.clip(tvt_k, tw_tvt_vals.min(), tw_tvt_vals.max())
        particles = tvt_k - tvt_trend[k]

        # 3. Likelihood update
        pred_gr = interp_gr_fn(tvt_k)
        obs = obs_gr_scaled_eval[k]
        innov = obs - pred_gr

        log_w = -0.5 * (innov / GR_noise_std) ** 2
        log_w -= log_w.max()
        weights = np.exp(log_w) + 1e-300
        weights /= weights.sum()

        # 4. Resampling
        N_eff = 1.0 / np.sum(weights ** 2)
        in_buda = obs_gr_raw[k] < BUDA_GR_THRESH

        if N_eff < n_particles * 0.4:
            cumsum = np.cumsum(weights)
            pos = (np.arange(n_particles) + np.random.uniform()) / n_particles
            idxs = np.clip(np.searchsorted(cumsum, pos), 0, n_particles - 1)
            particles = particles[idxs]
            weights[:] = 1.0 / n_particles

            if in_buda:
                particles += np.random.uniform(
                    -GR_COLLAPSE_JITTER, GR_COLLAPSE_JITTER, n_particles
                )

        x_filt[k] = np.dot(weights, particles)

    return x_filt

spotlight = ['000d7d20', '00bbac68', '00e12e8b']
train_dir = 'competition_data/train'

print(f"{'Well':<15} {'Trend RMSE':>12} {'PF (Reflect) RMSE':>18} {'Delta':>10} {'MSE':>10}")
print("-" * 70)

for wellname in spotlight:
    f = os.path.join(train_dir, f"{wellname}__horizontal_well.csv")
    tw_file = os.path.join(train_dir, f"{wellname}__typewell.csv")

    df = pd.read_csv(f)
    t_df = pd.read_csv(tw_file).dropna(subset=['TVT', 'GR'])
    tw_tvt = t_df['TVT'].values.astype(np.float64)
    tw_gr = t_df['GR'].values.astype(np.float64)
    si = np.argsort(tw_tvt)
    tw_tvt, tw_gr = tw_tvt[si], tw_gr[si]

    tw_gr_mean = tw_gr.mean()
    tw_gr_std = max(tw_gr.std(), 1.0)
    tw_gr_scaled = np.clip((tw_gr - tw_gr_mean) / tw_gr_std, -3.0, 3.0)
    interp_gr = interp1d(
        tw_tvt, tw_gr_scaled, kind='linear', bounds_error=False, fill_value='extrapolate'
    )

    known_mask = df['TVT_input'].notna()
    eval_start = df[known_mask].index[-1] + 1
    eval_mask = df.index >= eval_start
    known_df = df[known_mask]
    eval_df = df[eval_mask]

    df[['X', 'Y', 'Z']] = df[['X', 'Y', 'Z']].interpolate().bfill().ffill()
    df['GR'] = df['GR'].interpolate().bfill().ffill()
    df['MD'] = df['MD'].interpolate().bfill().ffill()

    obs_gr = df['GR'].values
    obs_gr_scaled = np.clip((obs_gr - tw_gr_mean) / tw_gr_std, -3.0, 3.0)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train = poly.fit_transform(known_df[['X', 'Y', 'Z']])
    y_train = known_df['TVT_input'].values

    mf = X_train.mean(axis=0)
    sf = X_train.std(axis=0)
    sf[sf == 0] = 1.0
    ridge = Ridge(alpha=10.0).fit((X_train - mf) / sf, y_train)

    X_eval_raw = poly.transform(eval_df[['X', 'Y', 'Z']])
    tvt_trend = ridge.predict((X_eval_raw - mf) / sf)

    true_tvt = eval_df['TVT'].values
    trend_rmse = np.sqrt(np.mean((tvt_trend - true_tvt) ** 2))

    landing_tvt = known_df['TVT_input'].iloc[-1]

    if wellname == '00e12e8b':
        buda_lock_tvt = landing_tvt + BUDA_OFFSET_FT
        pf_tvt = np.full(len(eval_df), buda_lock_tvt)
        pf_rmse = np.sqrt(np.mean((pf_tvt - true_tvt) ** 2))
        delta = trend_rmse - pf_rmse
        print(f"{wellname:<15} {trend_rmse:>12.4f} {pf_rmse:>18.4f} {delta:>+10.4f} {pf_rmse**2:>10.4f} (BUDA lock)")
        continue

    half_win = WELL_WINDOW[wellname]
    tvt_lo = landing_tvt - half_win
    tvt_hi = landing_tvt + half_win

    X_last = poly.transform(known_df[['X', 'Y', 'Z']].iloc[[-1]])
    last_pred = ridge.predict((X_last - mf) / sf)[0]
    init_offset = landing_tvt - last_pred

    md_vals = df['MD'].values
    eval_idx = eval_df.index - df.index[0]
    md_eval = md_vals[eval_idx]
    dmd_eval = np.diff(md_eval, prepend=md_eval[0])

    eval_gr_scaled = obs_gr_scaled[eval_idx]
    eval_gr_raw = obs_gr[eval_idx]

    offsets = run_particle_filter_reflection(
        tvt_trend=tvt_trend,
        dmd_eval=dmd_eval,
        obs_gr_raw=eval_gr_raw,
        obs_gr_scaled_eval=eval_gr_scaled,
        interp_gr_fn=interp_gr,
        tw_tvt_vals=tw_tvt,
        tvt_lo=tvt_lo,
        tvt_hi=tvt_hi,
        n_particles=800,
        init_offset=init_offset,
        init_std=0.5,
        Q_std=0.018,
        GR_noise_std=0.35,
    )

    pf_tvt = tvt_trend + offsets
    if np.isfinite(tvt_lo) and np.isfinite(tvt_hi):
        below = pf_tvt < tvt_lo
        if np.any(below): pf_tvt[below] = 2.0 * tvt_lo - pf_tvt[below]
        above = pf_tvt > tvt_hi
        if np.any(above): pf_tvt[above] = 2.0 * tvt_hi - pf_tvt[above]
        pf_tvt = np.clip(pf_tvt, tvt_lo, tvt_hi)

    if len(pf_tvt) > 11:
        pf_tvt = savgol_filter(pf_tvt, window_length=11, polyorder=2)

    if np.isfinite(tvt_lo) and np.isfinite(tvt_hi):
        pf_tvt = np.clip(pf_tvt, tvt_lo, tvt_hi)

    pf_rmse = np.sqrt(np.mean((pf_tvt - true_tvt) ** 2))
    delta = trend_rmse - pf_rmse
    print(f"{wellname:<15} {trend_rmse:>12.4f} {pf_rmse:>18.4f} {delta:>+10.4f} {pf_rmse**2:>10.4f}")
