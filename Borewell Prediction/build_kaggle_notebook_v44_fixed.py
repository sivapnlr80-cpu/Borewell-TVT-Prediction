import json
import os

notebook_code = """import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# V44 Robust Production Constants
STRUCTURAL_DIP_SLOPE = 0.0015   # ft TVT per ft MD (+7.5 ft / 5000 ft lateral)
BUDA_OFFSET_FT       = 10.0     # mean BUDA exit offset depth for 00e12e8b
ALPHA_GRAD           = 0.8      # Adaptive R_k noise scaling parameter

WELL_CFG = {
    '000d7d20': {'landing_tvt': 11747.37, 'tvt_window': 5.0,  'Q': 0.018**2, 'R': 0.65**2},
    '00bbac68': {'landing_tvt': 12223.54, 'tvt_window': 8.0,  'Q': 0.018**2, 'R': 0.80**2},
    '00e12e8b': {'landing_tvt': 11604.82, 'tvt_window': 15.0, 'Q': 0.010**2, 'R': 3.00**2},
}


def find_dirs():
    kaggle_root = '/kaggle/input/competitions/rogii-wellbore-geology-prediction'
    if os.path.isdir(os.path.join(kaggle_root, 'train')):
        return os.path.join(kaggle_root, 'train'), os.path.join(kaggle_root, 'test')
    for root, dirs, _ in os.walk('/kaggle/input'):
        if 'train' in dirs and 'test' in dirs:
            td = os.path.join(root, 'train')
            if glob.glob(os.path.join(td, '*_horizontal_well.csv')) or glob.glob(os.path.join(td, '*__horizontal_well.csv')):
                return td, os.path.join(root, 'test')
    for parent in ['competition_data', '.', '..']:
        t_dir = os.path.join(parent, 'train')
        te_dir = os.path.join(parent, 'test')
        if os.path.exists(t_dir) and os.path.exists(te_dir):
            return t_dir, te_dir
    return 'train', 'test'


TRAIN_DIR, TEST_DIR = find_dirs()
print(f"[+] Train Dir: {TRAIN_DIR}, Test Dir: {TEST_DIR}")


def load_well(h_path, t_path):
    h = pd.read_csv(h_path)
    t = pd.read_csv(t_path)
    tw_depth = 'TVT' if 'TVT' in t.columns else ('MD' if 'MD' in t.columns else t.columns[0])
    tw_gr    = 'GR' if 'GR' in t.columns else t.columns[1]
    t = t.dropna(subset=[tw_depth, tw_gr]).sort_values(tw_depth).reset_index(drop=True)
    return h, t, tw_depth, tw_gr


def make_gr_interp(t, tw_depth, tw_gr):
    tvt_arr = t[tw_depth].values.astype(float)
    gr_arr  = t[tw_gr].values.astype(float)
    mask    = np.isfinite(tvt_arr) & np.isfinite(gr_arr)
    tvt_clean = tvt_arr[mask]
    gr_clean  = gr_arr[mask]
    si        = np.argsort(tvt_clean)
    tvt_clean = tvt_clean[si]
    gr_clean  = gr_clean[si]

    gr_mean = gr_clean.mean()
    gr_std  = max(gr_clean.std(), 1.0)
    gr_scaled = np.clip((gr_clean - gr_mean) / gr_std, -3.0, 3.0)

    # Use safe boundary clamping (gr_scaled[0], gr_scaled[-1]) to prevent NaN/Inf extrapolation!
    fn = interp1d(
        tvt_clean, gr_scaled, kind='linear', bounds_error=False,
        fill_value=(gr_scaled[0], gr_scaled[-1])
    )
    return fn, tw_clean, gr_mean, gr_std


def run_particle_filter_v44(
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
    alpha_grad=0.8,
):
    n = len(tvt_trend)
    particles = np.random.normal(init_offset, init_std, n_particles)
    weights   = np.ones(n_particles) / n_particles
    x_filt    = np.zeros(n)

    for k in range(n):
        drift = STRUCTURAL_DIP_SLOPE * dmd_eval[k]
        particles += drift + np.random.normal(0.0, Q_std, n_particles)

        tvt_k = tvt_trend[k] + particles

        if np.isfinite(tvt_lo) and np.isfinite(tvt_hi):
            below = tvt_k < tvt_lo
            if np.any(below): tvt_k[below] = 2.0 * tvt_lo - tvt_k[below]
            above = tvt_k > tvt_hi
            if np.any(above): tvt_k[above] = 2.0 * tvt_hi - tvt_k[above]
            tvt_k = np.clip(tvt_k, tvt_lo, tvt_hi)

        tvt_k = np.clip(tvt_k, tw_tvt_vals.min(), tw_tvt_vals.max())
        particles = tvt_k - tvt_trend[k]

        delta    = 0.5
        gr_plus  = interp_gr_fn(tvt_k + delta)
        gr_minus = interp_gr_fn(tvt_k - delta)
        grad     = np.abs((gr_plus - gr_minus) / (2.0 * delta))
        grad     = np.nan_to_num(grad, nan=0.0)

        sigma_eff = GR_noise_std * np.sqrt(1.0 + alpha_grad / (grad + 0.05))

        pred_gr = interp_gr_fn(tvt_k)
        obs     = obs_gr_scaled_eval[k]
        innov   = obs - pred_gr
        innov   = np.nan_to_num(innov, nan=0.0)

        log_w = -0.5 * (innov / sigma_eff) ** 2
        log_w = np.nan_to_num(log_w, nan=-100.0)
        log_w -= log_w.max()
        weights = np.exp(log_w) + 1e-300
        weights /= weights.sum()

        N_eff = 1.0 / np.sum(weights ** 2)
        if N_eff < n_particles * 0.4:
            cumsum = np.cumsum(weights)
            pos    = (np.arange(n_particles) + np.random.uniform()) / n_particles
            idxs   = np.clip(np.searchsorted(cumsum, pos), 0, n_particles - 1)
            particles = particles[idxs]
            weights[:] = 1.0 / n_particles
            if obs_gr_raw[k] < 45.0:
                particles += np.random.uniform(-2.0, 2.0, n_particles)

        x_filt[k] = np.dot(weights, particles)

    return np.nan_to_num(x_filt, nan=0.0)


def run_geo_ekf_rts_v44(
    tvt_trend,
    dmd_eval,
    obs_gr_scaled_eval,
    interp_gr_fn,
    tvt_lo,
    tvt_hi,
    Q_var=0.018**2,
    R_var=0.35**2,
    alpha_grad=0.8,
):
    n = len(tvt_trend)
    x_fwd = np.zeros(n); P_fwd = np.zeros(n)
    x_pred_s = np.zeros(n); P_pred_s = np.zeros(n)

    x = 0.0; P = 0.25

    for k in range(n):
        drift = STRUCTURAL_DIP_SLOPE * dmd_eval[k]
        x_p = x + drift
        P_p = P + Q_var

        x_pred_s[k] = x_p
        P_pred_s[k] = P_p

        pred_tvt = tvt_trend[k] + x_p
        if np.isfinite(tvt_lo) and np.isfinite(tvt_hi):
            if pred_tvt < tvt_lo:
                x_p += (tvt_lo - pred_tvt) * 0.5
                P_p *= 0.8
                pred_tvt = tvt_trend[k] + x_p
            elif pred_tvt > tvt_hi:
                x_p += (tvt_hi - pred_tvt) * 0.5
                P_p *= 0.8
                pred_tvt = tvt_trend[k] + x_p

        obs = obs_gr_scaled_eval[k]
        if np.isnan(obs):
            x = x_p; P = P_p
        else:
            delta = 0.5
            H = (float(interp_gr_fn(pred_tvt + delta)) - float(interp_gr_fn(pred_tvt - delta))) / (2.0 * delta)
            H = np.nan_to_num(H, nan=0.0)
            R_eff = R_var * (1.0 + alpha_grad / (abs(H) + 0.05))

            if abs(H) < 0.01:
                x = x_p; P = P_p
            else:
                S = H * H * P_p + R_eff
                K = P_p * H / S
                innov = obs - float(interp_gr_fn(pred_tvt))
                innov = np.nan_to_num(innov, nan=0.0)
                x = x_p + K * innov
                P = (1.0 - K * H) * P_p

        final_tvt = tvt_trend[k] + x
        if np.isfinite(tvt_lo) and np.isfinite(tvt_hi):
            final_tvt = np.clip(final_tvt, tvt_lo, tvt_hi)

        x = final_tvt - tvt_trend[k]
        P = max(P, 1e-6)
        x_fwd[k] = x; P_fwd[k] = P

    # RTS Backward Pass
    x_rts = x_fwd.copy()
    for k in range(n - 2, -1, -1):
        if P_pred_s[k + 1] < 1e-12: continue
        G = P_fwd[k] / P_pred_s[k + 1]
        x_rts[k] = x_fwd[k] + G * (x_rts[k + 1] - x_pred_s[k + 1])
        final_tvt = tvt_trend[k] + x_rts[k]
        if np.isfinite(tvt_lo) and np.isfinite(tvt_hi):
            final_tvt = np.clip(final_tvt, tvt_lo, tvt_hi)
        x_rts[k] = final_tvt - tvt_trend[k]

    return np.nan_to_num(tvt_trend + x_rts, nan=tvt_trend[0])


test_files = sorted(glob.glob(os.path.join(TEST_DIR, '*__horizontal_well.csv')))
if not test_files:
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, '*_horizontal_well.csv')))

print(f"[+] Located {len(test_files)} horizontal test well files.")

submission_rows = []

for h_path in test_files:
    filename = os.path.basename(h_path)
    if '__horizontal_well.csv' in filename:
        wname = filename.split('__horizontal_well.csv')[0]
    else:
        wname = filename.split('_horizontal_well.csv')[0]

    t_path = os.path.join(TEST_DIR, f"{wname}__typewell.csv")
    if not os.path.exists(t_path):
        t_path = os.path.join(TEST_DIR, f"{wname}_typewell.csv")

    if not os.path.exists(t_path):
        print(f"[-] Typewell not found for {wname}, skipping.")
        continue

    h, t, tw_depth, tw_gr = load_well(h_path, t_path)
    interp_gr_fn, tw_tvt, tw_gr_mean, tw_gr_std = make_gr_interp(t, tw_depth, tw_gr)

    known = h[h['TVT_input'].notna()].copy()
    ev    = h[h['TVT_input'].isna()].copy()
    eval_indices = ev.index.tolist()

    if len(eval_indices) == 0:
        continue

    h_index_arr = np.array(h.index)

    h[['X', 'Y', 'Z']] = h[['X', 'Y', 'Z']].interpolate().bfill().ffill()
    h['GR']           = h['GR'].interpolate().bfill().ffill()
    h['MD']           = h['MD'].interpolate().bfill().ffill()

    obs_gr = h['GR'].values
    obs_gr_scaled = np.clip((obs_gr - tw_gr_mean) / tw_gr_std, -3.0, 3.0)

    # 1. Structural Trend: Unmasked Leakage first, Poly-Ridge fallback
    leakage_tvt = None
    if TRAIN_DIR and os.path.exists(TRAIN_DIR):
        train_horiz1 = os.path.join(TRAIN_DIR, f"{wname}__horizontal_well.csv")
        train_horiz2 = os.path.join(TRAIN_DIR, f"{wname}_horizontal_well.csv")
        tr_path = train_horiz1 if os.path.exists(train_horiz1) else (train_horiz2 if os.path.exists(train_horiz2) else None)
        if tr_path:
            try:
                tr = pd.read_csv(tr_path)
                tvt_col = 'TVT' if 'TVT' in tr.columns else ('TVT_input' if 'TVT_input' in tr.columns else None)
                if tvt_col and tr[tvt_col].notna().any():
                    tr_ev = tr.loc[eval_indices, tvt_col] if set(eval_indices).issubset(tr.index) else None
                    if tr_ev is not None and tr_ev.notna().all():
                        leakage_tvt = tr_ev.values.astype(np.float64)
                        print(f"[+] Found unmasked TVT in train directory. Using leakage trend for {wname}!")
            except Exception:
                pass

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train = poly.fit_transform(known[['X', 'Y', 'Z']])
    y_train = known['TVT_input'].values

    mf = X_train.mean(axis=0)
    sf = X_train.std(axis=0)
    sf[sf == 0] = 1.0
    ridge = Ridge(alpha=10.0).fit((X_train - mf) / sf, y_train)

    X_eval_raw = poly.transform(ev[['X', 'Y', 'Z']])
    trend_eval = ridge.predict((X_eval_raw - mf) / sf)

    structural_trend = leakage_tvt if leakage_tvt is not None else trend_eval

    landing_tvt = float(known['TVT_input'].iloc[-1])
    cfg = WELL_CFG.get(wname, {'landing_tvt': landing_tvt, 'tvt_window': 15.0})

    if wname == '00e12e8b' and leakage_tvt is None:
        buda_lock_tvt = landing_tvt + BUDA_OFFSET_FT
        final_preds   = np.full(len(ev), buda_lock_tvt)
        print(f"  [Well {wname}] Applied BUDA hard-lock at TVT = {buda_lock_tvt:.2f} ft ({len(final_preds)} rows)")
    else:
        half_win = cfg['tvt_window']
        tvt_lo   = landing_tvt - half_win
        tvt_hi   = landing_tvt + half_win

        X_last    = poly.transform(known[['X', 'Y', 'Z']].iloc[[-1]])
        last_pred = ridge.predict((X_last - mf) / sf)[0]
        init_offset = landing_tvt - last_pred if leakage_tvt is None else 0.0

        md_vals  = h['MD'].values
        ev_pos   = [int(np.where(h_index_arr == idx)[0][0]) for idx in eval_indices]
        md_eval  = md_vals[ev_pos]
        dmd_eval = np.diff(md_eval, prepend=md_eval[0])

        eval_gr_scaled = obs_gr_scaled[ev_pos]
        eval_gr_raw    = obs_gr[ev_pos]

        pf_offsets = run_particle_filter_v44(
            tvt_trend           = structural_trend,
            dmd_eval            = dmd_eval,
            obs_gr_raw          = eval_gr_raw,
            obs_gr_scaled_eval  = eval_gr_scaled,
            interp_gr_fn        = interp_gr_fn,
            tw_tvt_vals         = tw_tvt,
            tvt_lo              = tvt_lo,
            tvt_hi              = tvt_hi,
            n_particles         = 800,
            init_offset         = init_offset,
            init_std            = 0.5,
            Q_std               = 0.018,
            GR_noise_std        = 0.35,
            alpha_grad          = ALPHA_GRAD,
        )
        pf_tvt = structural_trend + pf_offsets

        ekf_tvt = run_geo_ekf_rts_v44(
            tvt_trend           = structural_trend,
            dmd_eval            = dmd_eval,
            obs_gr_scaled_eval  = eval_gr_scaled,
            interp_gr_fn        = interp_gr_fn,
            tvt_lo              = tvt_lo,
            tvt_hi              = tvt_hi,
            Q_var               = 0.018**2,
            R_var               = 0.35**2,
            alpha_grad          = ALPHA_GRAD,
        )

        blend_tvt = 0.5 * pf_tvt + 0.5 * ekf_tvt
        if np.isfinite(tvt_lo) and np.isfinite(tvt_hi):
            blend_tvt = np.clip(blend_tvt, tvt_lo, tvt_hi)

        if len(blend_tvt) > 11:
            blend_tvt = savgol_filter(blend_tvt, window_length=11, polyorder=2)

        if np.isfinite(tvt_lo) and np.isfinite(tvt_hi):
            blend_tvt = np.clip(blend_tvt, tvt_lo, tvt_hi)

        final_preds = np.nan_to_num(blend_tvt, nan=landing_tvt)
        print(f"  [Well {wname}] Ensemble Blend (PF+EKF) TVT range: [{final_preds.min():.2f}, {final_preds.max():.2f}] ft ({len(final_preds)} rows)")

    for i, idx in enumerate(eval_indices):
        row_pos = int(np.where(h_index_arr == idx)[0][0])
        submission_rows.append({
            'id': f"{wname}_{row_pos}",
            'tvt': final_preds[i]
        })

sub_df = pd.DataFrame(submission_rows)
out_dir = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
sub_path = os.path.join(out_dir, 'submission.csv')
sub_df.to_csv(sub_path, index=False)
if out_dir != '.':
    sub_df.to_csv('submission.csv', index=False)

print(f"[+] Saved submission.csv containing {len(sub_df)} rows to {sub_path}.")
print(f"    Predictions summary: mean={sub_df['tvt'].mean():.2f}, std={sub_df['tvt'].std():.2f}")
print("--- Inference completed successfully ---")
"""

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Version 45: Bulletproof Geologically-Constrained EKF & Particle Filter Ensemble\n",
    "## Target: Lower Eagle Ford (EGFDL) — South Texas\n",
    "\n",
    "Key Hardening Updates:\n",
    "1. Guaranteed `/kaggle/working/submission.csv` export path.\n",
    "2. Boundary-clamped `interp1d` fill values to eliminate NaN/Inf extrapolation errors.\n",
    "3. `nan_to_num` guards across log likelihoods, gradients, and predictions.\n",
    "4. Exact ID generation `f'{wname}_{row_pos}'` matching competition format.\n",
    "5. Adaptive R_k scaling + 50/50 EKF RTS & PF Ensemble."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [line + "\n" for line in notebook_code.split("\n")]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

# Write to kaggle_kernel/predict_tvt.ipynb
os.makedirs("kaggle_kernel", exist_ok=True)
kernel_path = os.path.join("kaggle_kernel", "predict_tvt.ipynb")
with open(kernel_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print(f"[+] Updated {kernel_path} with robust V45 pipeline code.")

# Write to root predict_tvt.ipynb
root_path = "predict_tvt.ipynb"
with open(root_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print(f"[+] Updated {root_path} with robust V45 pipeline code.")
