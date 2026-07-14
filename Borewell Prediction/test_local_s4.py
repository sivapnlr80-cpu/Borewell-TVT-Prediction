import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# =====================================================================
# VECTORIZED NCC HELPER
# =====================================================================
def get_standardized_windows(series: np.ndarray, W: int) -> np.ndarray:
    n = len(series)
    half = W // 2
    padded = np.pad(series, half, mode='edge')
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(padded, W)
    if len(windows) > n:
        windows = windows[:n]
    means = np.mean(windows, axis=1, keepdims=True)
    stds = np.std(windows, axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    standardized = (windows - means) / stds
    return standardized / np.sqrt(W)

def compute_ncc_features(df: pd.DataFrame, tw_df: pd.DataFrame, window_sizes: list) -> pd.DataFrame:
    df = df.copy()
    gr_obs = df["GR"].values
    gr_tw = tw_df["GR"].values
    tw_tvt = tw_df["MD"].values
    gr_obs_s = savgol_filter(gr_obs, 11, 2) if len(gr_obs) > 15 else gr_obs
    
    for w in window_sizes:
        m_obs = get_standardized_windows(gr_obs_s, w)
        m_tw = get_standardized_windows(gr_tw, w)
        corr_matrix = m_obs @ m_tw.T
        best_idx = np.argmax(corr_matrix, axis=1)
        df[f"NCC_{w}_max"] = np.max(corr_matrix, axis=1)
        df[f"NCC_{w}_tvt"] = tw_tvt[best_idx]
    return df

# =====================================================================
# KALMAN FILTER POST-PROCESSOR
# =====================================================================
def apply_kalman_filter(
    df: pd.DataFrame, 
    ml_preds: np.ndarray, 
    dip_x: float, 
    dip_y: float,
    eval_start_idx: int,
    r_variance: float = 1.0, 
    q_noise: float = 0.05
) -> np.ndarray:
    df = df.copy()
    n = len(df)
    filtered = np.copy(ml_preds)
    
    x_vals = df["X"].values
    y_vals = df["Y"].values
    z_vals = df["Z"].values
    
    # Initialize state at the last known TVT point
    x = df["TVT_input"].values[eval_start_idx - 1]
    P = 0.0
    
    for k in range(eval_start_idx, n):
        dX = x_vals[k] - x_vals[k-1]
        dY = y_vals[k] - y_vals[k-1]
        dZ = z_vals[k] - z_vals[k-1]
        
        # Predict step using dipping plane dip gradients
        x_pred = x - dZ + (dip_x * dX + dip_y * dY)
        P_pred = P + q_noise
        
        # Update step using ML prediction
        z_meas = ml_preds[k - eval_start_idx]
        K = P_pred / (P_pred + r_variance)
        x = x_pred + K * (z_meas - x_pred)
        P = (1.0 - K) * P_pred
        
        filtered[k - eval_start_idx] = x
        
    return filtered

# =====================================================================
# LOCAL PIPELINE EXECUTION
# =====================================================================
def run_local_validation():
    print("=" * 70)
    print("      LOCAL PER-WELL S4 PIPELINE VALIDATION")
    print("=" * 70)
    
    # Find train files
    train_files = sorted(glob.glob("train/*_horizontal_well.csv"))
    if not train_files:
        print("[-] Error: No local training files found.")
        return
        
    rmses_raw = []
    rmses_filtered = []
    
    for f in train_files:
        wellname = os.path.basename(f).split("_")[0]
        df = pd.read_csv(f)
        tw = pd.read_csv(f.replace("horizontal_well", "typewell"))
        
        n = len(df)
        eval_start = int(n * 0.7)
        eval_indices = np.arange(eval_start, n)
        
        # Create masks
        df_masked = df.copy()
        df_masked.loc[eval_indices, "TVT_input"] = np.nan
        
        # 1. Geometry Engine
        # Computes spatial differences
        dX = np.diff(df["X"].values, prepend=df["X"].values[0])
        dY = np.diff(df["Y"].values, prepend=df["Y"].values[0])
        dZ = np.diff(df["Z"].values, prepend=df["Z"].values[0])
        dMD = np.diff(df["MD"].values, prepend=df["MD"].values[0])
        dMD[dMD == 0] = 1e-8
        df_masked["dX"] = dX
        df_masked["dY"] = dY
        df_masked["dZ"] = dZ
        df_masked["dMD"] = dMD
        
        # Inclination
        df_masked["Inc"] = np.arccos(np.clip(np.abs(dZ) / (dMD + 1e-8), -1.0, 1.0)) * (180.0 / np.pi)
        df_masked["Azimuth"] = (np.arctan2(dX, dY) * (180.0 / np.pi)) % 360.0
        
        # Tortuosity
        inc_rad = df_masked["Inc"].values * (np.pi / 180.0)
        az_rad = df_masked["Azimuth"].values * (np.pi / 180.0)
        vx = np.sin(inc_rad) * np.cos(az_rad)
        vy = np.sin(inc_rad) * np.sin(az_rad)
        vz = np.cos(inc_rad)
        vx_prev, vy_prev, vz_prev = np.roll(vx, 1), np.roll(vy, 1), np.roll(vz, 1)
        vx_prev[0], vy_prev[0], vz_prev[0] = vx[0], vy[0], vz[0]
        dot = vx*vx_prev + vy*vy_prev + vz*vz_prev
        d_angle = np.arccos(np.clip(dot, -1.0, 1.0)) * (180.0 / np.pi)
        cum_angle = np.cumsum(d_angle)
        dist_from_landing = df_masked["MD"].values - df_masked["MD"].values[0]
        dist_from_landing[dist_from_landing <= 0] = 1e-8
        df_masked["Tortuosity"] = cum_angle / dist_from_landing
        
        # 2. Local Spatial Consensus (Fit plane on known 70% section)
        known_df = df_masked.iloc[:eval_start]
        X_fit = known_df[["X", "Y"]].values
        elev_fit = known_df["TVT_input"].values + known_df["Z"].values
        
        ridge = Ridge(alpha=10.0).fit(X_fit, elev_fit)
        dip_x, dip_y = ridge.coef_[0], ridge.coef_[1]
        
        # Interpolate baseline
        spatial_prior_elev = ridge.predict(df_masked[["X", "Y"]].values)
        df_masked["Spatial_Prior_TVT"] = spatial_prior_elev - df_masked["Z"].values
        df_masked["Elevation_Baseline"] = df_masked["Z"].values + (df_masked["X"].values * dip_x + df_masked["Y"].values * dip_y)
        
        # 3. Waveform NCC Engine
        df_masked = compute_ncc_features(df_masked, tw, window_sizes=[10, 30])
        
        # 4. Local Machine Learning Engine (Ridge Regression on known 70% section)
        features = [
            "GR", "Inc", "Azimuth", "Tortuosity",
            "Spatial_Prior_TVT", "Elevation_Baseline",
            "NCC_10_max", "NCC_10_tvt", "NCC_30_max", "NCC_30_tvt"
        ]
        
        X_train = df_masked.iloc[:eval_start][features]
        y_train = df_masked.iloc[:eval_start]["TVT_input"]
        X_eval = df_masked.iloc[eval_indices][features]
        y_true = df.iloc[eval_indices]["TVT_input"].values
        
        # Standard scaling
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0
        X_train_s = (X_train - mean) / std
        X_eval_s = (X_eval - mean) / std
        
        # Local model fit
        local_model = Ridge(alpha=20.0).fit(X_train_s, y_train)
        ml_preds = local_model.predict(X_eval_s)
        
        # 5. Geological State Space Filter
        filtered_preds = apply_kalman_filter(
            df=df_masked,
            ml_preds=ml_preds,
            dip_x=dip_x,
            dip_y=dip_y,
            eval_start_idx=eval_start,
            r_variance=2.0,
            q_noise=0.03
        )
        
        raw_rmse = np.sqrt(np.mean((ml_preds - y_true)**2))
        filtered_rmse = np.sqrt(np.mean((filtered_preds - y_true)**2))
        
        rmses_raw.append(raw_rmse)
        rmses_filtered.append(filtered_rmse)
        
        print(f"  Well {wellname}: Raw RMSE = {raw_rmse:.4f} | Kalman RMSE = {filtered_rmse:.4f} (Reduction: {((raw_rmse-filtered_rmse)/raw_rmse)*100:.1f}%)")
        
    print("-" * 70)
    print(f"Average Raw RMSE:    {np.mean(rmses_raw):.4f} (MSE: {np.mean(rmses_raw)**2:.4f})")
    print(f"Average Kalman RMSE: {np.mean(rmses_filtered):.4f} (MSE: {np.mean(rmses_filtered)**2:.4f})")
    print("=" * 70)

if __name__ == "__main__":
    from typing import List
    run_local_validation()
