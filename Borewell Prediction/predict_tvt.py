"""
True Vertical Thickness (TVT) Prediction Pipeline for Horizontal Wells
Author: Kaggle Grandmaster & Senior Data Scientist
Specialization: Geophysics and Tabular/Time-Series ML
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import warnings
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="Borewell TVT Prediction Pipeline")
    parser.add_argument("--train_dir", type=str, default="train", help="Directory containing training wells")
    parser.add_argument("--test_dir", type=str, default="test", help="Directory containing test wells")
    parser.add_argument("--output", type=str, default="submission.csv", help="Path to save the final submission file")
    parser.add_argument("--mode", type=str, default="advanced", choices=["baseline", "advanced", "ablation"],
                        help="Pipeline run mode (baseline features, advanced features with alignment, or comparative ablation study)")
    return parser.parse_args()


def resolve_column_name(df, options, required=True):
    """
    Dynamically maps expected database column names to actual columns present,
    handling different conventions (e.g. Depth vs MD, Easting vs X, gr vs GR).
    """
    for opt in options:
        if opt in df.columns:
            return opt
    # Case-insensitive search
    for opt in options:
        for col in df.columns:
            if col.lower() == opt.lower():
                return col
    if required:
        raise ValueError(f"Could not find any of options {options} in columns {list(df.columns)}")
    return None


def reduce_mem_usage(df):
    """
    Optimizes memory by downcasting numeric data types.
    """
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and not pd.api.types.is_categorical_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).startswith('int'):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df


def generate_mock_data():
    """
    Generates geologically plausible mock data to enable dry-running the pipeline offline
    without requiring external files.
    """
    print("\n[!] No local datasets found. Generating synthetic geological data for dry-run...")
    os.makedirs("train", exist_ok=True)
    os.makedirs("test", exist_ok=True)
    
    # Define vertical stratigraphy template (true vertical profiles)
    # The layers have distinct Gamma Ray (GR) profiles
    def get_ref_gr(depths):
        # A combination of sin waves and step functions to mimic stratigraphic layers
        gr = 50 + 30 * np.sin(depths / 100) + 15 * np.sin(depths / 20)
        gr[(depths > 2200) & (depths < 2600)] += 40  # Shale bed (High GR)
        gr[(depths > 3200) & (depths < 3500)] -= 30  # Sandstone (Low GR)
        return np.clip(gr + np.random.normal(0, 3, len(depths)), 10, 150)

    # 1. Generate Typewells (Vertical profiles)
    well_names = [f"WELL_{i}" for i in range(1, 8)]  # 1 to 5 for train, 6 & 7 for test
    for i, name in enumerate(well_names):
        # Reference vertical depth
        t_depths = np.arange(1000, 5000, 10, dtype=np.float32)
        gr_vals = get_ref_gr(t_depths)
        
        # Formation columns
        buda = ((t_depths > 3000) & (t_depths < 3800)).astype(np.int32)
        egfdl = ((t_depths > 2000) & (t_depths < 2800)).astype(np.int32)
        
        typewell_df = pd.DataFrame({
            'MD': t_depths,
            'GR': gr_vals,
            'BUDA': buda,
            'EGFDL': egfdl
        })
        
        folder = "train" if i < 5 else "test"
        typewell_df.to_csv(f"{folder}/{name}__typewell.csv", index=False)
        
        # 2. Generate Horizontal Wells (Sinuous path intersecting the vertical profile)
        h_md = np.arange(4000, 8000, 5, dtype=np.float32)
        N = len(h_md)
        
        # Sinuous path in 3D
        x = np.cumsum(np.random.uniform(3, 5, N)).astype(np.float32)
        y = (100 * np.sin(h_md / 500) + np.random.normal(0, 1, N)).astype(np.float32)
        z = (2400 + 120 * np.cos(h_md / 600) + np.random.normal(0, 2, N)).astype(np.float32)
        h_gr = get_ref_gr(z)
        tvt_true = (30 + 15 * np.sin(h_md / 1000) + (z - 2400) * 0.05 + np.random.normal(0, 0.5, N)).astype(np.float32)
        
        horiz_df = pd.DataFrame({
            'MD': h_md,
            'X': x,
            'Y': y,
            'Z': z,
            'GR': h_gr,
            'TVT_input': tvt_true
        })
        
        # Mask target for test set in the evaluation zone (last 30% of the well length)
        if folder == "test":
            eval_start_idx = int(N * 0.7)
            horiz_df.loc[eval_start_idx:, 'TVT_input'] = np.nan
            
        horiz_df.to_csv(f"{folder}/{name}__horizontal_well.csv", index=False)
    print("[!] Synthetic datasets generated successfully.\n")


def compute_trajectory_geometry(df):
    """
    Computes 3D structural trajectory features.
    """
    x_col = resolve_column_name(df, ['X', 'x', 'Easting', 'E_coord'])
    y_col = resolve_column_name(df, ['Y', 'y', 'Northing', 'N_coord'])
    z_col = resolve_column_name(df, ['Z', 'z', 'TVD', 'tvd', 'Elevation'])
    md_col = resolve_column_name(df, ['MD', 'Depth', 'DEPTH', 'md', 'depth'])

    dx = df[x_col].diff().fillna(0).values
    dy = df[y_col].diff().fillna(0).values
    dz = df[z_col].diff().fillna(0).values
    dmd = df[md_col].diff().fillna(1e-5).values
    
    step_dist = np.sqrt(dx**2 + dy**2 + dz**2)
    cum_dist = np.cumsum(step_dist)
    inclination = np.arccos(np.clip(dz / dmd, -1.0, 1.0)) * 180 / np.pi
    azimuth = (np.arctan2(dy, dx) * 180 / np.pi + 360) % 360
    
    df['dx'] = dx
    df['dy'] = dy
    df['dz'] = dz
    df['step_dist'] = step_dist
    df['cum_dist'] = cum_dist
    df['inclination'] = inclination
    df['azimuth'] = azimuth
    
    for w in [5, 15, 30]:
        df[f'inclination_std_{w}'] = df['inclination'].rolling(window=w, min_periods=1, center=True).std().fillna(0)
        df[f'azimuth_std_{w}'] = df['azimuth'].rolling(window=w, min_periods=1, center=True).std().fillna(0)
        
    return df


def compute_sequential_logs(df):
    """
    Computes rolling window and lagging context features along Measured Depth (MD).
    """
    gr_col = resolve_column_name(df, ['GR', 'gr', 'Gamma', 'GammaRay', 'Gamma_Ray', 'GAM'])
    
    for w in [5, 15, 30, 50]:
        rolling = df[gr_col].rolling(window=w, min_periods=1, center=True)
        df[f'GR_roll_mean_{w}'] = rolling.mean()
        df[f'GR_roll_std_{w}'] = rolling.std().fillna(0)
        df[f'GR_roll_max_{w}'] = rolling.max()
        df[f'GR_roll_min_{w}'] = rolling.min()
        df[f'GR_roll_median_{w}'] = rolling.median()
        
    for shift in [-5, -2, -1, 1, 2, 5]:
        df[f'GR_lag_{shift}'] = df[gr_col].shift(shift).bfill().ffill()
        
    return df


def align_sequences(horiz_df, type_df):
    """
    Dynamic sequence matching between horizontal well and vertical typewell reference.
    """
    h_gr_col = resolve_column_name(horiz_df, ['GR', 'gr', 'Gamma', 'GammaRay', 'Gamma_Ray', 'GAM'])
    t_gr_col = resolve_column_name(type_df, ['GR', 'gr', 'Gamma', 'GammaRay', 'Gamma_Ray', 'GAM'])
    t_md_col = resolve_column_name(type_df, ['MD', 'Depth', 'DEPTH', 'md', 'depth', 'TVT', 'tvt'])

    scales = [5, 15, 30]
    h_feats = [horiz_df[h_gr_col].values]
    t_feats = [type_df[t_gr_col].values]
    
    for w in scales:
        h_feats.append(horiz_df[h_gr_col].rolling(window=w, min_periods=1, center=True).mean().values)
        t_feats.append(type_df[t_gr_col].rolling(window=w, min_periods=1, center=True).mean().values)
        
    H_mat = np.stack(h_feats, axis=1)
    T_mat = np.stack(t_feats, axis=1)
    
    H_mat = np.nan_to_num(H_mat, nan=50.0)
    T_mat = np.nan_to_num(T_mat, nan=50.0)
    
    dists = cdist(H_mat, T_mat, metric='sqeuclidean')
    best_indices = np.argmin(dists, axis=1)
    
    match_df = pd.DataFrame(index=horiz_df.index)
    match_df['typewell_MD'] = type_df[t_md_col].values[best_indices]
    match_df['typewell_GR'] = type_df[t_gr_col].values[best_indices]
    match_df['typewell_match_dist'] = dists[np.arange(len(best_indices)), best_indices]
    
    for col in type_df.columns:
        if col not in [t_md_col, t_gr_col]:
            match_df[f'typewell_{col}'] = type_df[col].values[best_indices]
            
    return match_df


def load_and_preprocess_directory(directory, mode='advanced'):
    """
    Traverses directory, loads horizontal and typewells, handles features and alignments.
    """
    horiz_files = glob.glob(os.path.join(directory, "*_horizontal_well.csv"))
    if not horiz_files:
        return []
        
    well_dfs = []
    
    for h_file in horiz_files:
        filename = os.path.basename(h_file)
        if '__horizontal_well.csv' in filename:
            wellname = filename.split('__horizontal_well.csv')[0]
        else:
            wellname = filename.split('_horizontal_well.csv')[0]
            
        t_pattern1 = os.path.join(directory, f"{wellname}__typewell.csv")
        t_pattern2 = os.path.join(directory, f"{wellname}_typewell.csv")
        t_file = t_pattern1 if os.path.exists(t_pattern1) else (t_pattern2 if os.path.exists(t_pattern2) else None)
        
        if not t_file:
            print(f"[-] Reference typewell not found for horizontal well: {h_file}. Skipping.")
            continue
            
        h_df = pd.read_csv(h_file)
        t_df = pd.read_csv(t_file)
        
        h_df = reduce_mem_usage(h_df)
        t_df = reduce_mem_usage(t_df)
        
        h_df['WELLNAME'] = wellname
        h_df = compute_trajectory_geometry(h_df)
        h_df = compute_sequential_logs(h_df)
        
        if mode in ['advanced', 'ablation']:
            align_df = align_sequences(h_df, t_df)
            h_df = pd.concat([h_df, align_df], axis=1)
            
        well_dfs.append(h_df)
        
    return well_dfs


def label_encode_categoricals(train_df, test_dfs):
    """
    Identifies string/object columns (excluding WELLNAME) and maps them to consistent integer codes
    across train and test sets to prevent model training errors.
    """
    cat_cols = [c for c in train_df.columns if train_df[c].dtype == 'object' and c != 'WELLNAME']
    for col in cat_cols:
        unique_vals = list(train_df[col].dropna().unique())
        for test_df in test_dfs:
            if col in test_df.columns:
                unique_vals.extend(list(test_df[col].dropna().unique()))
        unique_vals = sorted(list(set(unique_vals)))
        
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        train_df[col] = train_df[col].map(mapping).fillna(-1).astype(np.int32)
        for test_df in test_dfs:
            if col in test_df.columns:
                test_df[col] = test_df[col].map(mapping).fillna(-1).astype(np.int32)
                
    return train_df, test_dfs


def calibrate_predictions(df, preds, target_col='TVT_input'):
    """
    Calibrates test evaluation zone predictions using boundary offset correction.
    """
    resolved_target = resolve_column_name(df, [target_col, 'TVT_input', 'tvt', 'TVT'], required=False)
    if resolved_target is None:
        return preds
        
    calibrated = preds.copy()
    is_nan = df[resolved_target].isna().values
    
    if np.any(is_nan):
        non_nan_indices = np.where(~is_nan)[0]
        if len(non_nan_indices) > 0:
            last_idx = non_nan_indices[-1]
            last_true = df[resolved_target].iloc[last_idx]
            last_pred = preds[last_idx]
            offset = last_true - last_pred
            eval_indices = np.where(is_nan)[0]
            calibrated[eval_indices] = preds[eval_indices] + offset
            
    return calibrated


def smooth_predictions(preds, window_len=15):
    """
    Applies Savitzky-Golay filtering to smooth physical thickness values.
    """
    if len(preds) <= window_len:
        window_len = len(preds)
        if window_len % 2 == 0:
            window_len -= 1
    if window_len < 3:
        return preds
    try:
        return savgol_filter(preds, window_len, polyorder=2)
    except Exception:
        return pd.Series(preds).rolling(window=5, min_periods=1, center=True).mean().values


def run_cross_validation(train_data, features, target_col):
    """
    Executes a GroupKFold cross-validation strategy grouped by WELLNAME.
    """
    gkf = GroupKFold(n_splits=5)
    groups = train_data['WELLNAME'].values
    X = train_data[features]
    y = train_data[target_col]
    
    oof_preds = np.zeros(len(train_data))
    models_lgb = []
    models_xgb = []
    
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'verbosity': -1,
        'random_state': 42
    }
    
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0
    }
    
    try:
        import xgboost as xgb_test
        xgb_test.XGBRegressor(tree_method='hist', device='cuda')
        xgb_params['device'] = 'cuda'
        lgb_params['device'] = 'gpu'
        print("[+] GPU Acceleration enabled for models.")
    except Exception:
        print("[+] Training on CPU.")
        
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # LightGBM
        train_dataset = lgb.Dataset(X_train, label=y_train)
        val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)
        
        lgb_model = lgb.train(
            lgb_params,
            train_dataset,
            num_boost_round=1500,
            valid_sets=[val_dataset],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        models_lgb.append(lgb_model)
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(n_estimators=1500, **xgb_params)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        models_xgb.append(xgb_model)
        
        pred_lgb = lgb_model.predict(X_val)
        pred_xgb = xgb_model.predict(X_val)
        oof_preds[val_idx] = 0.5 * pred_lgb + 0.5 * pred_xgb
        
    cv_rmse = np.sqrt(np.mean((oof_preds - y) ** 2))
    return models_lgb, models_xgb, oof_preds, cv_rmse


def find_data_dirs(train_default, test_default):
    """
    Dynamically searches for 'train' and 'test' folders containing horizontal well logs.
    Supports current directory, Kaggle input directory recursively, and parent folders.
    """
    if os.path.exists(train_default) and os.path.exists(test_default):
        if glob.glob(os.path.join(train_default, "*_horizontal_well.csv")):
            return train_default, test_default
            
    kaggle_input = "/kaggle/input"
    if os.path.exists(kaggle_input):
        for root, dirs, files in os.walk(kaggle_input):
            if "train" in dirs and "test" in dirs:
                t_dir = os.path.join(root, "train")
                te_dir = os.path.join(root, "test")
                if glob.glob(os.path.join(t_dir, "*_horizontal_well.csv")):
                    return t_dir, te_dir

    for parent in [".", "..", "../.."]:
        t_dir = os.path.join(parent, "train")
        te_dir = os.path.join(parent, "test")
        if os.path.exists(t_dir) and os.path.exists(te_dir):
            if glob.glob(os.path.join(t_dir, "*_horizontal_well.csv")):
                return t_dir, te_dir
                
    return None, None


def resolve_data_dirs(train_default, test_default):
    t_dir, te_dir = find_data_dirs(train_default, test_default)
    if t_dir is not None and te_dir is not None:
        print(f"[+] Located datasets at: {t_dir} and {te_dir}")
        return t_dir, te_dir
        
    generate_mock_data()
    return "train", "test"


def main():
    args = parse_args()
    
    train_dir, test_dir = resolve_data_dirs(args.train_dir, args.test_dir)
        
    print(f"\n--- Loading and preprocessing datasets ---")
    train_dfs = load_and_preprocess_directory(train_dir, mode=args.mode)
    test_dfs = load_and_preprocess_directory(test_dir, mode=args.mode)
    
    if not train_dfs or not test_dfs:
        raise ValueError("Train or test datasets could not be loaded. Check filenames and paths.")
        
    print(f"[+] Loaded {len(train_dfs)} training wells and {len(test_dfs)} test wells.")
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    train_df, test_dfs = label_encode_categoricals(train_df, test_dfs)
    
    target_col = resolve_column_name(train_df, ['TVT_input', 'tvt', 'TVT'])
    train_data = train_df[train_df[target_col].notna()].copy()
    
    exclude_names = ['wellname', 'x', 'y', 'z', 'md', 'cum_dist', 'typewell_md', 'tvt_input', 'tvt']
    exclude_cols = [c for c in train_data.columns if c.lower() in exclude_names or c == target_col]
    
    common_cols = set(train_data.columns)
    for test_df in test_dfs:
        common_cols = common_cols.intersection(test_df.columns)
    features = [c for c in common_cols if c not in exclude_cols]
    
    print(f"[+] Total features engineered: {len(features)}")
    
    if args.mode == "ablation":
        print("\n--- Running Feature Ablation Study ---")
        base_features = ['MD', 'X', 'Y', 'Z', 'GR', 'dx', 'dy', 'dz', 'step_dist', 'cum_dist', 'inclination', 'azimuth']
        base_features = [resolve_column_name(train_data, [c], required=False) or c for c in base_features]
        base_features = [c for c in base_features if c in train_data.columns]
        base_features += [c for c in train_data.columns if 'std' in c]
        
        _, _, _, cv_base = run_cross_validation(train_data, base_features, target_col)
        print(f"[*] Approach 1 (Baseline Geometry Features) - Validation RMSE: {cv_base:.4f}")
        
        align_features = features.copy()
        _, _, _, cv_align = run_cross_validation(train_data, align_features, target_col)
        print(f"[*] Approach 2 (Typewell Alignment & Lags)  - Validation RMSE: {cv_align:.4f}")
        
        cal_rmses = []
        for well in train_dfs:
            resolved_t = resolve_column_name(well, [target_col, 'TVT_input', 'tvt', 'TVT'])
            w_data = well[well[resolved_t].notna()].copy()
            split_idx = int(len(w_data) * 0.7)
            w_train = w_data.iloc[:split_idx]
            w_val = w_data.iloc[split_idx:]
            if len(w_val) == 0:
                continue
            
            other_wells = pd.concat([df for df in train_dfs if df['WELLNAME'].iloc[0] != well['WELLNAME'].iloc[0]], ignore_index=True)
            other_data = other_wells[other_wells[target_col].notna()]
            
            model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, verbose=-1, random_state=42)
            model.fit(other_data[features], other_data[target_col])
            
            w_preds = model.predict(w_data[features])
            raw_rmse = np.sqrt(np.mean((w_preds[split_idx:] - w_val[target_col])**2))
            
            cal_preds = w_preds.copy()
            last_known_true = w_train[target_col].iloc[-1]
            last_pred = w_preds[split_idx - 1]
            offset = last_known_true - last_pred
            cal_preds[split_idx:] += offset
            
            cal_rmse = np.sqrt(np.mean((cal_preds[split_idx:] - w_val[target_col])**2))
            cal_rmses.append((raw_rmse, cal_rmse))
            
        mean_raw_rmse = np.mean([x[0] for x in cal_rmses])
        mean_cal_rmse = np.mean([x[1] for x in cal_rmses])
        print(f"[*] Approach 3 (Offset Calibration Check)   - Simulated Eval Zone Raw RMSE: {mean_raw_rmse:.4f} | Calibrated: {mean_cal_rmse:.4f}")
        print("-" * 50)
        
    print("\n--- Training Final Ensembles ---")
    models_lgb, models_xgb, oof_preds, cv_rmse = run_cross_validation(train_data, features, target_col)
    print(f"[+] Out-of-fold Validation RMSE: {cv_rmse:.4f}")
    
    importances = np.mean([m.feature_importance(importance_type='gain') for m in models_lgb], axis=0)
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
    print("\nTop 5 Most Influential Features:")
    for rank, (feat, val) in enumerate(feat_imp.head(5).items(), 1):
        print(f"  {rank}. {feat}: {val:.2f}")
        
    print("\n--- Generating Predictions on Test Set ---")
    submission_rows = []
    uncertainty_rows = []
    
    for well_df in test_dfs:
        wellname = well_df['WELLNAME'].iloc[0]
        resolved_t = resolve_column_name(well_df, [target_col, 'TVT_input', 'tvt', 'TVT'])
        eval_mask = well_df[resolved_t].isna()
        eval_indices = np.where(eval_mask)[0]
        
        if len(eval_indices) == 0:
            continue
            
        X_test = well_df[features]
        
        pred_matrix = []
        for m_lgb in models_lgb:
            pred_matrix.append(m_lgb.predict(X_test))
        for m_xgb in models_xgb:
            pred_matrix.append(m_xgb.predict(X_test))
            
        pred_matrix = np.array(pred_matrix)
        mean_preds = np.mean(pred_matrix, axis=0)
        std_preds = np.std(pred_matrix, axis=0)
        
        calibrated_preds = calibrate_predictions(well_df, mean_preds, target_col)
        final_preds = smooth_predictions(calibrated_preds)
        
        for i in eval_indices:
            row_id = f"{wellname}_{i}"
            submission_rows.append({
                'id': row_id,
                'tvt': final_preds[i]
            })
            uncertainty_rows.append({
                'id': row_id,
                'tvt_pred': final_preds[i],
                'uncertainty_std': std_preds[i]
            })
            
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(args.output, index=False)
    print(f"[+] Saved {len(submission_df)} predictions to {args.output}")
    
    uncertainty_df = pd.DataFrame(uncertainty_rows)
    uncertainty_df.to_csv("predictions_uncertainty.csv", index=False)
    print(f"[+] Saved model predictions uncertainty report to predictions_uncertainty.csv")
    print("--- Pipeline completed successfully ---")


if __name__ == "__main__":
    main()
