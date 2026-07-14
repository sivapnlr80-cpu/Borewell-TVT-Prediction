import os
import time
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

# =====================================================================
# SYNTHETIC DATA GENERATOR (SELF-CONTAINED SANDBOX)
# =====================================================================
class GeosteeringDataGenerator:
    """
    Generates high-fidelity synthetic 3D wellbore trajectories, dipped formations,
    and simulated wireline Gamma Ray logs. Matches ROGII competition structure.
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_typewell_gr(self, tvt_min: float, tvt_max: float, step: float = 0.5) -> pd.DataFrame:
        """Generates a vertical reference typewell GR log with distinct geological layer signatures."""
        tvt_axis = np.arange(tvt_min, tvt_max, step)
        n = len(tvt_axis)
        
        # Build synthetic geological stratigraphic column (shale/sandstone alternation)
        gr = np.zeros(n)
        for i, t in enumerate(tvt_axis):
            # Base clay baseline (high GR) vs clean sand baseline (low GR)
            clay_fraction = 0.5 * (np.sin(t * 0.05) + np.cos(t * 0.015)) + 0.5
            base_gr = 30.0 + clay_fraction * 110.0
            # High-frequency thin bed laminations
            noise = self.rng.normal(0, 4.0)
            gr[i] = base_gr + noise
            
        return pd.DataFrame({"MD": tvt_axis, "GR": gr})

    def generate_wellbore_trajectory(
        self, 
        start_x: float, 
        start_y: float, 
        start_z: float, 
        n_rows: int = 1500, 
        step_md: float = 5.0,
        dip_x: float = 0.01,
        dip_y: float = -0.005
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates a 3D horizontal lateral trajectory drilled relative to a dipping formation.
        Returns the horizontal lateral well log and the vertical reference typewell.
        """
        md = np.arange(0, n_rows * step_md, step_md)
        x = start_x + md * 0.98  # primary drilling azimuth along X
        y = start_y + md * 0.15 + np.sin(md * 0.01) * 30.0  # slight trajectory azimuth wiggles
        
        # Z trajectory: starts vertical-ish, lands horizontal, and active geosteering wiggles
        z = np.zeros_like(md)
        for i, m in enumerate(md):
            if m < 800:
                # Build section
                z[i] = start_z + m * 0.95
            else:
                # Lateral section: horizontal with active undulating control paths
                z[i] = start_z + 800 * 0.95 + (m - 800) * 0.005 + np.sin((m - 800) * 0.025) * 6.0
                
        # Formation Top plane dipping: Elev = dip_x * X + dip_y * Y + constant
        const_top = 2200.0
        elev_top = dip_x * x + dip_y * y + const_top
        
        # TVT is the vertical thickness relative to this structural horizon
        tvt_true = elev_top - z
        
        # Generate Typewell
        typewell_df = self.generate_typewell_gr(tvt_min=tvt_true.min() - 50.0, tvt_max=tvt_true.max() + 50.0)
        
        # Interpolate observed GR from the stratigraphic column
        from scipy.interpolate import interp1d
        interp_gr = interp1d(typewell_df["MD"].values, typewell_df["GR"].values, kind="linear", fill_value="extrapolate")
        gr_obs = interp_gr(tvt_true) + self.rng.normal(0, 3.0, size=len(tvt_true))
        
        df = pd.DataFrame({
            "MD": md,
            "X": x,
            "Y": y,
            "Z": z,
            "GR": gr_obs,
            "TVT_input": tvt_true
        })
        
        return df, typewell_df

    def create_mock_competition_data(self, n_train: int = 5, n_test: int = 3) -> Dict[str, Any]:
        """Creates a mock set of training and test wells simulating the Kaggle file layout."""
        wells_data = {}
        
        # Train wells
        for i in range(1, n_train + 1):
            wname = f"WELL_TRAIN_{i}"
            # Randomized locations and dip planes
            df, tw = self.generate_wellbore_trajectory(
                start_x=self.rng.uniform(1000, 5000),
                start_y=self.rng.uniform(1000, 5000),
                start_z=2000.0,
                dip_x=0.012,
                dip_y=-0.004
            )
            wells_data[f"train/{wname}__horizontal_well.csv"] = df
            wells_data[f"train/{wname}__typewell.csv"] = tw
            
        # Test wells
        for i in range(1, n_test + 1):
            wname = f"WELL_TEST_{i}"
            df, tw = self.generate_wellbore_trajectory(
                start_x=self.rng.uniform(6000, 9000),
                start_y=self.rng.uniform(6000, 9000),
                start_z=2000.0,
                dip_x=0.012,
                dip_y=-0.004
            )
            
            # Save the true TVT for verification before masking
            df_true = df.copy()
            # Partially hide TVT_input in test wells (lateral section goes NaN)
            df.loc[df["MD"] >= 1000, "TVT_input"] = np.nan
            wells_data[f"test/{wname}__horizontal_well.csv"] = df
            wells_data[f"test/{wname}__typewell.csv"] = tw
            wells_data[f"test/{wname}__ground_truth.csv"] = df_true
            
        return wells_data

# =====================================================================
# ENGINE A: DRILLING GEOMETRY & TORTUOSITY ENGINE
# =====================================================================
class GeometryEngine:
    """Computes spatial derivatives, inclination, azimuth, and dogleg tortuosity."""
    @staticmethod
    def process(df: pd.DataFrame, epsilon: float = 1e-8) -> pd.DataFrame:
        df = df.copy()
        
        # Computes spatial differences
        dX = np.diff(df["X"].values, prepend=df["X"].values[0])
        dY = np.diff(df["Y"].values, prepend=df["Y"].values[0])
        dZ = np.diff(df["Z"].values, prepend=df["Z"].values[0])
        dMD = np.diff(df["MD"].values, prepend=df["MD"].values[0])
        dMD[dMD == 0] = epsilon
        
        df["dX"] = dX
        df["dY"] = dY
        df["dZ"] = dZ
        df["dMD"] = dMD
        
        # Inclination: Angle from vertical (0 is straight down, 90 is horizontal)
        # Note: Z is vertical depth, we use absolute dZ for the geometric angle
        df["Inc"] = np.arccos(np.clip(np.abs(dZ) / (dMD + epsilon), -1.0, 1.0)) * (180.0 / np.pi)
        
        # Azimuth: Angle in horizontal plane (0 to 360 degrees)
        df["Azimuth"] = np.arctan2(dX, dY) * (180.0 / np.pi)
        df["Azimuth"] = df["Azimuth"] % 360.0
        
        # Dogleg Severity / Angular Change in 3D space
        # v = [sin(Inc)*cos(Az), sin(Inc)*sin(Az), cos(Inc)]
        inc_rad = df["Inc"].values * (np.pi / 180.0)
        az_rad = df["Azimuth"].values * (np.pi / 180.0)
        
        vx = np.sin(inc_rad) * np.cos(az_rad)
        vy = np.sin(inc_rad) * np.sin(az_rad)
        vz = np.cos(inc_rad)
        
        # Dot product with previous step direction
        vx_prev = np.roll(vx, 1)
        vy_prev = np.roll(vy, 1)
        vz_prev = np.roll(vz, 1)
        vx_prev[0], vy_prev[0], vz_prev[0] = vx[0], vy[0], vz[0]
        
        dot = vx*vx_prev + vy*vy_prev + vz*vz_prev
        d_angle = np.arccos(np.clip(dot, -1.0, 1.0)) * (180.0 / np.pi) # in degrees
        
        # Identify landing point: first point where inclination exceeds 80 degrees
        landing_idx = np.where(df["Inc"].values >= 80.0)[0]
        landing_md = df["MD"].values[landing_idx[0]] if len(landing_idx) > 0 else df["MD"].values[0]
        
        # Cumulative angular change divided by distance from landing point
        cum_angle = np.cumsum(d_angle)
        dist_from_landing = df["MD"].values - landing_md
        dist_from_landing[dist_from_landing <= 0] = epsilon
        
        df["Tortuosity"] = cum_angle / dist_from_landing
        df["Tortuosity"] = df["Tortuosity"].fillna(0.0)
        
        return df

# =====================================================================
# ENGINE B: SPATIAL CONSENSUS ENGINE
# =====================================================================
class SpatialConsensusEngine:
    """Builds spatial KD-Tree and interpolates structural dip gradients and baselines."""
    def __init__(self, anchor_wells: Dict[str, pd.DataFrame]):
        self.anchor_wells = anchor_wells
        self.anchor_coords = []
        self.anchor_names = []
        self.anchor_elevs = []  # Elevation ref at landing
        
        for k, v in anchor_wells.items():
            if "horizontal_well.csv" in k:
                # Find landing point (Inc >= 80)
                inc = GeometryEngine.process(v)["Inc"].values
                landing_idx = np.where(inc >= 80.0)[0]
                idx = landing_idx[0] if len(landing_idx) > 0 else 0
                
                # Structural Elevation Ref = TVT + Z
                elev_ref = v["TVT_input"].values[idx] + v["Z"].values[idx]
                
                self.anchor_coords.append([v["X"].values[idx], v["Y"].values[idx]])
                self.anchor_elevs.append(elev_ref)
                self.anchor_names.append(k)
                
        self.coords_arr = np.array(self.anchor_coords)
        self.elevs_arr = np.array(self.anchor_elevs)
        self.kdtree = cKDTree(self.coords_arr)

    def query(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
        """Queries the k-nearest neighbors and estimates regional dip gradients."""
        df = df.copy()
        inc = GeometryEngine.process(df)["Inc"].values
        landing_idx = np.where(inc >= 80.0)[0]
        idx = landing_idx[0] if len(landing_idx) > 0 else 0
        target_coord = np.array([df["X"].values[idx], df["Y"].values[idx]])
        
        # Query 3 neighbors
        dists, indices = self.kdtree.query(target_coord, k=min(3, len(self.coords_arr)))
        
        # Estimate regional dipping plane using Ridge regression (robust to collinearity)
        X_mat = np.column_stack((self.coords_arr[indices], np.ones_like(indices)))
        y_vec = self.elevs_arr[indices]
        
        ridge = Ridge(alpha=1e-3).fit(X_mat[:, :2], y_vec)
        dip_x, dip_y = ridge.coef_[0], ridge.coef_[1]
        
        # Interpolate a spatial structural elevation prior
        spatial_prior_elev = ridge.predict(df[["X", "Y"]].values)
        
        # Baseline regional structural elevation: TVT_prior = Elev_prior - Z
        df["Spatial_Prior_TVT"] = spatial_prior_elev - df["Z"].values
        df["Elevation_Baseline"] = df["Z"].values + (df["X"].values * dip_x + df["Y"].values * dip_y)
        
        return df, dip_x, dip_y

# =====================================================================
# ENGINE C: SEQUENCE ALIGNMENT ENGINE (VECTORIZED ROLLING NCC)
# =====================================================================
class NCCEngine:
    """Computes Normalized Cross Correlation (NCC) over multi-scale sliding windows."""
    @staticmethod
    def get_standardized_windows(series: np.ndarray, W: int) -> np.ndarray:
        n = len(series)
        half = W // 2
        padded = np.pad(series, half, mode='edge')
        
        # Standardize sliding windows using numpy stride tricks (vectorized)
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(padded, W)
        
        # Clean up any potential size mismatch
        if len(windows) > n:
            windows = windows[:n]
        
        means = np.mean(windows, axis=1, keepdims=True)
        stds = np.std(windows, axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        
        # Normalize so dot product is correlation coefficient
        standardized = (windows - means) / stds
        return standardized / np.sqrt(W)

    @classmethod
    def compute_ncc_features(cls, df: pd.DataFrame, tw_df: pd.DataFrame, window_sizes: List[int]) -> pd.DataFrame:
        df = df.copy()
        gr_obs = df["GR"].values
        gr_tw = tw_df["GR"].values
        tw_tvt = tw_df["MD"].values
        
        # Smooth observed GR slightly for cleaner match
        gr_obs_s = savgol_filter(gr_obs, 11, 2) if len(gr_obs) > 15 else gr_obs
        
        for w in window_sizes:
            # Get normalized sliding windows
            m_obs = cls.get_standardized_windows(gr_obs_s, w)
            m_tw = cls.get_standardized_windows(gr_tw, w)
            
            # Fast vectorized correlation matrix multiplication (Shape: N_obs x N_tw)
            corr_matrix = m_obs @ m_tw.T
            
            # Extract correlation metrics
            best_idx = np.argmax(corr_matrix, axis=1)
            df[f"NCC_{w}_max"] = np.max(corr_matrix, axis=1)
            df[f"NCC_{w}_tvt"] = tw_tvt[best_idx]
            
        return df

# =====================================================================
# ENGINE D: TABULAR MACHINE LEARNING ENGINE
# =====================================================================
class MLEngine:
    """Tabular LightGBM model utilizing GroupKFold validation to predict TVT."""
    def __init__(self, features: List[str]):
        self.features = features
        self.model = None
        self.oof_predictions = None
        self.cv_rmse = 0.0
        
    def train_and_validate(self, train_df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """Performs GroupKFold cross-validation strictly grouped by well_id."""
        groups = train_df["well_id"].values
        group_kfold = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        
        oof = np.zeros(len(train_df))
        models = []
        
        X = train_df[self.features]
        y = train_df["TVT_input"]
        
        for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, groups)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            # Define LightGBM regressor
            model = lgb.LGBMRegressor(
                objective="rmse",
                n_estimators=300,
                learning_rate=0.03,
                max_depth=6,
                num_leaves=31,
                random_state=42,
                verbosity=-1
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
            
            oof[val_idx] = model.predict(X_val)
            models.append(model)
            
        # Overall out-of-fold validation RMSE
        self.oof_predictions = oof
        self.cv_rmse = np.sqrt(np.mean((oof - y) ** 2))
        
        # Fit final model on all data
        self.model = lgb.LGBMRegressor(
            objective="rmse",
            n_estimators=100,
            learning_rate=0.03,
            max_depth=6,
            random_state=42,
            verbosity=-1
        )
        self.model.fit(X, y)
        
        return oof, self.cv_rmse

    def predict(self, eval_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(eval_df[self.features])

# =====================================================================
# ENGINE E: GEOLOGICAL STATE SPACE FILTER (POST-PROCESSOR)
# =====================================================================
class KalmanPostProcessor:
    """
    Applies a geological state-space Kalman filter to physically couple
    predicted TVT values to wellbore depth changes and regional dip.
    """
    @staticmethod
    def filter(
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
        
        # Process covariance, measurement variance
        P = 0.0  # Perfect certainty at the boundary landing point
        
        # State value: initialized at the last known TVT point
        x = df["TVT_input"].values[eval_start_idx - 1]
        
        for k in range(eval_start_idx, n):
            dX = x_vals[k] - x_vals[k-1]
            dY = y_vals[k] - y_vals[k-1]
            dZ = z_vals[k] - z_vals[k-1]
            
            # Predict step physically coupled to trajectory & structural dip plane changes
            # For inverted subsea Z (negative), dZ is negative, so -dZ is positive vertical movement
            x_pred = x - dZ + (dip_x * dX + dip_y * dY)
            P_pred = P + q_noise
            
            # Update step using ML prediction as measurement
            z_meas = ml_preds[k - eval_start_idx]
            
            # Kalman gain
            K = P_pred / (P_pred + r_variance)
            x = x_pred + K * (z_meas - x_pred)
            P = (1.0 - K) * P_pred
            
            filtered[k - eval_start_idx] = x
            
        return filtered

# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================
def run_s4_pipeline():
    print("=" * 70)
    print("      ADVANCED S4 GEOSteering PIPELINE FOR WELLBORE GEOLOGY")
    print("=" * 70)
    
    # 1. Generate high-fidelity synthetic database
    print("[*] Generating synthetic database...")
    gen = GeosteeringDataGenerator(seed=42)
    mock_files = gen.create_mock_competition_data(n_train=6, n_test=3)
    
    # Separate datasets
    train_horiz_files = sorted([k for k in mock_files.keys() if "train/" in k and "horizontal" in k])
    test_horiz_files = sorted([k for k in mock_files.keys() if "test/" in k and "horizontal" in k])
    
    # Build spatial index database for anchor wells
    anchor_wells = {k: mock_files[k] for k in train_horiz_files}
    spatial_engine = SpatialConsensusEngine(anchor_wells)
    
    # 2. Build feature datasets for training wells
    print("[*] Feature extraction on training wells...")
    train_processed_list = []
    
    for i, h_file in enumerate(train_horiz_files):
        df = mock_files[h_file]
        tw = mock_files[h_file.replace("horizontal_well", "typewell")]
        wname = os.path.basename(h_file).split("__")[0]
        
        # Engine A: Geometry
        df = GeometryEngine.process(df)
        
        # Engine B: Spatial
        df, dip_x, dip_y = spatial_engine.query(df)
        
        # Engine C: Waveform NCC
        df = NCCEngine.compute_ncc_features(df, tw, window_sizes=[10, 30])
        
        df["well_id"] = i
        train_processed_list.append(df)
        
    train_all = pd.concat(train_processed_list, ignore_index=True)
    
    # 3. Train Tabular Machine Learning Engine
    features = [
        "GR", "Inc", "Azimuth", "Tortuosity",
        "Spatial_Prior_TVT", "Elevation_Baseline",
        "NCC_10_max", "NCC_10_tvt", "NCC_30_max", "NCC_30_tvt"
    ]
    
    print(f"[*] Training Tabular ML (LightGBM) on {len(features)} features...")
    ml_engine = MLEngine(features)
    oof_preds, cv_rmse = ml_engine.train_and_validate(train_all)
    print(f"  [+] Out-of-Fold GroupKFold Cross-Validation RMSE = {cv_rmse:.4f}")
    print(f"  [+] Out-of-Fold GroupKFold Cross-Validation MSE  = {cv_rmse**2:.4f}")
    
    # 4. Inference on Test Wells & Geological State-Space Filtering
    print("[*] Performing inference and Kalman filtering on test wells...")
    
    for h_file in test_horiz_files:
        df = mock_files[h_file].copy()
        tw = mock_files[h_file.replace("horizontal_well", "typewell")]
        wname = os.path.basename(h_file).split("__")[0]
        
        eval_mask = df["TVT_input"].isna()
        eval_indices = np.where(eval_mask.values)[0]
        eval_start = eval_indices[0]
        
        # Target values for comparison
        df_target = df.copy()
        # Retrieve actual ground truth before masking for comparison
        df_gt = mock_files[h_file.replace("horizontal_well.csv", "ground_truth.csv")]
        y_true = df_gt["TVT_input"].values[eval_indices]
        
        # Run Engines
        df = GeometryEngine.process(df)
        df, dip_x, dip_y = spatial_engine.query(df)
        df = NCCEngine.compute_ncc_features(df, tw, window_sizes=[10, 30])
        
        # Tabular prediction
        eval_df = df.iloc[eval_indices]
        ml_preds = ml_engine.predict(eval_df)
        raw_rmse = np.sqrt(np.mean((ml_preds - y_true)**2))
        
        # Engine E: Geological Kalman Filter Post-Processor
        # Set R measurement variance equal to the cross-validation error variance
        r_var = cv_rmse ** 2
        filtered_preds = KalmanPostProcessor.filter(
            df=df,
            ml_preds=ml_preds,
            dip_x=dip_x,
            dip_y=dip_y,
            eval_start_idx=eval_start,
            r_variance=r_var,
            q_noise=0.03
        )
        filtered_rmse = np.sqrt(np.mean((filtered_preds - y_true)**2))
        
        print(f"\n  === Test Well {wname} Results ===")
        print(f"    Raw LightGBM Prediction RMSE: {raw_rmse:.4f} (MSE: {raw_rmse**2:.4f})")
        print(f"    Kalman Filtered TVT RMSE:   {filtered_rmse:.4f} (MSE: {filtered_rmse**2:.4f})")
        print(f"    Error Reduction:            {((raw_rmse - filtered_rmse)/raw_rmse)*100:.2f}%")
        
    print("\n" + "=" * 70)
    print("                      PIPELINE RUN COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    run_s4_pipeline()
