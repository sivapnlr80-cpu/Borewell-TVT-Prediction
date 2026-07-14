import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

def evaluate_constant_shift():
    rmses = []
    for f in sorted(glob.glob('train/*_horizontal_well.csv')):
        df = pd.read_csv(f)
        t_df = pd.read_csv(f.replace('horizontal_well', 'typewell'))
        
        n = len(df)
        es = int(n * 0.7)
        known = df.iloc[:es]
        eval_df = df.iloc[es:]
        
        # Fit Ridge regression on known section
        X_cols = ['MD', 'X', 'Y', 'Z']
        X_train = known[X_cols]
        y_train = known['TVT_input']
        X_eval = eval_df[X_cols]
        
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0
        X_train_scaled = (X_train - mean) / std
        X_eval_scaled = (X_eval - mean) / std
        
        model = Ridge(alpha=10.0).fit(X_train_scaled, y_train)
        pred_linear = model.predict(X_eval_scaled)
        
        # Typewell interpolator
        tw_tvt = t_df['MD'].values.astype(np.float64)
        tw_gr = t_df['GR'].values.astype(np.float64)
        si = np.argsort(tw_tvt)
        tw_tvt, tw_gr = tw_tvt[si], tw_gr[si]
        interp_func = interp1d(tw_tvt, tw_gr, kind='linear', bounds_error=False, fill_value='extrapolate')
        
        eval_gr = eval_df['GR'].values
        
        # Grid search for constant shift
        best_shift = 0.0
        best_cost = 1e18
        for shift in np.linspace(-5.0, 5.0, 101):
            pred_shifted = pred_linear + shift
            ref_gr = interp_func(pred_shifted)
            cost = np.mean((eval_gr - ref_gr) ** 2)
            if cost < best_cost:
                best_cost = cost
                best_shift = shift
                
        pred_final = pred_linear + best_shift
        rmse = np.sqrt(np.mean((pred_final - eval_df['TVT_input'])**2))
        rmses.append(rmse)
        print(f"Well: {os.path.basename(f).split('__')[0]} | Best shift: {best_shift:+.2f} | RMSE: {rmse:.4f}")
        
    print(f"\nAverage RMSE with Constant Shift = {np.mean(rmses):.4f}")

evaluate_constant_shift()
