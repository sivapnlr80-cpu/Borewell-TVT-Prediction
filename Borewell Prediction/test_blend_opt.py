import os
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression

def evaluate_blend(alpha, weight):
    rmses = []
    for f in sorted(glob.glob('train/*_horizontal_well.csv')):
        df = pd.read_csv(f)
        n = len(df)
        es = int(n * 0.7)
        known = df.iloc[:es]
        eval_df = df.iloc[es:]
        
        # Ridge on all coords
        X_train1 = known[['MD', 'X', 'Y', 'Z']]
        y_train1 = known['TVT_input']
        X_eval1 = eval_df[['MD', 'X', 'Y', 'Z']]
        
        mean = X_train1.mean(axis=0)
        std = X_train1.std(axis=0)
        std[std == 0] = 1
        X_train1_scaled = (X_train1 - mean) / std
        X_eval1_scaled = (X_eval1 - mean) / std
        
        lr1 = Ridge(alpha=alpha).fit(X_train1_scaled, y_train1)
        pred1 = lr1.predict(X_eval1_scaled)
        
        # OLS on MD and Z
        lr2 = LinearRegression().fit(known[['MD', 'Z']], known['TVT_input'])
        pred2 = lr2.predict(eval_df[['MD', 'Z']])
        
        # Blend
        pred_ensemble = weight * pred1 + (1.0 - weight) * pred2
        
        rmse = np.sqrt(np.mean((pred_ensemble - eval_df['TVT_input'])**2))
        rmses.append(rmse)
    return np.mean(rmses)

# Grid search
best_rmse = 1e18
best_params = {}

for alpha in [5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]:
    for w in np.linspace(0.4, 0.6, 21):
        rmse = evaluate_blend(alpha, w)
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = {'alpha': alpha, 'weight': w}

print(f"Best blend: alpha={best_params['alpha']:.1f}, weight={best_params['weight']:.4f} => RMSE = {best_rmse:.5f}")
