import os
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression

def evaluate_triple_blend(w1, w2, w3):
    rmses = []
    for f in sorted(glob.glob('train/*_horizontal_well.csv')):
        df = pd.read_csv(f)
        n = len(df)
        es = int(n * 0.7)
        known = df.iloc[:es]
        eval_df = df.iloc[es:]
        
        # Model 1: Ridge scaled
        X_train1 = known[['MD', 'X', 'Y', 'Z']]
        y_train1 = known['TVT_input']
        X_eval1 = eval_df[['MD', 'X', 'Y', 'Z']]
        mean1 = X_train1.mean(axis=0)
        std1 = X_train1.std(axis=0)
        std1[std1 == 0] = 1
        X_train1_scaled = (X_train1 - mean1) / std1
        X_eval1_scaled = (X_eval1 - mean1) / std1
        model1 = Ridge(alpha=51.0).fit(X_train1_scaled, y_train1)
        pred1 = model1.predict(X_eval1_scaled)
        
        # Model 2: OLS MD, Z
        model2 = LinearRegression().fit(known[['MD', 'Z']], known['TVT_input'])
        pred2 = model2.predict(eval_df[['MD', 'Z']])
        
        # Model 3: Ridge raw
        model3 = Ridge(alpha=100000.0).fit(X_train1, y_train1)
        pred3 = model3.predict(X_eval1)
        
        # Ensemble
        pred_ensemble = w1 * pred1 + w2 * pred2 + w3 * pred3
        
        rmse = np.sqrt(np.mean((pred_ensemble - eval_df['TVT_input'])**2))
        rmses.append(rmse)
    return np.mean(rmses)

# Grid search
best_rmse = 1e18
best_weights = []

for w1 in np.linspace(0.0, 1.0, 21):
    for w2 in np.linspace(0.0, 1.0 - w1, 21):
        w3 = 1.0 - w1 - w2
        if w3 < -1e-5:
            continue
        rmse = evaluate_triple_blend(w1, w2, w3)
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = [w1, w2, w3]

print(f"Best Triple Blend: w_scaled_ridge={best_weights[0]:.4f}, w_ols={best_weights[1]:.4f}, w_raw_ridge={best_weights[2]:.4f} => RMSE = {best_rmse:.5f}")
