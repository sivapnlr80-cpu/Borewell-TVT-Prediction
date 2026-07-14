import os
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

def evaluate_poly_ridge():
    rmses = []
    files = sorted(glob.glob('train/*_horizontal_well.csv'))
    for f in files:
        wellname = os.path.basename(f).split('__')[0]
        df = pd.read_csv(f)
        n = len(df)
        es = int(n * 0.7)
        
        known = df.iloc[:es]
        eval_df = df.iloc[es:]
        
        # We use TVT_input for training (same as TVT in known section)
        # and compare against true TVT in evaluation zone
        X_train = known[['MD', 'X', 'Y', 'Z']]
        y_train = known['TVT_input']
        
        X_eval = eval_df[['MD', 'X', 'Y', 'Z']]
        # In train set, the full target column is TVT_input (since it has no NaNs locally)
        # Let's verify if there is a 'TVT' column, otherwise use TVT_input
        target_col = 'TVT' if 'TVT' in df.columns else 'TVT_input'
        y_eval_true = eval_df[target_col]
        
        # Fit polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_eval_poly = poly.transform(X_eval)
        
        # Scale features
        mean = X_train_poly.mean(axis=0)
        std = X_train_poly.std(axis=0)
        std[std == 0] = 1.0
        
        X_train_scaled = (X_train_poly - mean) / std
        X_eval_scaled = (X_eval_poly - mean) / std
        
        # Fit Ridge Regression
        model = Ridge(alpha=8.0)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        preds = model.predict(X_eval_scaled)
        
        rmse = np.sqrt(np.mean((preds - y_eval_true)**2))
        rmses.append(rmse)
        print(f"Well {wellname}: Poly-Ridge RMSE = {rmse:.4f}")
        
    print(f"\nAverage RMSE across all wells = {np.mean(rmses):.4f}")

evaluate_poly_ridge()
