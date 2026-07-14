import os
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures

def evaluate_model(model_class, **kwargs):
    rmses = []
    for f in sorted(glob.glob('train/*_horizontal_well.csv')):
        df = pd.read_csv(f)
        n = len(df)
        es = int(n * 0.7)
        known = df.iloc[:es]
        eval_df = df.iloc[es:]
        
        X_train = known[['MD', 'X', 'Y', 'Z']]
        y_train = known['TVT_input']
        X_eval = eval_df[['MD', 'X', 'Y', 'Z']]
        
        # Scale features
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1
        X_train_scaled = (X_train - mean) / std
        X_eval_scaled = (X_eval - mean) / std
        
        model = model_class(**kwargs).fit(X_train_scaled, y_train)
        pred = model.predict(X_eval_scaled)
        
        rmse = np.sqrt(np.mean((pred - eval_df['TVT_input'])**2))
        rmses.append(rmse)
    return np.mean(rmses), rmses

# Sweep HuberRegressor, Lasso, ElasticNet, Ridge
for alpha in [0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0]:
    avg, details = evaluate_model(Ridge, alpha=alpha)
    print(f"Ridge (alpha={alpha}): avg RMSE = {avg:.4f}")

for alpha in [0.001, 0.01, 0.1, 1.0]:
    avg, details = evaluate_model(Lasso, alpha=alpha, max_iter=10000)
    print(f"Lasso (alpha={alpha}): avg RMSE = {avg:.4f}")

for epsilon in [1.1, 1.35, 1.5, 2.0]:
    avg, details = evaluate_model(HuberRegressor, epsilon=epsilon, max_iter=10000)
    print(f"HuberRegressor (epsilon={epsilon}): avg RMSE = {avg:.4f}")
