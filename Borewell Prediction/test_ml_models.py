import os
import glob
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

def evaluate_ml_model(model_class, **kwargs):
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
        
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1
        X_train_scaled = (X_train - mean) / std
        X_eval_scaled = (X_eval - mean) / std
        
        model = model_class(**kwargs).fit(X_train_scaled, y_train)
        pred = model.predict(X_eval_scaled)
        
        rmse = np.sqrt(np.mean((pred - eval_df['TVT_input'])**2))
        rmses.append(rmse)
    return np.mean(rmses)

# SVR Linear
for C in [0.1, 1.0, 10.0, 50.0]:
    rmse = evaluate_ml_model(SVR, kernel='linear', C=C)
    print(f"SVR Linear (C={C}) RMSE: {rmse:.5f}")

# Random Forest
for depth in [2, 3, 4]:
    rmse = evaluate_ml_model(RandomForestRegressor, max_depth=depth, n_estimators=100, random_state=42)
    print(f"RandomForest (depth={depth}) RMSE: {rmse:.5f}")

# MLP
for hidden in [(8, 8), (16, 16)]:
    rmse = evaluate_ml_model(MLPRegressor, hidden_layer_sizes=hidden, alpha=0.1, max_iter=2000, random_state=42)
    print(f"MLP {hidden} RMSE: {rmse:.5f}")
