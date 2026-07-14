import os
import glob
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

def evaluate_pls(n_components):
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
        
        # PLS scales internal variables automatically
        model = PLSRegression(n_components=n_components).fit(X_train, y_train)
        pred = model.predict(X_eval).flatten()
        
        rmse = np.sqrt(np.mean((pred - eval_df['TVT_input'])**2))
        rmses.append(rmse)
    return np.mean(rmses)

for comp in [1, 2, 3, 4]:
    rmse = evaluate_pls(comp)
    print(f"PLSRegression (n_components={comp}) RMSE: {rmse:.5f}")
