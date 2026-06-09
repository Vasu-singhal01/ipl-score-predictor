import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# 8 consistent IPL teams (as per PPT)
TEAMS = [
    "Chennai Super Kings",
    "Delhi Daredevils",
    "Kings XI Punjab",
    "Kolkata Knight Riders",
    "Mumbai Indians",
    "Rajasthan Royals",
    "Royal Challengers Bangalore",
    "Sunrisers Hyderabad"
]

def preprocess_data(df):
    """
    Preprocessing as per PPT methodology:
    1. Remove unnecessary columns
    2. Filter consistent teams
    3. Filter overs >= 5
    4. One-hot encoding
    5. Train/test split by year
    """
    print("=" * 55)
    print("DATA PREPROCESSING")
    print("=" * 55)

    print(f"\nOriginal shape: {df.shape}")

    # Step 1: Filter consistent teams (as per PPT)
    df = df[df['bat_team'].isin(TEAMS)]
    df = df[df['bowl_team'].isin(TEAMS)]
    print(f"After team filter: {df.shape}")

    # Step 2: Filter overs >= 5 (as per PPT)
    df = df[df['overs'] >= 5]
    print(f"After over filter (>=5): {df.shape}")

    # Step 3: Remove unnecessary columns (as per PPT)
    drop_cols = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non_striker']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    print(f"After dropping unnecessary cols: {df.shape}")

    # Step 4: Extract year for train/test split
    df['year'] = pd.to_datetime(df['date']).dt.year
    df = df.drop(columns=['date'])

    # Step 5: One-hot encoding (as per PPT)
    df = pd.get_dummies(df, columns=['bat_team', 'bowl_team'], drop_first=False)
    print(f"After one-hot encoding: {df.shape}")

    return df


def train_and_evaluate():
    print("\n" + "=" * 55)
    print("IPL SCORE PREDICTOR - MODEL TRAINING")
    print("=" * 55)

    # Load data
    df = pd.read_csv("data/ipl_data.csv")
    print(f"\nDataset loaded: {df.shape}")

    # Preprocess
    df = preprocess_data(df)

    # Train/Test split by year (as per PPT - train up to 2016, test 2017+)
    train = df[df['year'] <= 2016]
    test  = df[df['year'] >  2016]
    print(f"\nTrain set: {train.shape} (up to 2016)")
    print(f"Test set:  {test.shape} (2017+)")

    feature_cols = [c for c in df.columns if c not in ['total', 'year']]
    X_train = train[feature_cols]
    y_train = train['total']
    X_test  = test[feature_cols]
    y_test  = test['total']

    results = {}

    print("\n" + "=" * 55)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 55)

    # ── 1. Linear Regression ──────────────────────────────
    print("\n[1] Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    results['Linear Regression'] = {
        'model': lr,
        'mae':   round(mean_absolute_error(y_test, lr_pred), 2),
        'mse':   round(mean_squared_error(y_test, lr_pred), 2),
        'rmse':  round(np.sqrt(mean_squared_error(y_test, lr_pred)), 2),
        'r2':    round(r2_score(y_test, lr_pred), 4)
    }
    print(f"   MAE : {results['Linear Regression']['mae']}")
    print(f"   RMSE: {results['Linear Regression']['rmse']}")
    print(f"   R²  : {results['Linear Regression']['r2']}")

    # ── 2. Lasso Regression with GridSearchCV ─────────────
    print("\n[2] Lasso Regression (with GridSearchCV)...")
    lasso_params = {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]}
    lasso_cv = GridSearchCV(Lasso(max_iter=10000), lasso_params,
                            cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    lasso_cv.fit(X_train, y_train)
    best_lasso = lasso_cv.best_estimator_
    lasso_pred = best_lasso.predict(X_test)
    results['Lasso Regression'] = {
        'model':       best_lasso,
        'best_alpha':  lasso_cv.best_params_['alpha'],
        'mae':         round(mean_absolute_error(y_test, lasso_pred), 2),
        'mse':         round(mean_squared_error(y_test, lasso_pred), 2),
        'rmse':        round(np.sqrt(mean_squared_error(y_test, lasso_pred)), 2),
        'r2':          round(r2_score(y_test, lasso_pred), 4)
    }
    print(f"   Best alpha: {results['Lasso Regression']['best_alpha']}")
    print(f"   MAE : {results['Lasso Regression']['mae']}")
    print(f"   RMSE: {results['Lasso Regression']['rmse']}")
    print(f"   R²  : {results['Lasso Regression']['r2']}")

    # ── 3. Ridge Regression with GridSearchCV ─────────────
    print("\n[3] Ridge Regression (with GridSearchCV)...")
    ridge_params = {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]}
    ridge_cv = GridSearchCV(Ridge(), ridge_params,
                            cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    ridge_cv.fit(X_train, y_train)
    best_ridge = ridge_cv.best_estimator_
    ridge_pred = best_ridge.predict(X_test)
    results['Ridge Regression'] = {
        'model':       best_ridge,
        'best_alpha':  ridge_cv.best_params_['alpha'],
        'mae':         round(mean_absolute_error(y_test, ridge_pred), 2),
        'mse':         round(mean_squared_error(y_test, ridge_pred), 2),
        'rmse':        round(np.sqrt(mean_squared_error(y_test, ridge_pred)), 2),
        'r2':          round(r2_score(y_test, ridge_pred), 4)
    }
    print(f"   Best alpha: {results['Ridge Regression']['best_alpha']}")
    print(f"   MAE : {results['Ridge Regression']['mae']}")
    print(f"   RMSE: {results['Ridge Regression']['rmse']}")
    print(f"   R²  : {results['Ridge Regression']['r2']}")

    # ── Select best model ─────────────────────────────────
    best_name  = min(results, key=lambda k: results[k]['mae'])
    best_model = results[best_name]['model']

    print("\n" + "=" * 55)
    print(f"BEST MODEL: {best_name}")
    print(f"MAE  : {results[best_name]['mae']} runs")
    print(f"RMSE : {results[best_name]['rmse']} runs")
    print(f"R²   : {results[best_name]['r2']}")
    print("=" * 55)

    # Save model and feature columns
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("models/feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    with open("models/all_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("\n✅ model.pkl saved!")
    print("✅ feature_cols.pkl saved!")
    print("✅ all_results.pkl saved!")

    return results, best_name, feature_cols


if __name__ == "__main__":
    results, best_name, feature_cols = train_and_evaluate()
