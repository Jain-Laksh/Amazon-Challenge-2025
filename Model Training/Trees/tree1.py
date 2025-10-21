import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings("ignore")

# ---------- SMAPE ----------
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)).clip(min=1e-6)
    return 200 * np.mean(np.abs(y_true - y_pred) / denom)

# ---------- Start ----------
start_time = time.time()
print("=== LightGBM Training Script Started ===")

# ---------- Load Data ----------
print("Loading data...")
df = pd.read_csv("../train_final.csv")
print(f"Loaded {len(df):,} samples and {df.shape[1]} columns")

# ---------- Prepare Features ----------
df = df.dropna(subset=["price"])
y = np.log1p(df["price"])
X = df.drop(columns=["price", "sample_id"])
print(f"Final feature matrix: {X.shape[0]} rows × {X.shape[1]} features")

# ---------- Split ----------
print("Splitting into train/validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}")

# ---------- Optuna Objective ----------
def objective(trial):
    params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,  # silent during tuning
        "boosting_type": "gbdt",
        "device_type": "gpu",
        "num_leaves": trial.suggest_int("num_leaves", 16, 512),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.05, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_val],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),   # early stopping
            lgb.log_evaluation(period=20)             # prints evaluation every 20 rounds
        ]
    )


    preds = model.predict(X_val)
    val_smape = smape(np.expm1(y_val), np.expm1(preds))
    print(f"Trial {trial.number}: SMAPE = {val_smape:.4f}")
    return val_smape

# ---------- Run Optuna ----------
print("Starting Optuna hyperparameter search...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30, show_progress_bar=True)
print("Optuna tuning complete.")
print(f"Best SMAPE: {study.best_value:.4f}")
print("Best hyperparameters:", study.best_params)

# ---------- Train Final Model ----------
print("\nTraining final LightGBM model with best hyperparameters...")
best_params = study.best_params
best_params.update({
    "objective": "regression",
    "metric": "mae",
    "verbosity": 1,  # show training progress now
    "boosting_type": "gbdt",
    "device_type": "gpu"
})

final_train = lgb.Dataset(X, y)
final_model = lgb.train(best_params, final_train, num_boost_round=2000, callbacks=[lgb.log_evaluation(100)])

# ---------- Save Model ----------
model_path = "best_lgbm_model.txt"
final_model.save_model(model_path)
print(f"\nModel saved as {model_path}")

elapsed = time.time() - start_time
print(f"=== Training Complete | Total Time: {elapsed/60:.2f} minutes ===")
