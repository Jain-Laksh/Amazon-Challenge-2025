import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import numpy as np
import xgboost as xgb
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
print("=== XGBoost Huber Training Script (float32) Started ===")

# ---------- Load Data ----------
print("Loading data...")
df = pd.read_csv("../train_final.csv")
df = df.dropna(subset=["price"])
y = np.log1p(df["price"])
X = df.drop(columns=["price", "sample_id"])
print(f"Loaded {len(df):,} samples and {X.shape[1]} features")

# ---------- Clip Features ----------
clip_value = 1e5
print(f"Clipping features to range [-{clip_value}, {clip_value}]")
X_clipped = X.clip(-clip_value, clip_value).astype(np.float32)

# ---------- Split ----------
print("Splitting into train/validation sets...")
X_train, X_val, y_train, y_val = train_test_split(
    X_clipped,
    y.values.astype(np.float32),
    test_size=0.2,
    random_state=42
)
print(f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}")

# ---------- Optuna Objective ----------
def objective(trial):
    params = {
        "tree_method": "gpu_hist",
        "gpu_id": 0,
        "predictor": "gpu_predictor",
        "booster": "gbtree",
        "objective": "reg:pseudohubererror",  # Huber loss
        "eval_metric": "mae",
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 20),
        "lambda": trial.suggest_float("lambda", 0.01, 5.0, log=True),
        "alpha": trial.suggest_float("alpha", 0.01, 5.0, log=True)
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
    dval = xgb.DMatrix(X_val, label=y_val, missing=np.nan)

    evals = [(dval, "validation")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=evals,
        early_stopping_rounds=100,
        verbose_eval=False
    )

    preds = model.predict(dval)
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
print("\nTraining final XGBoost model with best hyperparameters...")
best_params = study.best_params
best_params.update({
    "tree_method": "gpu_hist",
    "gpu_id": 0,
    "predictor": "gpu_predictor",
    "objective": "reg:pseudohubererror",
    "eval_metric": "mae"
})

dtrain_full = xgb.DMatrix(X_clipped, label=y.values.astype(np.float32), missing=np.nan)
final_model = xgb.train(
    best_params,
    dtrain_full,
    num_boost_round=2000,
    verbose_eval=100
)

# ---------- Save Model ----------
model_path = "best_xgb_huber_float32_model.json"
final_model.save_model(model_path)
print(f"\nModel saved as {model_path}")

elapsed = time.time() - start_time
print(f"=== Training Complete | Total Time: {elapsed/60:.2f} minutes ===")
