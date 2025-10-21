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
print("=== XGBoost Training Script Started ===")

# ---------- Load Data ----------
print("Loading data...")
df = pd.read_csv("../train_final.csv")
df = df.dropna(subset=["price"])
y = np.log1p(df["price"])
X = df.drop(columns=["price", "sample_id"])
print(f"Loaded {len(df):,} samples and {X.shape[1]} features")

# ---------- Split ----------
print("Splitting into train/validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X.values.astype(np.float32), y.values.astype(np.float32),
                                                  test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}")

# ---------- Optuna Objective ----------
def objective(trial):
    params = {
        "tree_method": "gpu_hist",
        "gpu_id": 0,
        "predictor": "gpu_predictor",
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "booster": "gbtree",
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.05, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    }

    # For CPU training, uncomment the following lines:
    # params["tree_method"] = "hist"
    # params.pop("gpu_id", None)
    # params.pop("predictor", None)


    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

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
    "objective": "reg:squarederror",
    "eval_metric": "mae"
})

dtrain_full = xgb.DMatrix(X.values.astype(np.float32), label=y.values.astype(np.float32))
final_model = xgb.train(
    best_params,
    dtrain_full,
    num_boost_round=2000,
    verbose_eval=100
)

# ---------- Save Model ----------
model_path = "best_xgb_model.json"
final_model.save_model(model_path)
print(f"\nModel saved as {model_path}")

elapsed = time.time() - start_time
print(f"=== Training Complete | Total Time: {elapsed/60:.2f} minutes ===")
