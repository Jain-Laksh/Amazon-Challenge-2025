import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import xgboost as xgb

# ======================= Parameters =======================
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- MLP Models (Stella + SigLIP) ---
mlp_model_paths = ["../MLP/optuna_best_48_46_siglip_stella.pt", "../MLP/optuna_best_48_22_siglip_stella.pt"]
mlp_params_list = [
    {"hidden_manual": 1024, "hidden_stella": 320, "hidden_siglip": 320, "final_hidden": 256, "dropout": 0.296},
    {"hidden_manual": 896, "hidden_stella": 384, "hidden_siglip": 320, "final_hidden": 448, "dropout": 0.296}
]

# --- Qwen MLP Model ---
qwen_mlp_model_path = "../MLP/optuna_best_siglip_qwen.pt"
qwen_mlp_params = {
    "hidden_manual": 256,
    "hidden_qwen": 128,
    "hidden_siglip": 192,
    "final_hidden": 384,
    "dropout": 0.264
}

# --- XGBoost Model ---
xgb_model_path = "best_xgb_model_regression.json"

# --- Equal Ensemble Weights ---
num_models = len(mlp_model_paths) + 1 + 1  # +1 for Qwen MLP, +1 for XGBoost
ensemble_weights = torch.tensor([1/num_models] * num_models, device=device)

print(f"Using {num_models} models with equal weights: {ensemble_weights.cpu().numpy()}")

# ==================== Load Test Data ====================
df_test = pd.read_csv("../test_final.csv")
print(f"Test data loaded with shape: {df_test.shape}")

# Load Qwen embeddings for test data
qwen_test_df = pd.read_csv("../qwen_embeddings_test.csv")
print(f"Qwen test embeddings loaded with shape: {qwen_test_df.shape}")

# Rename embedding columns
embedding_cols_qwen = [c for c in qwen_test_df.columns if c != "sample_id"]
qwen_test_df.rename(columns={c: f"qwen_{c}" for c in embedding_cols_qwen}, inplace=True)

# Merge test data with Qwen embeddings
df_test = df_test.merge(qwen_test_df, on="sample_id", how="inner")
print(f"Merged test dataset shape: {df_test.shape}")

sample_ids = df_test["sample_id"].values

# Define column groups
manual_cols = [c for c in df_test.columns if not (c.startswith("stella_") or c.startswith("siglip_") 
                                                   or c.startswith("tfidf_") or c.startswith("qwen_") 
                                                   or c == "sample_id")]
tfidf_cols = [c for c in df_test.columns if c.startswith("tfidf_")]
manual_tfidf_cols = manual_cols + tfidf_cols
stella_cols = [c for c in df_test.columns if c.startswith("stella_")]
siglip_cols = [c for c in df_test.columns if c.startswith("siglip_")]
qwen_cols = [c for c in df_test.columns if c.startswith("qwen_")]

# Load raw, unscaled data
X_manual_raw = df_test[manual_tfidf_cols].values
X_stella_raw = df_test[stella_cols].values
X_siglip_raw = df_test[siglip_cols].values
X_qwen_raw = df_test[qwen_cols].values

# Check for NaN values and handle them
print(f"NaN values found:")
print(f"  Manual: {np.isnan(X_manual_raw).sum()}")
print(f"  Stella: {np.isnan(X_stella_raw).sum()}")
print(f"  SigLIP: {np.isnan(X_siglip_raw).sum()}")
print(f"  Qwen: {np.isnan(X_qwen_raw).sum()}")

# Replace NaN values with 0
X_manual_raw = np.nan_to_num(X_manual_raw, nan=0.0)
X_stella_raw = np.nan_to_num(X_stella_raw, nan=0.0)
X_siglip_raw = np.nan_to_num(X_siglip_raw, nan=0.0)
X_qwen_raw = np.nan_to_num(X_qwen_raw, nan=0.0)

# Clip extreme values
clip_val = 1e5
X_manual_raw = np.clip(X_manual_raw, -clip_val, clip_val)
X_stella_raw = np.clip(X_stella_raw, -clip_val, clip_val)
X_siglip_raw = np.clip(X_siglip_raw, -clip_val, clip_val)
X_qwen_raw = np.clip(X_qwen_raw, -clip_val, clip_val)

# ==================== Preprocessing =====================
# Fitting scalers directly on the test data
scaler_manual = StandardScaler().fit(X_manual_raw)
scaler_stella = StandardScaler().fit(X_stella_raw)
scaler_siglip = StandardScaler().fit(X_siglip_raw)
scaler_qwen = StandardScaler().fit(X_qwen_raw)

# Create scaled data for MLPs
X_manual_scaled = scaler_manual.transform(X_manual_raw)
X_stella_scaled = scaler_stella.transform(X_stella_raw)
X_siglip_scaled = scaler_siglip.transform(X_siglip_raw)
X_qwen_scaled = scaler_qwen.transform(X_qwen_raw)

# ====================== Datasets for MLPs ======================
class TestDataset(Dataset):
    def __init__(self, X_m, X_s, X_i):
        self.X_m = torch.tensor(X_m, dtype=torch.float32)
        self.X_s = torch.tensor(X_s, dtype=torch.float32)
        self.X_i = torch.tensor(X_i, dtype=torch.float32)
    def __len__(self):
        return len(self.X_m)
    def __getitem__(self, idx):
        return self.X_m[idx], self.X_s[idx], self.X_i[idx]

class TestDatasetQwen(Dataset):
    def __init__(self, X_m, X_q, X_i):
        self.X_m = torch.tensor(X_m, dtype=torch.float32)
        self.X_q = torch.tensor(X_q, dtype=torch.float32)
        self.X_i = torch.tensor(X_i, dtype=torch.float32)
    def __len__(self):
        return len(self.X_m)
    def __getitem__(self, idx):
        return self.X_m[idx], self.X_q[idx], self.X_i[idx]

# Create datasets
test_ds = TestDataset(X_manual_scaled, X_stella_scaled, X_siglip_scaled)
test_ds_qwen = TestDatasetQwen(X_manual_scaled, X_qwen_scaled, X_siglip_scaled)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
test_loader_qwen = DataLoader(test_ds_qwen, batch_size=512, shuffle=False)

# ===================== Model Definitions =====================
class DeepWideEmbeddingMLP(nn.Module):
    def __init__(self, manual_dim, stella_dim, siglip_dim, params):
        super().__init__()
        self.manual_net = nn.Sequential(
            nn.Linear(manual_dim, int(params["hidden_manual"])),
            nn.ReLU(),
            nn.BatchNorm1d(int(params["hidden_manual"])),
            nn.Dropout(params["dropout"])
        )
        self.stella_net = nn.Sequential(
            nn.Linear(stella_dim, int(params["hidden_stella"])),
            nn.ReLU(),
            nn.BatchNorm1d(int(params["hidden_stella"])),
            nn.Dropout(params["dropout"])
        )
        self.siglip_net = nn.Sequential(
            nn.Linear(siglip_dim, int(params["hidden_siglip"])),
            nn.ReLU(),
            nn.BatchNorm1d(int(params["hidden_siglip"])),
            nn.Dropout(params["dropout"])
        )
        self.final = nn.Sequential(
            nn.Linear(int(params["hidden_manual"])+int(params["hidden_stella"])+int(params["hidden_siglip"]), int(params["final_hidden"])),
            nn.ReLU(),
            nn.BatchNorm1d(int(params["final_hidden"])),
            nn.Dropout(params["dropout"]),
            nn.Linear(int(params["final_hidden"]), 1)
        )
    def forward(self, x_m, x_s, x_i):
        m = self.manual_net(x_m)
        s = self.stella_net(x_s)
        i = self.siglip_net(x_i)
        combined = torch.cat([m,s,i], dim=1)
        return self.final(combined)

class DeepWideEmbeddingMLPQwen(nn.Module):
    def __init__(self, manual_dim, qwen_dim, siglip_dim,
                 hidden_manual, hidden_qwen, hidden_siglip,
                 final_hidden, dropout):
        super().__init__()
        self.manual_net = nn.Sequential(
            nn.Linear(manual_dim, hidden_manual),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_manual),
            nn.Dropout(dropout)
        )
        self.qwen_net = nn.Sequential(
            nn.Linear(qwen_dim, hidden_qwen),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_qwen),
            nn.Dropout(dropout)
        )
        self.siglip_net = nn.Sequential(
            nn.Linear(siglip_dim, hidden_siglip),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_siglip),
            nn.Dropout(dropout)
        )
        self.final = nn.Sequential(
            nn.Linear(hidden_manual + hidden_qwen + hidden_siglip, final_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(final_hidden),
            nn.Dropout(dropout),
            nn.Linear(final_hidden, 1)
        )

    def forward(self, x_m, x_q, x_i):
        m = self.manual_net(x_m)
        q = self.qwen_net(x_q)
        i = self.siglip_net(x_i)
        combined = torch.cat([m, q, i], dim=1)
        return self.final(combined)

# ======================= Load Models =======================
print("Loading models...")

# Load Stella+SigLIP MLP Models
mlp_models = []
for i, (path, params) in enumerate(zip(mlp_model_paths, mlp_params_list)):
    model = DeepWideEmbeddingMLP(len(manual_tfidf_cols), len(stella_cols), len(siglip_cols), params).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    mlp_models.append(model)
    print(f"Loaded Stella+SigLIP MLP model {i+1}")

# Load Qwen+SigLIP MLP Model
qwen_model = DeepWideEmbeddingMLPQwen(
    len(manual_tfidf_cols), len(qwen_cols), len(siglip_cols),
    qwen_mlp_params["hidden_manual"], qwen_mlp_params["hidden_qwen"],
    qwen_mlp_params["hidden_siglip"], qwen_mlp_params["final_hidden"],
    qwen_mlp_params["dropout"]
).to(device)
qwen_model.load_state_dict(torch.load(qwen_mlp_model_path))
qwen_model.eval()
print("Loaded Qwen+SigLIP MLP model")

# Load XGBoost Model
xgb_model = xgb.Booster()
xgb_model.load_model(xgb_model_path)
print("Loaded XGBoost model")

# ===================== Ensemble Inference =====================
print("Making predictions...")

# Get Stella+SigLIP MLP predictions
all_mlp_preds = [[] for _ in mlp_models]
with torch.no_grad():
    for xb_m, xb_s, xb_i in test_loader:
        xb_m, xb_s, xb_i = xb_m.to(device), xb_s.to(device), xb_i.to(device)
        for i, model in enumerate(mlp_models):
            preds_batch = model(xb_m, xb_s, xb_i)
            all_mlp_preds[i].append(preds_batch.cpu())

# Concatenate all batch predictions for each Stella+SigLIP MLP model
mlp_preds_tensors = [torch.cat(preds, dim=0) for preds in all_mlp_preds]

# Get Qwen+SigLIP MLP predictions
qwen_preds = []
with torch.no_grad():
    for xb_m, xb_q, xb_i in test_loader_qwen:
        xb_m, xb_q, xb_i = xb_m.to(device), xb_q.to(device), xb_i.to(device)
        preds_batch = qwen_model(xb_m, xb_q, xb_i)
        qwen_preds.append(preds_batch.cpu())

qwen_preds_tensor = torch.cat(qwen_preds, dim=0)

# Get XGBoost predictions on the UNSCALED data
X_test_xgb = np.hstack([X_manual_raw, X_stella_raw, X_siglip_raw])
xgb_preds = xgb_model.predict(xgb.DMatrix(X_test_xgb))
xgb_preds_tensor = torch.tensor(xgb_preds).unsqueeze(1)

# Check for NaN in individual model predictions
print("Checking individual model predictions for NaN:")
for i, preds_tensor in enumerate(mlp_preds_tensors):
    nan_count = torch.isnan(preds_tensor).sum().item()
    print(f"  Stella+SigLIP MLP {i+1}: {nan_count} NaN values")

qwen_nan_count = torch.isnan(qwen_preds_tensor).sum().item()
print(f"  Qwen+SigLIP MLP: {qwen_nan_count} NaN values")

xgb_nan_count = np.isnan(xgb_preds).sum()
print(f"  XGBoost: {xgb_nan_count} NaN values")

# Combine all predictions with equal weights
final_preds = torch.zeros_like(mlp_preds_tensors[0]).to(device)

# Add Stella+SigLIP MLP predictions
for i, preds_tensor in enumerate(mlp_preds_tensors):
    # Replace NaN with 0 before adding
    clean_preds = torch.nan_to_num(preds_tensor.to(device), nan=0.0)
    final_preds += ensemble_weights[i].to(device) * clean_preds

# Add Qwen+SigLIP MLP predictions
clean_qwen_preds = torch.nan_to_num(qwen_preds_tensor.to(device), nan=0.0)
final_preds += ensemble_weights[len(mlp_models)].to(device) * clean_qwen_preds

# Add XGBoost predictions
clean_xgb_preds = torch.nan_to_num(xgb_preds_tensor.to(device), nan=0.0)
final_preds += ensemble_weights[-1].to(device) * clean_xgb_preds

# Post-process predictions
final_predictions = final_preds.cpu().numpy().flatten()
pred_prices = np.expm1(final_predictions)  # Inverse log1p

# Debug information
print(f"Debug Info:")
print(f"Final predictions shape: {final_predictions.shape}")
print(f"Sample IDs shape: {sample_ids.shape}")
print(f"Pred prices shape: {pred_prices.shape}")
print(f"Sample of pred_prices: {pred_prices[:5]}")
print(f"Any NaN in pred_prices: {np.isnan(pred_prices).any()}")
print(f"Any inf in pred_prices: {np.isinf(pred_prices).any()}")

# ======================= Save to CSV ========================
submission = pd.DataFrame({"sample_id": sample_ids, "price": pred_prices})
print(f"Submission dataframe shape: {submission.shape}")
print(f"Submission columns: {submission.columns.tolist()}")
print(f"First few rows of submission:\n{submission.head()}")

submission.to_csv("test_predictions_mlp_xg_qwen.csv", index=False)
print("Predictions saved to test_predictions_mlp_xg_qwen.csv")

print("\n" + "="*50)
print("ENSEMBLE EVALUATION COMPLETE (NO GEMMA)")
print(f"Models used: {num_models}")
print("- 2 Stella+SigLIP MLP models")
print("- 1 Qwen+SigLIP MLP model")
print("- 1 XGBoost model")
print(f"Equal weights applied: {ensemble_weights.cpu().numpy()}")
print(f"Total test samples: {len(sample_ids)}")
print("="*50)