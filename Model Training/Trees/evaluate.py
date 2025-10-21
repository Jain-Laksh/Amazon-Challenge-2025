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

# --- MLP Models ---
mlp_model_paths = ["../MLP/optuna_best_48_46.pt", "../MLP/optuna_best_48_22.pt"]
mlp_params_list = [
    {"hidden_manual": 1024, "hidden_stella": 320, "hidden_siglip": 320, "final_hidden": 256, "dropout": 0.296},
    {"hidden_manual": 896, "hidden_stella": 384, "hidden_siglip": 320, "final_hidden": 448, "dropout": 0.296}
]

# --- XGBoost Model ---
xgb_model_path = "best_xgb_model_regression.json"

# --- Learned Ensemble Weights ---
# MODIFIED: Weights are now equal for all 3 models.
num_models = len(mlp_model_paths) + 1 # +1 for XGBoost
ensemble_weights = torch.tensor([1/num_models] * num_models, device=device)


# ==================== Load Test Data ====================
df_test = pd.read_csv("../test_final.csv")
sample_ids = df_test["sample_id"].values

manual_cols = [c for c in df_test.columns if not (c.startswith("stella_") or c.startswith("siglip_") or c.startswith("tfidf_") or c=="sample_id")]
tfidf_cols = [c for c in df_test.columns if c.startswith("tfidf_")]
manual_tfidf_cols = manual_cols + tfidf_cols
stella_cols = [c for c in df_test.columns if c.startswith("stella_")]
siglip_cols = [c for c in df_test.columns if c.startswith("siglip_")]

# Load raw, unscaled data
X_manual_raw = df_test[manual_tfidf_cols].values
X_stella_raw = df_test[stella_cols].values
X_siglip_raw = df_test[siglip_cols].values

# Clip extreme values
clip_val = 1e5
X_manual_raw = np.clip(X_manual_raw, -clip_val, clip_val)
X_stella_raw = np.clip(X_stella_raw, -clip_val, clip_val)
X_siglip_raw = np.clip(X_siglip_raw, -clip_val, clip_val)

# ==================== Preprocessing =====================
# MODIFIED: Fitting new scalers directly on the test data as requested.
scaler_manual = StandardScaler().fit(X_manual_raw)
scaler_stella = StandardScaler().fit(X_stella_raw)
scaler_siglip = StandardScaler().fit(X_siglip_raw)

# Create scaled data for MLPs
X_manual_scaled = scaler_manual.transform(X_manual_raw)
X_stella_scaled = scaler_stella.transform(X_stella_raw)
X_siglip_scaled = scaler_siglip.transform(X_siglip_raw)

# ====================== Dataset for MLPs ======================
class TestDataset(Dataset):
    def __init__(self, X_m, X_s, X_i):
        self.X_m = torch.tensor(X_m, dtype=torch.float32)
        self.X_s = torch.tensor(X_s, dtype=torch.float32)
        self.X_i = torch.tensor(X_i, dtype=torch.float32)
    def __len__(self):
        return len(self.X_m)
    def __getitem__(self, idx):
        return self.X_m[idx], self.X_s[idx], self.X_i[idx]

test_ds = TestDataset(X_manual_scaled, X_stella_scaled, X_siglip_scaled)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

# ===================== Model Definition =====================
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

# ======================= Load Models =======================
# Load MLP Models
mlp_models = []
for path, params in zip(mlp_model_paths, mlp_params_list):
    model = DeepWideEmbeddingMLP(len(manual_tfidf_cols), len(stella_cols), len(siglip_cols), params).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    mlp_models.append(model)

# Load XGBoost Model
xgb_model = xgb.Booster()
xgb_model.load_model(xgb_model_path)

# ===================== Ensemble Inference =====================
all_mlp_preds = [[] for _ in mlp_models]

with torch.no_grad():
    for xb_m, xb_s, xb_i in test_loader:
        xb_m, xb_s, xb_i = xb_m.to(device), xb_s.to(device), xb_i.to(device)
        for i, model in enumerate(mlp_models):
            preds_batch = model(xb_m, xb_s, xb_i)
            all_mlp_preds[i].append(preds_batch.cpu())

# Concatenate all batch predictions for each MLP model
mlp_preds_tensors = [torch.cat(preds, dim=0) for preds in all_mlp_preds]

# Get XGBoost predictions on the UNSCALED data
X_test_xgb = np.hstack([X_manual_raw, X_stella_raw, X_siglip_raw])
xgb_preds = xgb_model.predict(xgb.DMatrix(X_test_xgb))
xgb_preds_tensor = torch.tensor(xgb_preds).unsqueeze(1)

# Ensure all tensors are on the same device
final_preds = torch.zeros_like(mlp_preds_tensors[0]).to(device)
for i, preds_tensor in enumerate(mlp_preds_tensors):
    final_preds += ensemble_weights[i].to(device) * preds_tensor.to(device)

# Add the weighted XGBoost predictions
final_preds += ensemble_weights[-1].to(device) * xgb_preds_tensor.to(device)

# Post-process predictions
final_predictions = final_preds.cpu().numpy().flatten()
pred_prices = np.expm1(final_predictions) # Inverse log1p

# ======================= Save to CSV ========================
submission = pd.DataFrame({"sample_id": sample_ids, "price": pred_prices})
submission.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")