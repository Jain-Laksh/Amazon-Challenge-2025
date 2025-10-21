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

# --- Gemma MLP Model ---
gemma_mlp_model_path = "../MLP/optuna_best_siglip_gemma.pth"
gemma_mlp_params = {
    "hidden_manual": 256,
    "hidden_gemma": 384,
    "hidden_siglip": 448,
    "final_hidden": 448,
    "dropout": 0.271
}

# --- XGBoost Model ---
xgb_model_path = "best_xgb_model_regression.json"

# --- Equal Ensemble Weights ---
num_models = len(mlp_model_paths) + 1 + 1  # +1 for Gemma MLP, +1 for XGBoost
ensemble_weights = torch.tensor([1/num_models] * num_models, device=device)

print(f"Using {num_models} models with equal weights: {ensemble_weights.cpu().numpy()}")

# ==================== Load Test Data ====================
df_test = pd.read_csv("../test_final.csv")
print(f"Test data loaded with shape: {df_test.shape}")

# Load Gemma embeddings for test data
gemma_test_df = pd.read_csv("../gemma_embeddings_test.csv")
print(f"Gemma test embeddings loaded with shape: {gemma_test_df.shape}")

# Rename embedding columns
embedding_cols = [c for c in gemma_test_df.columns if c != "sample_id"]
gemma_test_df.rename(columns={c: f"gemma_{c}" for c in embedding_cols}, inplace=True)

# Merge test data with Gemma embeddings
df_test = df_test.merge(gemma_test_df, on="sample_id", how="inner")
print(f"Merged test dataset shape: {df_test.shape}")

sample_ids = df_test["sample_id"].values

# Define column groups
manual_cols = [c for c in df_test.columns if not (c.startswith("stella_") or c.startswith("siglip_") 
                                                   or c.startswith("tfidf_") or c.startswith("gemma_")
                                                   or c == "sample_id")]
tfidf_cols = [c for c in df_test.columns if c.startswith("tfidf_")]
manual_tfidf_cols = manual_cols + tfidf_cols
stella_cols = [c for c in df_test.columns if c.startswith("stella_")]
siglip_cols = [c for c in df_test.columns if c.startswith("siglip_")]
gemma_cols = [c for c in df_test.columns if c.startswith("gemma_")]

# Load raw, unscaled data
X_manual_raw = df_test[manual_tfidf_cols].values
X_stella_raw = df_test[stella_cols].values
X_siglip_raw = df_test[siglip_cols].values
X_gemma_raw = df_test[gemma_cols].values

# Clip extreme values
clip_val = 1e5
X_manual_raw = np.clip(X_manual_raw, -clip_val, clip_val)
X_stella_raw = np.clip(X_stella_raw, -clip_val, clip_val)
X_siglip_raw = np.clip(X_siglip_raw, -clip_val, clip_val)
X_gemma_raw = np.clip(X_gemma_raw, -clip_val, clip_val)

# ==================== Preprocessing =====================
# Fitting scalers directly on the test data
scaler_manual = StandardScaler().fit(X_manual_raw)
scaler_stella = StandardScaler().fit(X_stella_raw)
scaler_siglip = StandardScaler().fit(X_siglip_raw)
scaler_gemma = StandardScaler().fit(X_gemma_raw)

# Create scaled data for MLPs
X_manual_scaled = scaler_manual.transform(X_manual_raw)
X_stella_scaled = scaler_stella.transform(X_stella_raw)
X_siglip_scaled = scaler_siglip.transform(X_siglip_raw)
X_gemma_scaled = scaler_gemma.transform(X_gemma_raw)

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

class TestDatasetGemma(Dataset):
    def __init__(self, X_m, X_g, X_i):
        self.X_m = torch.tensor(X_m, dtype=torch.float32)
        self.X_g = torch.tensor(X_g, dtype=torch.float32)
        self.X_i = torch.tensor(X_i, dtype=torch.float32)
    def __len__(self):
        return len(self.X_m)
    def __getitem__(self, idx):
        return self.X_m[idx], self.X_g[idx], self.X_i[idx]

# Create datasets
test_ds = TestDataset(X_manual_scaled, X_stella_scaled, X_siglip_scaled)
test_ds_gemma = TestDatasetGemma(X_manual_scaled, X_gemma_scaled, X_siglip_scaled)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
test_loader_gemma = DataLoader(test_ds_gemma, batch_size=512, shuffle=False)

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

class DeepWideEmbeddingMLPGemma(nn.Module):
    def __init__(self, manual_dim, gemma_dim, siglip_dim,
                 hidden_manual, hidden_gemma, hidden_siglip,
                 final_hidden, dropout):
        super().__init__()
        self.manual_net = nn.Sequential(
            nn.Linear(manual_dim, hidden_manual),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_manual),
            nn.Dropout(dropout)
        )
        self.gemma_net = nn.Sequential(
            nn.Linear(gemma_dim, hidden_gemma),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_gemma),
            nn.Dropout(dropout)
        )
        self.siglip_net = nn.Sequential(
            nn.Linear(siglip_dim, hidden_siglip),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_siglip),
            nn.Dropout(dropout)
        )
        self.final = nn.Sequential(
            nn.Linear(hidden_manual + hidden_gemma + hidden_siglip, final_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(final_hidden),
            nn.Dropout(dropout),
            nn.Linear(final_hidden, 1)
        )

    def forward(self, x_m, x_g, x_i):
        m = self.manual_net(x_m)
        g = self.gemma_net(x_g)
        i = self.siglip_net(x_i)
        combined = torch.cat([m, g, i], dim=1)
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

# Load Gemma+SigLIP MLP Model
gemma_model = DeepWideEmbeddingMLPGemma(
    len(manual_tfidf_cols), len(gemma_cols), len(siglip_cols),
    gemma_mlp_params["hidden_manual"], gemma_mlp_params["hidden_gemma"],
    gemma_mlp_params["hidden_siglip"], gemma_mlp_params["final_hidden"],
    gemma_mlp_params["dropout"]
).to(device)
gemma_model.load_state_dict(torch.load(gemma_mlp_model_path))
gemma_model.eval()
print("Loaded Gemma+SigLIP MLP model")

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

# Get Gemma+SigLIP MLP predictions
gemma_preds = []
with torch.no_grad():
    for xb_m, xb_g, xb_i in test_loader_gemma:
        xb_m, xb_g, xb_i = xb_m.to(device), xb_g.to(device), xb_i.to(device)
        preds_batch = gemma_model(xb_m, xb_g, xb_i)
        gemma_preds.append(preds_batch.cpu())

gemma_preds_tensor = torch.cat(gemma_preds, dim=0)

# Get XGBoost predictions on the UNSCALED data
X_test_xgb = np.hstack([X_manual_raw, X_stella_raw, X_siglip_raw])
xgb_preds = xgb_model.predict(xgb.DMatrix(X_test_xgb))
xgb_preds_tensor = torch.tensor(xgb_preds).unsqueeze(1)

# Combine all predictions with equal weights
final_preds = torch.zeros_like(mlp_preds_tensors[0]).to(device)

# Add Stella+SigLIP MLP predictions
for i, preds_tensor in enumerate(mlp_preds_tensors):
    final_preds += ensemble_weights[i].to(device) * preds_tensor.to(device)

# Add Gemma+SigLIP MLP predictions
final_preds += ensemble_weights[len(mlp_models)].to(device) * gemma_preds_tensor.to(device)

# Add XGBoost predictions
final_preds += ensemble_weights[-1].to(device) * xgb_preds_tensor.to(device)

# Post-process predictions
final_predictions = final_preds.cpu().numpy().flatten()
pred_prices = np.expm1(final_predictions)  # Inverse log1p

# ======================= Save to CSV ========================
submission = pd.DataFrame({"sample_id": sample_ids, "price": pred_prices})
submission.to_csv("test_predictions_mlp_xg_gemma.csv", index=False)
print("Predictions saved to test_predictions_mlp_xg_gemma.csv")

print("\n" + "="*50)
print("ENSEMBLE EVALUATION COMPLETE")
print(f"Models used: {num_models}")
print("- 2 Stella+SigLIP MLP models")
print("- 1 Gemma+SigLIP MLP model") 
print("- 1 XGBoost model")
print(f"Equal weights applied: {ensemble_weights.cpu().numpy()}")
print(f"Total test samples: {len(sample_ids)}")
print("="*50)
