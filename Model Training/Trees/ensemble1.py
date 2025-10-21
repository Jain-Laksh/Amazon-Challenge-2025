import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

# ================= Load and preprocess data =================
df = pd.read_csv("../train_final.csv")
y = df["price"].values
y = np.log1p(y)  # log scale

manual_cols = [c for c in df.columns if not (c.startswith("stella_") or c.startswith("siglip_") or c.startswith("tfidf_") or c in ["sample_id","price"])]
tfidf_cols = [c for c in df.columns if c.startswith("tfidf_")]
manual_tfidf_cols = manual_cols + tfidf_cols
stella_cols = [c for c in df.columns if c.startswith("stella_")]
siglip_cols = [c for c in df.columns if c.startswith("siglip_")]

X_manual = df[manual_tfidf_cols].values
X_stella = df[stella_cols].values
X_siglip = df[siglip_cols].values

# Clip extreme values
clip_val = 1e5
X_manual = np.clip(X_manual, -clip_val, clip_val)
X_stella = np.clip(X_stella, -clip_val, clip_val)
X_siglip = np.clip(X_siglip, -clip_val, clip_val)


# ================= Corrected Data Splitting and Scaling =================
# 1. Split the raw, unscaled data first to ensure alignment
X_m_train_raw, X_m_val_raw, \
X_s_train_raw, X_s_val_raw, \
X_i_train_raw, X_i_val_raw, \
y_train, y_val = train_test_split(
    X_manual, X_stella, X_siglip, y, test_size=0.25, random_state=42
)

# 2. Fit scalers ONLY on the training data to prevent data leakage
scaler_manual = StandardScaler().fit(X_m_train_raw)
scaler_stella = StandardScaler().fit(X_s_train_raw)
scaler_siglip = StandardScaler().fit(X_i_train_raw)

# 3. Transform both train and validation sets for MLPs
X_m_train = scaler_manual.transform(X_m_train_raw)
X_m_val = scaler_manual.transform(X_m_val_raw)

X_s_train = scaler_stella.transform(X_s_train_raw)
X_s_val = scaler_stella.transform(X_s_val_raw)

X_i_train = scaler_siglip.transform(X_i_train_raw)
X_i_val = scaler_siglip.transform(X_i_val_raw)

# ========== Dataset ==========
class EmbeddingDataset(Dataset):
    def __init__(self, X_m, X_s, X_i, y):
        self.X_m = torch.tensor(X_m, dtype=torch.float32)
        self.X_s = torch.tensor(X_s, dtype=torch.float32)
        self.X_i = torch.tensor(X_i, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X_m)
    def __getitem__(self, idx):
        return self.X_m[idx], self.X_s[idx], self.X_i[idx], self.y[idx]

# This now correctly uses the scaled validation data
val_ds = EmbeddingDataset(X_m_val, X_s_val, X_i_val, y_val)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

# ========== Define MLP ==========
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
            nn.Linear(int(params["hidden_manual"]) + int(params["hidden_stella"]) + int(params["hidden_siglip"]), int(params["final_hidden"])),
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

# ========== Load MLP models ==========
device = "cuda" if torch.cuda.is_available() else "cpu"

params1 = {"hidden_manual":1024,"hidden_stella":320,"hidden_siglip":320,"final_hidden":256,"dropout":0.296}
model1 = DeepWideEmbeddingMLP(len(manual_tfidf_cols), len(stella_cols), len(siglip_cols), params1).to(device)
model1.load_state_dict(torch.load("../MLP/optuna_best_48_46_siglip_stella.pt"))
model1.eval()

params2 = {"hidden_manual":896,"hidden_stella":384,"hidden_siglip":320,"final_hidden":448,"dropout":0.296}
model2 = DeepWideEmbeddingMLP(len(manual_tfidf_cols), len(stella_cols), len(siglip_cols), params2).to(device)
model2.load_state_dict(torch.load("../MLP/optuna_best_48_22_siglip_stella.pt"))
model2.eval()

mlp_models = [model1, model2]

# ========== Load XGBoost model ==========
xgb_model = xgb.Booster()
xgb_model.load_model("best_xgb_model_regression.json")

# ========== SMAPE ==========
def smape(y_true, y_pred):
    y_true = y_true.to(y_pred.device)
    denom = (torch.abs(y_true) + torch.abs(y_pred)).clamp(min=1e-8) # Use a slightly smaller epsilon
    return 200 * torch.mean(torch.abs(y_true - y_pred)/denom)

# ========== Weighted Ensemble ==========
weights = torch.tensor([1.0/(len(mlp_models)+1)]*(len(mlp_models)+1), requires_grad=True, device=device)
optimizer = torch.optim.Adam([weights], lr=0.01)

# 4. Use the correctly aligned UNSCALED validation data for XGBoost
X_val_xgb = np.hstack([X_m_val_raw, X_s_val_raw, X_i_val_raw]).astype(np.float32)

xgb_val_preds = torch.tensor(
    xgb_model.predict(xgb.DMatrix(X_val_xgb)),
    dtype=torch.float32,
    device=device
).unsqueeze(1)

# Collect MLP predictions
val_preds_mlp = []
val_truths = []
with torch.no_grad():
    for xb_m, xb_s, xb_i, yb in val_loader:
        xb_m, xb_s, xb_i = xb_m.to(device), xb_s.to(device), xb_i.to(device)
        preds_batch = [m(xb_m, xb_s, xb_i) for m in mlp_models]
        val_preds_mlp.append(torch.stack(preds_batch, dim=0))
        val_truths.append(yb)

val_preds_mlp = torch.cat(val_preds_mlp, dim=1) # Shape: [n_mlp_models, total_samples, 1]
val_truths = torch.cat(val_truths, dim=0) # Shape: [total_samples, 1]

# Append XGBoost predictions (correctly aligned)
# Shape of val_preds_mlp: [2, N, 1]
# Shape of xgb_val_preds: [N, 1] -> unsqueeze to [1, N, 1]
val_preds = torch.cat([val_preds_mlp, xgb_val_preds.unsqueeze(0)], dim=0) # Final shape: [n_models_total, total_samples, 1]

# Optimize ensemble weights
for step in range(200):
    optimizer.zero_grad()
    
    # Ensure weights sum to 1 and are non-negative at each step
    with torch.no_grad():
        weights.clamp_(min=0)
        weights.div_(weights.sum())

    # Calculate weighted prediction
    weighted_pred = torch.sum(weights.view(-1, 1, 1) * val_preds, dim=0)
    
    # Calculate loss and backpropagate
    loss = smape(torch.expm1(val_truths), torch.expm1(weighted_pred))
    loss.backward() # CORRECTED: Minimize the loss, do not maximize it.
    
    optimizer.step()
    
    if step % 20 == 0:
        print(f"Step {step}, SMAPE: {loss.item():.4f}, Weights: {weights.detach().cpu().numpy()}")

# Final projection of weights after training loop
with torch.no_grad():
    weights.clamp_(min=0)
    weights.div_(weights.sum())

weighted_pred = torch.sum(weights.view(-1,1,1) * val_preds, dim=0)
ensemble_smape = smape(torch.expm1(val_truths), torch.expm1(weighted_pred))
print("\n" + "="*30)
print(f"Final Ensemble SMAPE: {ensemble_smape.item():.4f}")
print(f"Learned Ensemble Weights: {weights.detach().cpu().numpy()}")
print("="*30)