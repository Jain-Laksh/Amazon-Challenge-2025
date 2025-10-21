import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ========== Load and preprocess data ==========
df = pd.read_csv("../train_final.csv")
y = df["price"].values
y = np.log1p(y)

manual_cols = [c for c in df.columns if not (c.startswith("stella_") or c.startswith("siglip_") or c.startswith("tfidf_") or c in ["sample_id","price"])]
tfidf_cols = [c for c in df.columns if c.startswith("tfidf_")]
manual_tfidf_cols = manual_cols + tfidf_cols
stella_cols = [c for c in df.columns if c.startswith("stella_")]
siglip_cols = [c for c in df.columns if c.startswith("siglip_")]

X_manual_raw = df[manual_tfidf_cols].values
X_stella_raw = df[stella_cols].values
X_siglip_raw = df[siglip_cols].values

# Clip extreme values
clip_val = 1e5
X_manual_raw = np.clip(X_manual_raw, -clip_val, clip_val)
X_stella_raw = np.clip(X_stella_raw, -clip_val, clip_val)
X_siglip_raw = np.clip(X_siglip_raw, -clip_val, clip_val)

# CORRECTED: 1. Split the raw data first
X_m_train_raw, X_m_val_raw, \
X_s_train_raw, X_s_val_raw, \
X_i_train_raw, X_i_val_raw, \
y_train, y_val = train_test_split(
    X_manual_raw, X_stella_raw, X_siglip_raw, y, test_size=0.1, random_state=42
)

# CORRECTED: 2. Fit scalers ONLY on training data
scaler_manual = StandardScaler().fit(X_m_train_raw)
scaler_stella = StandardScaler().fit(X_s_train_raw)
scaler_siglip = StandardScaler().fit(X_i_train_raw)

# CORRECTED: 3. Transform both train and validation sets
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

# ========== Load saved models ==========
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model 1
params1 = {"hidden_manual": 1024, "hidden_stella": 320, "hidden_siglip": 320, "final_hidden": 256, "dropout": 0.296}
model1 = DeepWideEmbeddingMLP(len(manual_tfidf_cols), len(stella_cols), len(siglip_cols), params1).to(device)
model1.load_state_dict(torch.load("optuna_best_48_46.pt"))
model1.eval()

# Model 2
params2 = {"hidden_manual": 896, "hidden_stella": 384, "hidden_siglip": 320, "final_hidden": 448, "dropout": 0.296}
model2 = DeepWideEmbeddingMLP(len(manual_tfidf_cols), len(stella_cols), len(siglip_cols), params2).to(device)
model2.load_state_dict(torch.load("optuna_best_48_22.pt"))
model2.eval()

models = [model1, model2]

# ========== SMAPE ==========
def smape(y_true, y_pred):
    y_true = y_true.to(y_pred.device)
    denom = (torch.abs(y_true) + torch.abs(y_pred)).clamp(min=1e-8)
    return 200 * torch.mean(torch.abs(y_true - y_pred)/denom)


# ========== Weighted Ensemble ==========
weights = torch.tensor([1.0/len(models)]*len(models), requires_grad=True, device=device)
optimizer = torch.optim.Adam([weights], lr=0.01)

# Collect model predictions on validation set
val_preds = []
val_truths = []
with torch.no_grad():
    for xb_m, xb_s, xb_i, yb in val_loader:
        xb_m, xb_s, xb_i = xb_m.to(device), xb_s.to(device), xb_i.to(device)
        preds_batch = [model(xb_m, xb_s, xb_i) for model in models]
        val_preds.append(torch.stack(preds_batch, dim=0))
        val_truths.append(yb)

val_preds = torch.cat(val_preds, dim=1)
val_truths = torch.cat(val_truths, dim=0)

# Optimize weights to minimize SMAPE
for step in range(200):
    optimizer.zero_grad()
    weighted_pred = torch.sum(weights.view(-1,1,1) * val_preds, dim=0)
    loss = smape(torch.expm1(val_truths), torch.expm1(weighted_pred))
    
    # CORRECTED: Perform gradient descent to MINIMIZE the loss
    loss.backward()
    
    optimizer.step()
    
    # Project weights to be non-negative and sum to 1
    with torch.no_grad():
        weights.clamp_(min=0)
        weights.div_(weights.sum())
        
    if step % 20 == 0:
        print(f"Step {step}, SMAPE: {loss.item():.4f}, Weights: {weights.detach().cpu().numpy()}")

# Final ensemble prediction
weighted_pred = torch.sum(weights.view(-1,1,1) * val_preds, dim=0)
ensemble_smape = smape(torch.expm1(val_truths), torch.expm1(weighted_pred))
print("\n" + "="*30)
print(f"Final Ensemble SMAPE: {ensemble_smape.item():.4f}")
print(f"Learned Ensemble Weights: {weights.detach().cpu().numpy()}")
print("="*30)