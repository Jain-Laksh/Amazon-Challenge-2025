import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler

# ========== Device ==========
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== Hyperparameters (from Optuna) ==========
# params = {
#     "batch_size": 256,
#     "lr": 0.002356942487509462,
#     "weight_decay": 0.00018832767605932036,
#     "dropout": 0.2962526572499149,
#     "hidden_manual": 1024,
#     "hidden_stella": 320,
#     "hidden_siglip": 320,
#     "final_hidden": 256,
#     "huber_delta": 5.644195842378224,
#     "epochs": 50,
#     "patience": 5,
#     "clip_grad": 5.0,
#     "input_clip": 1e5
# }

params = {
    "batch_size": 256,
    "lr": 0.0032220992669932786,
    "weight_decay": 0.0002312628964085081,
    "dropout": 0.2584560252965782,
    "hidden_manual": 896,
    "hidden_stella": 384,
    "hidden_siglip": 320,
    "final_hidden": 448,
    "huber_delta": 8.42681603724758,
    "epochs": 50,
    "patience": 5,
    "clip_grad": 5.0,
    "input_clip": 1e5
}

# ========== Load Data ==========
df = pd.read_csv("../train_final.csv")
y = df["price"].values
assert np.all(y >= 0), "Price must be non-negative for log1p"
y = np.log1p(y)

manual_cols = [c for c in df.columns if not (c.startswith("stella_") or c.startswith("siglip_") or c.startswith("tfidf_") or c in ["sample_id","price"])]
tfidf_cols = [c for c in df.columns if c.startswith("tfidf_")]
manual_tfidf_cols = manual_cols + tfidf_cols
stella_cols = [c for c in df.columns if c.startswith("stella_")]
siglip_cols = [c for c in df.columns if c.startswith("siglip_")]

X_manual = df[manual_tfidf_cols].values
X_stella = df[stella_cols].values
X_siglip = df[siglip_cols].values

# Clip extreme values
X_manual = np.clip(X_manual, -params["input_clip"], params["input_clip"])
X_stella = np.clip(X_stella, -params["input_clip"], params["input_clip"])
X_siglip = np.clip(X_siglip, -params["input_clip"], params["input_clip"])

# Scale separately
scaler_manual = StandardScaler()
scaler_stella = StandardScaler()
scaler_siglip = StandardScaler()
X_manual = scaler_manual.fit_transform(X_manual)
X_stella = scaler_stella.fit_transform(X_stella)
X_siglip = scaler_siglip.fit_transform(X_siglip)

# Train/validation split
X_m_train, X_m_val, X_s_train, X_s_val, X_i_train, X_i_val, y_train, y_val = train_test_split(
    X_manual, X_stella, X_siglip, y, test_size=0.1, random_state=42
)

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

train_ds = EmbeddingDataset(X_m_train, X_s_train, X_i_train, y_train)
val_ds = EmbeddingDataset(X_m_val, X_s_val, X_i_val, y_val)

train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False, num_workers=4)

# ========== Model ==========
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

model = DeepWideEmbeddingMLP(len(manual_tfidf_cols), len(stella_cols), len(siglip_cols), params).to(device)

# ========== Loss, optimizer, scaler ==========
criterion = nn.HuberLoss(delta=params["huber_delta"])
optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
scaler_amp = GradScaler()

# ========== SMAPE ==========
def smape(y_true, y_pred):
    denom = (torch.abs(y_true) + torch.abs(y_pred)).clamp(min=1e-6)
    return 200 * torch.mean(torch.abs(y_true - y_pred)/denom)

# ========== Training ==========
best_smape = float('inf')
epochs_no_improve = 0

for epoch in range(params["epochs"]):
    model.train()
    total_loss = 0
    for xb_m, xb_s, xb_i, yb in train_loader:
        xb_m, xb_s, xb_i, yb = xb_m.to(device), xb_s.to(device), xb_i.to(device), yb.to(device)
        optimizer.zero_grad()
        with autocast():
            pred = model(xb_m, xb_s, xb_i)
            loss = criterion(pred, yb)
        if torch.isnan(loss):
            continue
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params["clip_grad"])
        scaler_amp.step(optimizer)
        scaler_amp.update()
        total_loss += loss.item() * xb_m.size(0)
    
    train_loss = total_loss / len(train_loader.dataset)
    
    # Validation
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for xb_m, xb_s, xb_i, yb in val_loader:
            xb_m, xb_s, xb_i, yb = xb_m.to(device), xb_s.to(device), xb_i.to(device), yb.to(device)
            with autocast():
                pred = model(xb_m, xb_s, xb_i)
            preds.append(pred.cpu())
            truths.append(yb.cpu())
    preds = torch.cat(preds, dim=0)
    truths = torch.cat(truths, dim=0)
    val_smape = smape(torch.expm1(truths), torch.expm1(preds))
    
    print(f"Epoch {epoch+1}/{params['epochs']} | Train Loss: {train_loss:.4f} | Val SMAPE: {val_smape:.2f}")
    
    if val_smape < best_smape:
        best_smape = val_smape
        epochs_no_improve = 0
        torch.save(model.state_dict(), "optuna_best.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= params["patience"]:
            print(f"Early stopping triggered. Best SMAPE: {best_smape:.2f}")
            break

# Load best model
model.load_state_dict(torch.load("optuna_best.pt"))
print("Training complete. Best model loaded with SMAPE:", best_smape)
