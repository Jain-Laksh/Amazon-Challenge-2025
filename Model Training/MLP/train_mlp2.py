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

# ========== Config ==========
csv_path = "../train_final.csv"
target_col = "price"
batch_size = 512
epochs = 20
lr = 1e-3
weight_decay = 1e-5
patience = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== Load Data ==========
df = pd.read_csv(csv_path)
print(f"Data loaded with shape: {df.shape}")
y = np.log1p(df[target_col].values)

# Separate features
manual_tfidf_cols = [c for c in df.columns if c.startswith("tfidf_") or not (c.startswith("stella_") or c.startswith("siglip_") or c in ["sample_id", "price"])]
stella_cols = [c for c in df.columns if c.startswith("stella_")]
siglip_cols = [c for c in df.columns if c.startswith("siglip_")]

X_manual = df[manual_tfidf_cols].values
X_stella = df[stella_cols].values
X_siglip = df[siglip_cols].values

# ========== Scale features separately ==========
scaler_manual = StandardScaler()
scaler_stella = StandardScaler()
scaler_siglip = StandardScaler()

X_manual = scaler_manual.fit_transform(X_manual)
X_stella = scaler_stella.fit_transform(X_stella)
X_siglip = scaler_siglip.fit_transform(X_siglip)

# Train-validation split
X_manual_train, X_manual_val, X_stella_train, X_stella_val, X_siglip_train, X_siglip_val, y_train, y_val = train_test_split(
    X_manual, X_stella, X_siglip, y, test_size=0.1, random_state=42
)

# ========== Dataset ==========
class EmbeddingDataset(Dataset):
    def __init__(self, X_manual, X_stella, X_siglip, y):
        self.X_manual = torch.tensor(X_manual, dtype=torch.float32)
        self.X_stella = torch.tensor(X_stella, dtype=torch.float32)
        self.X_siglip = torch.tensor(X_siglip, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X_manual)

    def __getitem__(self, idx):
        return self.X_manual[idx], self.X_stella[idx], self.X_siglip[idx], self.y[idx]

train_ds = EmbeddingDataset(X_manual_train, X_stella_train, X_siglip_train, y_train)
val_ds = EmbeddingDataset(X_manual_val, X_stella_val, X_siglip_val, y_val)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

# ========== Model ==========
class DeepWideEmbeddingMLP(nn.Module):
    def __init__(self, manual_dim, stella_dim, siglip_dim):
        super().__init__()
        # Subnetworks
        self.manual_net = nn.Sequential(
            nn.Linear(manual_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2)
        )
        self.stella_net = nn.Sequential(
            nn.Linear(stella_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2)
        )
        self.siglip_net = nn.Sequential(
            nn.Linear(siglip_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2)
        )

        # Final MLP
        self.final = nn.Sequential(
            nn.Linear(512 + 256 + 256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, x_manual, x_stella, x_siglip):
        m = self.manual_net(x_manual)
        s = self.stella_net(x_stella)
        i = self.siglip_net(x_siglip)
        combined = torch.cat([m, s, i], dim=1)
        return self.final(combined)

model = DeepWideEmbeddingMLP(len(manual_tfidf_cols), len(stella_cols), len(siglip_cols)).to(device)

# ========== Loss & Optimizer ==========
criterion = nn.HuberLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scaler_amp = GradScaler()

# ========== SMAPE ==========
def smape(y_true, y_pred):
    denom = (torch.abs(y_true) + torch.abs(y_pred)).clamp(min=1e-6)
    return 200 * torch.mean(torch.abs(y_true - y_pred) / denom)

# ========== Training Loop (simplified logging) ==========
best_smape = float('inf')
epochs_no_improve = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb_m, xb_s, xb_i, yb in train_loader:
        xb_m, xb_s, xb_i, yb = xb_m.to(device), xb_s.to(device), xb_i.to(device), yb.to(device)
        optimizer.zero_grad()
        with autocast():
            pred = model(xb_m, xb_s, xb_i)
            loss = criterion(pred, yb)
        scaler_amp.scale(loss).backward()
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

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val SMAPE: {val_smape:.2f}")

    # Early stopping
    if val_smape < best_smape:
        best_smape = val_smape
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered. Best SMAPE: {best_smape:.2f}")
            break

# Load best model
model.load_state_dict(torch.load("best_model.pt"))
