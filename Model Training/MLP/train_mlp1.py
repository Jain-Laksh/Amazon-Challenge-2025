import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler

# ========== Config ==========
csv_path = "../train_final.csv"
target_col = "price"
batch_size = 512
epochs = 50
lr = 1e-3
weight_decay = 1e-5
patience = 5
log_interval = 20  # print every 20 batches
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== Load Data ==========
df = pd.read_csv(csv_path)
print(f"Data loaded with shape: {df.shape}")
y = np.log1p(df[target_col].values)
X = df.drop(columns=["sample_id", target_col])

# # Trying with dropping tfidf columns
# tfidf_cols = [col for col in X.columns if col.startswith('tfidf_')]
# X = X.drop(columns=tfidf_cols)

# # Trying with only tfidf columns and manual features
# cols_to_drop = [col for col in X.columns if col.startswith('stella_') or col.startswith('siglip_')]
# X = X.drop(columns=cols_to_drop)

# ========== Scale Features ==========
scaler = StandardScaler()
# scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# ========== Dataset ==========
class PriceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_ds = PriceDataset(X_train, y_train)
val_ds = PriceDataset(X_val, y_val)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

# ========== Model ==========
class DeepWideMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        self.output = nn.Linear(input_dim + 512, 1)

    def forward(self, x):
        deep_out = self.deep(x)
        combined = torch.cat([x, deep_out], dim=1)
        return self.output(combined)

input_dim = X_train.shape[1]
model = DeepWideMLP(input_dim).to(device)

# ========== Loss & Optimizer ==========
criterion = nn.HuberLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scaler_amp = GradScaler()

# ========== SMAPE Metric ==========
def smape(y_true, y_pred):
    denom = (torch.abs(y_true) + torch.abs(y_pred)).clamp(min=1e-6)
    return 200 * torch.mean(torch.abs(y_true - y_pred) / denom)

# ========== Training Loop with Logging ==========
best_smape = float('inf')
epochs_no_improve = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    print(f"\n=== Epoch {epoch+1}/{epochs} ===")
    for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        with autocast():
            pred = model(xb)
            loss = criterion(pred, yb)
        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()
        total_loss += loss.item() * xb.size(0)

        if batch_idx % log_interval == 0 or batch_idx == len(train_loader):
            avg_loss = total_loss / (batch_idx * batch_size)
            print(f"Batch {batch_idx}/{len(train_loader)} | Avg Loss: {avg_loss:.4f}")

    train_loss = total_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            with autocast():
                pred = model(xb)
            preds.append(pred.cpu())
            truths.append(yb.cpu())
    preds = torch.cat(preds, dim=0)
    truths = torch.cat(truths, dim=0)
    val_smape = smape(torch.expm1(truths), torch.expm1(preds))
    print(f"Epoch {epoch+1} finished | Train Loss: {train_loss:.4f} | Val SMAPE: {val_smape:.2f}")

    # Early stopping
    if val_smape < best_smape:
        best_smape = val_smape
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model.pt")
        print(f"New best model saved with SMAPE: {best_smape:.2f}")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s)")
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered. Best SMAPE: {best_smape:.2f}")
            break

# Load best model
model.load_state_dict(torch.load("best_model.pt"))
print("Training complete. Best model loaded.")
