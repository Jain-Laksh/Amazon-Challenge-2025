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
import optuna

# ========== Load Data ==========
df = pd.read_csv("../train_final.csv")
print(f"Data loaded with shape: {df.shape}")
y = df["price"].values
assert np.all(y >= 0), "Price must be non-negative"
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
clip_val = 1e5
X_manual = np.clip(X_manual, -clip_val, clip_val)
X_stella = np.clip(X_stella, -clip_val, clip_val)
X_siglip = np.clip(X_siglip, -clip_val, clip_val)

# Scale separately
scaler_manual = StandardScaler()
scaler_stella = StandardScaler()
scaler_siglip = StandardScaler()
X_manual = scaler_manual.fit_transform(X_manual)
X_stella = scaler_stella.fit_transform(X_stella)
X_siglip = scaler_siglip.fit_transform(X_siglip)

# Train/val split
X_m_train, X_m_val, X_s_train, X_s_val, X_i_train, X_i_val, y_train, y_val = train_test_split(
    X_manual, X_stella, X_siglip, y, test_size=0.1, random_state=42
)

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

device = "cuda" if torch.cuda.is_available() else "cpu"

class DeepWideEmbeddingMLP(nn.Module):
    def __init__(self, manual_dim, stella_dim, siglip_dim, hidden_manual, hidden_stella, hidden_siglip, final_hidden, dropout):
        super().__init__()
        self.manual_net = nn.Sequential(
            nn.Linear(manual_dim, hidden_manual),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_manual),
            nn.Dropout(dropout)
        )
        self.stella_net = nn.Sequential(
            nn.Linear(stella_dim, hidden_stella),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_stella),
            nn.Dropout(dropout)
        )
        self.siglip_net = nn.Sequential(
            nn.Linear(siglip_dim, hidden_siglip),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_siglip),
            nn.Dropout(dropout)
        )
        self.final = nn.Sequential(
            nn.Linear(hidden_manual + hidden_stella + hidden_siglip, final_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(final_hidden),
            nn.Dropout(dropout),
            nn.Linear(final_hidden, 1)
        )
    def forward(self, x_m, x_s, x_i):
        m = self.manual_net(x_m)
        s = self.stella_net(x_s)
        i = self.siglip_net(x_i)
        combined = torch.cat([m,s,i], dim=1)
        return self.final(combined)

# SMAPE metric
def smape(y_true, y_pred):
    denom = (torch.abs(y_true) + torch.abs(y_pred)).clamp(min=1e-6)
    return 200 * torch.mean(torch.abs(y_true - y_pred)/denom)

# ========== Objective function for Optuna ==========
def objective(trial):
    # Hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    lr = trial.suggest_loguniform("lr", 1e-4, 5e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    hidden_manual = trial.suggest_int("hidden_manual", 256, 1024, step=128)
    hidden_stella = trial.suggest_int("hidden_stella", 128, 512, step=64)
    hidden_siglip = trial.suggest_int("hidden_siglip", 128, 512, step=64)
    final_hidden = trial.suggest_int("final_hidden", 256, 512, step=64)
    huber_delta = trial.suggest_float("huber_delta", 1.0, 10.0)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = DeepWideEmbeddingMLP(len(manual_tfidf_cols), len(stella_cols), len(siglip_cols),
                                hidden_manual, hidden_stella, hidden_siglip, final_hidden, dropout).to(device)
    
    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler_amp = GradScaler()
    
    best_val_smape = float('inf')
    patience_counter = 0
    max_epochs = 20
    
    for epoch in range(max_epochs):
        model.train()
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        
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
        
        # Early stopping
        if val_smape < best_val_smape:
            best_val_smape = val_smape
            patience_counter = 0
            torch.save(model.state_dict(), "best_trial_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                break
    return best_val_smape.item()

# ========== Run Optuna ==========
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)  # try 30 combinations

print("Best SMAPE:", study.best_value)
print("Best hyperparameters:", study.best_params)

# Load best model
best_model = DeepWideEmbeddingMLP(
    len(manual_tfidf_cols), len(stella_cols), len(siglip_cols),
    study.best_params["hidden_manual"],
    study.best_params["hidden_stella"],
    study.best_params["hidden_siglip"],
    study.best_params["final_hidden"],
    study.best_params["dropout"]
).to(device)
best_model.load_state_dict(torch.load("best_trial_model.pt"))
print("Best model loaded and ready.")
