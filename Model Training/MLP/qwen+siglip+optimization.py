import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
print(f"Train data loaded with shape: {df.shape}")

qwen_df = pd.read_csv("../qwen_embeddings_train.csv")
print(f"Qwen embeddings loaded with shape: {qwen_df.shape}")

# Rename embedding columns
embedding_cols = [c for c in qwen_df.columns if c != "sample_id"]
qwen_df.rename(columns={c: f"qwen_{c}" for c in embedding_cols}, inplace=True)

# Merge
df = df.merge(qwen_df, on="sample_id", how="inner")
print(f"Merged dataset shape: {df.shape}")

# Target
y = np.log1p(df["price"].values)
assert np.all(y >= 0), "Price must be non-negative"

# Separate column groups
manual_cols = [c for c in df.columns if not (c.startswith("stella_") or c.startswith("siglip_")
                                             or c.startswith("tfidf_") or c.startswith("qwen_")
                                             or c in ["sample_id", "price"])]
tfidf_cols = [c for c in df.columns if c.startswith("tfidf_")]
manual_tfidf_cols = manual_cols + tfidf_cols

qwen_cols = [c for c in df.columns if c.startswith("qwen_")]
siglip_cols = [c for c in df.columns if c.startswith("siglip_")]

# Extract features
X_manual = df[manual_tfidf_cols].values
X_qwen = df[qwen_cols].values
X_siglip = df[siglip_cols].values

# Clip extreme values
clip_val = 1e5
X_manual = np.clip(X_manual, -clip_val, clip_val)
X_qwen = np.clip(X_qwen, -clip_val, clip_val)
X_siglip = np.clip(X_siglip, -clip_val, clip_val)

# Scale separately
scaler_manual = StandardScaler()
scaler_qwen = StandardScaler()
scaler_siglip = StandardScaler()
X_manual = scaler_manual.fit_transform(X_manual)
X_qwen = scaler_qwen.fit_transform(X_qwen)
X_siglip = scaler_siglip.fit_transform(X_siglip)

# Split
X_m_train, X_m_val, X_q_train, X_q_val, X_i_train, X_i_val, y_train, y_val = train_test_split(
    X_manual, X_qwen, X_siglip, y, test_size=0.1, random_state=42
)

# ========== Dataset ==========
class EmbeddingDataset(Dataset):
    def __init__(self, X_m, X_q, X_i, y):
        self.X_m = torch.tensor(X_m, dtype=torch.float32)
        self.X_q = torch.tensor(X_q, dtype=torch.float32)
        self.X_i = torch.tensor(X_i, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X_m)

    def __getitem__(self, idx):
        return self.X_m[idx], self.X_q[idx], self.X_i[idx], self.y[idx]

train_ds = EmbeddingDataset(X_m_train, X_q_train, X_i_train, y_train)
val_ds = EmbeddingDataset(X_m_val, X_q_val, X_i_val, y_val)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== Model ==========
class DeepWideEmbeddingMLP(nn.Module):
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

# ========== SMAPE ==========
def smape(y_true, y_pred):
    denom = (torch.abs(y_true) + torch.abs(y_pred)).clamp(min=1e-6)
    return 200 * torch.mean(torch.abs(y_true - y_pred) / denom)

# ========== Objective ==========
def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    lr = trial.suggest_loguniform("lr", 1e-4, 5e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    hidden_manual = trial.suggest_int("hidden_manual", 256, 1024, step=128)
    hidden_qwen = trial.suggest_int("hidden_qwen", 128, 512, step=64)
    hidden_siglip = trial.suggest_int("hidden_siglip", 128, 512, step=64)
    final_hidden = trial.suggest_int("final_hidden", 256, 512, step=64)
    huber_delta = trial.suggest_float("huber_delta", 1.0, 10.0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = DeepWideEmbeddingMLP(
        len(manual_tfidf_cols), len(qwen_cols), len(siglip_cols),
        hidden_manual, hidden_qwen, hidden_siglip, final_hidden, dropout
    ).to(device)

    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler_amp = GradScaler()

    best_val_smape = float('inf')
    patience_counter = 0
    max_epochs = 20

    for epoch in range(max_epochs):
        model.train()
        for xb_m, xb_q, xb_i, yb in train_loader:
            xb_m, xb_q, xb_i, yb = xb_m.to(device), xb_q.to(device), xb_i.to(device), yb.to(device)
            optimizer.zero_grad()
            with autocast():
                pred = model(xb_m, xb_q, xb_i)
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
            for xb_m, xb_q, xb_i, yb in val_loader:
                xb_m, xb_q, xb_i, yb = xb_m.to(device), xb_q.to(device), xb_i.to(device), yb.to(device)
                with autocast():
                    pred = model(xb_m, xb_q, xb_i)
                preds.append(pred.cpu())
                truths.append(yb.cpu())
        preds = torch.cat(preds, dim=0)
        truths = torch.cat(truths, dim=0)
        val_smape = smape(torch.expm1(truths), torch.expm1(preds))

        if val_smape < best_val_smape:
            best_val_smape = val_smape
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                break

    return best_val_smape.item()

# ========== Run Optuna ==========
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best SMAPE:", study.best_value)
print("Best hyperparameters:", study.best_params)

# ========== Retrain Final Model with Best Params ==========
print("RETRAINING THE FINAL MODEL WITH BEST HYPERPARAMETERS...")
best_params = study.best_params

batch_size = best_params["batch_size"]
lr = best_params["lr"]
weight_decay = best_params["weight_decay"]
dropout = best_params["dropout"]
hidden_manual = best_params["hidden_manual"]
hidden_qwen = best_params["hidden_qwen"]
hidden_siglip = best_params["hidden_siglip"]
final_hidden = best_params["final_hidden"]
huber_delta = best_params["huber_delta"]

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

final_model = DeepWideEmbeddingMLP(
    len(manual_tfidf_cols), len(qwen_cols), len(siglip_cols),
    hidden_manual, hidden_qwen, hidden_siglip, final_hidden, dropout
).to(device)

criterion = nn.HuberLoss(delta=huber_delta)
optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=weight_decay)
scaler_amp = GradScaler()

# Train final model with more epochs
best_val_smape = float('inf')
patience_counter = 0
max_epochs = 100  # More epochs for final training

print("Training final model with best hyperparameters...")
for epoch in range(max_epochs):
    # Training phase
    final_model.train()
    train_loss = 0.0
    num_batches = 0
    
    for xb_m, xb_q, xb_i, yb in train_loader:
        xb_m, xb_q, xb_i, yb = xb_m.to(device), xb_q.to(device), xb_i.to(device), yb.to(device)
        optimizer.zero_grad()
        
        with autocast():
            pred = final_model(xb_m, xb_q, xb_i)
            loss = criterion(pred, yb)
        
        if torch.isnan(loss):
            continue
            
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=5.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        
        train_loss += loss.item()
        num_batches += 1

    avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0

    # Validation phase
    final_model.eval()
    preds, truths = [], []
    val_loss = 0.0
    val_batches = 0
    
    with torch.no_grad():
        for xb_m, xb_q, xb_i, yb in val_loader:
            xb_m, xb_q, xb_i, yb = xb_m.to(device), xb_q.to(device), xb_i.to(device), yb.to(device)
            
            with autocast():
                pred = final_model(xb_m, xb_q, xb_i)
                loss = criterion(pred, yb)
            
            preds.append(pred.cpu())
            truths.append(yb.cpu())
            val_loss += loss.item()
            val_batches += 1
    
    preds = torch.cat(preds, dim=0)
    truths = torch.cat(truths, dim=0)
    val_smape = smape(torch.expm1(truths), torch.expm1(preds))
    avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0

    print(f"Epoch {epoch+1:3d}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Val SMAPE: {val_smape:.4f}")

    # Early stopping
    if val_smape < best_val_smape:
        best_val_smape = val_smape
        patience_counter = 0
        # Save best model
        torch.save(final_model.state_dict(), 'optuna_best_siglip_qwen.pt')
        print(f"New best model saved with SMAPE: {best_val_smape:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= 10:  # More patience for final training
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model
final_model.load_state_dict(torch.load('optuna_best_siglip_qwen.pt'))
print(f"\nFinal model training completed. Best validation SMAPE: {best_val_smape:.4f}")

# Save the final model and scalers for inference
# torch.save({
#     'model_state_dict': final_model.state_dict(),
#     'best_params': best_params,
#     'scaler_manual': scaler_manual,
#     'scaler_qwen': scaler_qwen,
#     'scaler_siglip': scaler_siglip,
#     'manual_tfidf_cols': manual_tfidf_cols,
#     'qwen_cols': qwen_cols,
#     'siglip_cols': siglip_cols
# }, 'final_model_complete.pth')

# print("Final model and scalers saved to 'final_model_complete.pth'")
